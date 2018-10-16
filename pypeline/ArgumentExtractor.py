#!/usr/bin/python3

import os
import sys
import re
import codecs
import configparser
import pandas
import numpy
import random
import string
import spacy
import PCCParser
import utils
import dill as pickle
from collections import defaultdict
import time
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from nltk.parse import stanford
from nltk.tree import ParentedTree
import tensorflow as tf

class ArgumentExtractor:

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        #self.spacyPipeline = spacy.load(self.config['argumentExtractor']['spacyModel'])
        self.spacyPipeline = spacy.load('de')
        self.dim = 300

    def setGraph(self):
        self.graph = tf.get_default_graph()
        
    def run(self, parser, sentences, runtimeparsermemory, connectivepositions):

        relations = defaultdict(lambda : defaultdict(lambda : defaultdict()))
        rid = 1
        
        for cp in connectivepositions:
            sentence = sentences[cp[0]]
            connective = ' '.join(sentence.split()[cp[1][0]:cp[1][-1]+1])
            tokens = sentence.split()
            """
            ptree = None
            tree = self.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = ParentedTree.convert(t)
                break # always taking the first, assuming that this is the best scoring tree.
            """
            ptree = ParentedTree.convert(runtimeparsermemory[sentence])
            refconindex = cp[1][0]
            refcon = sentence.split()[refconindex]
            postag = utils.getPostagFromTree(ptree, refconindex)
            nodePosition = ptree.leaf_treeposition(refconindex)
            parent = ptree[nodePosition[:-1]].parent()
            pathToRoot = utils.getPathToRoot(parent, [])
            if pathToRoot[-1] == 'ROOT': # for some reason, the ones in training didn't have the final ROOT elem in there
                pathToRoot = pathToRoot[:-1]
            #row = [0, refcon, postag, refconindex, pathToRoot, cp[0]]
            #srow = [str(x) for x in row]
            nrow = []
            position_ohv = [0] * self.f2ohvlen[3]
            if str(refconindex) in self.f2ohvpos[3]:
                position_ohv[self.f2ohvpos[3][str(refconindex)]] = 1
            else:
                position_ohv[random.randint(0, len(position_ohv)-1)] = 1
            nrow += position_ohv
            rootroute_ohv = [0] * self.f2ohvlen[4]
            if str(pathToRoot) in self.f2ohvpos[4]:
                rootroute_ohv[self.f2ohvpos[4][str(pathToRoot)]] = 1
            else:
                mindist = 100
                val = None
                for route in self.f2ohvpos[4]:
                    dist = utils.levenshteinDistance(route, str(pathToRoot))
                    if dist < mindist:
                        mindist = dist
                        val = self.f2ohvpos[4][route]
                rootroute_ohv[val] = 1
            nrow += rootroute_ohv
            
            if refcon in parser.embd:
                for item in parser.embd[refcon]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            if postag in parser.posembd:
                for item in parser.posembd[postag]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            nrow.append('dummyLabel')
            nmatrix = [nrow]
            df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
            ds = df.values
            X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
            sentpospred = None
            samesentpred = None
            with self.graph.as_default():
                sentpospred = self.sentposclassifier.predict(X)
                samesentpred = self.samesentclassifier.predict(X)
            
            deptree = self.spacyPipeline(sentence)
            sid = cp[0]
            targetsid = sid - sentpospred[0]
            
            intargtokens = self.getIntArg(deptree, cp[1], tokens)
            relations[rid]['connective'] = cp
            relations[rid]['intarg'] = (sid, intargtokens)

            if targetsid in range(len(sentences)):
                extargtokens, targetsid = self.getExtArg(deptree, sid, targetsid, sentences, cp[1], tokens, samesentpred)
                # because punctuation is added manually above, it may overlap. Remove it from extarg if so:
                if sid == targetsid: # if in same sentence...
                    extargtokens = [x for x in extargtokens if not x in intargtokens]
                relations[rid]['extarg'] = (targetsid, extargtokens)
            # check out if there should be an else here (targetsid was out of range)
            rid += 1


        return relations
            
            
    def getExtArg(self, deptree, sid, targetsid, sentences, connectiveIndices, tokens, samesentpred):

        predictedTokenIds = [i for i, t in enumerate(sentences[targetsid].split())] # default. Modifying in next lines
        if sid == targetsid:
            for index, token in enumerate(deptree):
                if index == connectiveIndices[0] and token.text == tokens[connectiveIndices[0]]: # could be tokenisation differences between pre-tokenized and spacy. Figure out if there is some tokenisation flag that can be set to false in spacy
                    if index == 0:
                        negativeTokenIds = [x.i for x in token.head.subtree]
                        predictedTokenIds = [y for y, x in enumerate(deptree) if not y in negativeTokenIds]
                        # catch; if negativeTokenIds are all tokens in sentence, chances are sentpos classifier was off, let's assume previsous sent then:
                        if not predictedTokenIds and targetsid > 0:
                            predictedTokenIds = [i for i, t in enumerate(sentences[targetsid-1].split())]
                            targetsid = targetsid - 1
                    else:
                        if samesentpred[0] == 0:
                            prevverb = None
                            for ti in range(token.i, 0, -1):
                                if deptree[ti].pos_.lower().startswith('v'):
                                    prevverb = deptree[ti]
                                    break
                            if prevverb:
                                predictedTokenIds = [x.i for x in deptree[ti].head.subtree if x.i < token.i]
                            else:
                                predictedTokenIds = [x.i for x in deptree if x.i < token.i]
                        elif samesentpred[0] == 1:
                            nextverb = None
                            for ti in range(token.i, len(deptree)):
                                if deptree[ti].pos_.lower().startswith('v'):
                                    nextverb = deptree[ti]
                                    break
                            if nextverb:
                                predictedTokenIds = [x.i for x in deptree[ti].head.subtree if x.i > token.i]
                            else:
                                predictedTokenIds = [x.i for x in deptree if x.i > token.i]
        return predictedTokenIds, targetsid


    def getIntArg(self, deptree, connectiveIndices, tokens):

        reftoken = tokens[connectiveIndices[0]]
        refposition = connectiveIndices[0]
        predictedTokenIds = []
        for index, token in enumerate(deptree):
            if index == refposition and token.text == reftoken: # could be tokenisation differences between pre-tokenized and spacy. Figure out if there is some tokenisation flag that can be set to false in spacy
                predictedTokenIds = [x.i for x in token.head.subtree]
                # adding clause final punctuation:
                if predictedTokenIds[-1] < len(deptree)-1:
                    if deptree[predictedTokenIds[-1]+1].text in string.punctuation:
                        predictedTokenIds.append(predictedTokenIds[-1]+1)
                # exlcuding connective token(s):
                predictedTokenIds = [x for x in predictedTokenIds if not x in connectiveIndices]
                # only taking the right remaining string if the ref conn is a conjunction
                if token.pos_.lower().endswith('conj'):
                    predictedTokenIds = [x for x in predictedTokenIds if x > refposition]
        return sorted(predictedTokenIds)





    def train(self, parser, debugmode=False):

        connectivefiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['connectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['syntax']))
        tokenfiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['tokens']))
        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectors')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # not using the gold syntax, but this layer is needed to extract full sentences, as it's (I think) the only layer that knows about this.

        self.sent2tagged = pickle.load(codecs.open(parser.config['lexparser']['taggermemory'], 'rb')) # TODO: change this to using parser assigned tags (probably the same, but can't be sure)
        
        sentposmatrix, samesentmatrix = self.getFeatures(fdict)
        self.getf2ohvdicts(sentposmatrix) # sentpos and samesent features are identical (only the label differs)

        sentposlabels = []
        samesentlabels = []
        nmatrix = []
        for sentposrow, samesentrow in zip(sentposmatrix, samesentmatrix):
            nrow = []
            refconindex = sentposrow[3]
            position_ohv = [0] * self.f2ohvlen[3]
            if str(refconindex) in self.f2ohvpos[3]:
                position_ohv[self.f2ohvpos[3][str(refconindex)]] = 1
            else:
                position_ohv[random.randint(0, len(position_ohv)-1)] = 1
            nrow += position_ohv
            pathToRoot = sentposrow[4]
            rootroute_ohv = [0] * self.f2ohvlen[4]
            if str(pathToRoot) in self.f2ohvpos[4]:
                rootroute_ohv[self.f2ohvpos[4][str(pathToRoot)]] = 1
            else:
                mindist = 100
                val = None
                for route in self.f2ohvpos[4]:
                    dist = utils.levenshteinDistance(route, str(pathToRoot))
                    if dist < mindist:
                        mindist = dist
                        val = self.f2ohvpos[4][route]
                rootroute_ohv[val] = 1
            nrow += rootroute_ohv
            
            tok, pos = sentposrow[1], sentposrow[2]
            if tok in parser.embd:
                for item in parser.embd[tok]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            if pos in parser.posembd:
                for item in parser.posembd[pos]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
                    
            self.rowdim = len(nrow)
            nrow.append('dummyLabel') # struggling with dimensions. Fix when all is up and running.
            nmatrix.append(nrow)
            sentposlabels.append(sentposrow[-1])
            samesentlabels.append(samesentrow[-1])

        df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
        ds = df.values
        X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
        Y_sentpos = to_categorical(numpy.array(sentposlabels))
        Y_samesent = to_categorical(numpy.array(samesentlabels))

        self.keras_sentpos_outputdim = len(Y_sentpos[0])
        self.keras_samesent_outputdim = len(Y_samesent[0])

        seed = 6
        batch_size = 5
        epochs = 100
        verbosity = 1#0
        if debugmode:
            epochs = 1
            verbosity = 1
            sys.stderr.write('WARNING: Setting epochs at %i (debug mode)\n' % epochs)
            

        self.sentposclassifier = KerasClassifier(build_fn=self.create_sentposmodel, epochs=epochs, batch_size=batch_size)
        self.sentposclassifier.fit(X, Y_sentpos, verbose=verbosity)

        self.samesentclassifier = KerasClassifier(build_fn=self.create_samesentmodel, epochs=epochs, batch_size=batch_size)
        self.samesentclassifier.fit(X, Y_samesent, verbose=verbosity)

        
        
    def create_sentposmodel(self):
        hidden_dims = 250
        hidden_dims2 = 64
        model = Sequential()
        model.add(Dense(hidden_dims, input_dim=self.rowdim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.keras_sentpos_outputdim, activation='sigmoid')) # softmax/sigmoid
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def create_samesentmodel(self):
        hidden_dims = 250
        hidden_dims2 = 64
        model = Sequential()
        model.add(Dense(hidden_dims, input_dim=self.rowdim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.keras_samesent_outputdim, activation='sigmoid')) # softmax/sigmoid
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
        

    def getf2ohvdicts(self, fmatrix):
        
        self.f2ohvpos = defaultdict(lambda : defaultdict(int))
        f2 = defaultdict(set)
        self.f2ohvlen = defaultdict()
        self.knownvals = set()
        rowsize = 0
        for row in fmatrix:
            rowsize = len(row)
            for pos, val in enumerate(row):
                f2[pos].add(val)
                #self.knownvals.add(val)
        for i in f2:
            self.f2ohvlen[i] = len(f2[i])
            for c, i2 in enumerate(f2[i]):
                self.f2ohvpos[i][i2] = c
        pickle.dump(self.f2ohvpos, codecs.open(self.config['argumentExtractor']['f2ohvpos'], 'wb'))
        pickle.dump(self.f2ohvlen, codecs.open(self.config['argumentExtractor']['f2ohvlen'], 'wb'))
        pickle.dump(self.knownvals, codecs.open(self.config['argumentExtractor']['knownvals'], 'wb'))
        

    def getFeatures(self, fileversions):

        someId2extArgPosition = defaultdict(int)
        someId2connective = defaultdict(str)

        matrix = []
        samesentmatrix = []
        headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'class_label']# 'sentenceId', 'class_label']
        pos2column = {}
        for i2, j2 in enumerate(headers):
            if i2 > 0:
                pos2column[i2-1] = j2
        #matrix.append(headers)
        #samesentmatrix.append(headers)
        mid = 1
        connectiveSingleTokens = 0

        file2tokenlist = defaultdict(list)
        file2discourseRelations = defaultdict(list)
        for fno, name in enumerate(fileversions):
            tokenlist = PCCParser.parseConnectorFile(fileversions[name]['connectors'])
            tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
            #tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
            #tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
            file2tokenlist[name] = tokenlist
            file2discourseRelations[name] = PCCParser.discourseRelations
            rid2connsentid = defaultdict(set)
            rid2conn = defaultdict(list)
            rid2extargsentid = defaultdict(set)
            rid2conndt = defaultdict(list)
            rid2extargtokens = defaultdict(list)
            rid2intargtokens = defaultdict(list)

            for token in tokenlist:
                if token.segmentType == 'connective':
                    connectiveSingleTokens += 1
                    rid2connsentid[token.unitId].add(token.sentenceId)
                    rid2conn[token.unitId].append(token.token)
                    rid2conndt[token.unitId].append(token)
                elif token.segmentType == 'unit':
                    if token.intOrExt == 'ext':
                        rid2extargsentid[token.unitId].add(token.sentenceId)
                        rid2extargtokens[token.unitId].append(token)
                    elif token.intOrExt == 'int':
                        rid2intargtokens[token.unitId].append(token)
                for rid in token.embeddedUnits:
                    if token.embeddedUnits[rid] == 'ext':
                        rid2extargsentid[rid].add(token.sentenceId)

            for rid in rid2connsentid:

                c = ' '.join([dt.token for dt in rid2conndt[rid]])
                p = 0
                someId2connective[name + '_' + str(rid)] = ' '.join(rid2conn[rid])
                reldist = sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0]
                p = reldist
                someId2extArgPosition[name + '_' + str(rid)] = reldist # WARNING; not the same as p
                fullsent = rid2conndt[rid][0].fullSentence
                tagged = None
                if fullsent in self.sent2tagged:
                    tagged = self.sent2tagged[fullsent]
                else:
                    tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                    self.sent2tagged[fullsent] = tagged
                currpos = tagged[rid2conndt[rid][0].sentencePosition][1]
                nextpos = tagged[rid2conndt[rid][len(c.split())-1].sentencePosition + 1][1]

                row = [str(mid), c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, p]#rid2conndt[rid][0].sentenceId, p]
                srow = [str(r) for r in row]
                mid += 1
                matrix.append(srow)
                if p == 0:
                    sspos = 0 # default/extarg is before conn
                    if rid2extargtokens[rid]:
                        if rid2conndt[rid][-1].tokenId < rid2extargtokens[rid][0].tokenId:
                            sspos = 1
                        ssrow = [str(mid), c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, sspos]#rid2conndt[rid][0].sentenceId, sspos] # mid will not be consecutive, but think it is not really used anyway. And now at least it corresponds to the other matrix, should that come in handy some time.
                        sssrow = [str(s) for s in ssrow]
                        samesentmatrix.append(sssrow)
                        
        return matrix, samesentmatrix
