#!/usr/bin/python3

import os
import sys
import re
import codecs
import configparser
import pandas
import numpy
import random
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

class ArgumentExtractor:

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.spacyPipeline = spacy.load(self.config['argumentExtractor']['spacyModel'])
        self.dim = 300
        

    def run(self, sentences, connectivePositions):

        for cp in connectivePositions:
            sentence = sentences[cp[0]]
            connective = ' '.join(sentence.split()[cp[1][0]:cp[1][-1]+1])
            print('sent:', sentence)
            print('conn:', connective)
            tokens = sentence.split()
            """
            ptree = None
            tree = self.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = ParentedTree.convert(t)
                break # always taking the first, assuming that this is the best scoring tree.
            """
            ptree = ParentedTree.convert(self.runtimeparsermemory[sentence])
            refconindex = cp[1][0]
            refcon = sentence.split()[refconindex]
            print('refconindex:', refconindex)
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
            
            if refcon in self.embd:
                for item in self.embd[refcon]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            if postag in self.posembd:
                for item in self.posembd[postag]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            nrow.append('dummyLabel')
            nmatrix = [nrow]
            df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
            ds = df.values
            X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
            
            sentpospred = self.sentposclassifier.predict(X)
            sentpos2pred2 = self.sentposclassifier.predict(numpy.array([nrow[:-1],]))
            print('sentpospred:', sentpospred)
            print('argmax:', numpy.argmax(sentpospred))
            print('sentpospred2:', sentpospred)
            print('argmax2:', numpy.argmax(sentpospred))
            # same sent is only needed if sentpospred == 0...
            #samesentpred = self.samesentclassifier.predict(X)
            #print('samesentpred:', samesentpred)
            #print('argmax:', numpy.argmax(samesentpred))
        

            print('type input:', type(numpy.array([nrow[:-1],])))
            print('classifier type:', type(self.sentposclassifier))
            print('type output:', type(sentpos2pred2))





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
        verbosity = 0
        if debugmode:
            epochs = 1
            verbosity = 1
            sys.stderr.write('WARNING: Setting epochs at %i (debug mode)\n' % epochs)
            

        self.sentposclassifier = KerasClassifier(build_fn=self.create_sentposmodel, epochs=epochs, batch_size=batch_size)
        self.sentposclassifier.fit(X, Y_sentpos, verbose=verbosity)

        self.samesentclassifier = KerasClassifier(build_fn=self.create_samesentmodel, epochs=epochs, batch_size=batch_size)
        self.samesentclassifier.fit(X, Y_samesent, verbose=verbosity)

        

    """

            
    def trainPositionClassifiers(self):

        #connectivefiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['standoffConnectives']))
        connectivefiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['connectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['syntax']))
        tokenfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['tokens']))
        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectors')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # not using the gold syntax, but this layer is needed to extract full sentences, as it's (I think) the only layer that knows about this.

        self.sent2tagged = pickle.load(codecs.open(self.config['lexparser']['taggermemory'], 'rb'))
        
        sentposmatrix, samesentmatrix = self.getFeatures(fdict)
        self.getf2ohvdicts(sentposmatrix) # think in this respect sentpos and samesent are identical (only the label differs)

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
            if tok in self.embd:
                for item in self.embd[tok]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dim))):
                    nrow.append(item)
            if pos in self.posembd:
                for item in self.posembd[pos]:
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
        epochs = 20#100

        self.sentposclassifier = KerasClassifier(build_fn=self.create_sentposmodel, epochs=epochs, batch_size=batch_size)
        self.sentposclassifier.fit(X, Y_sentpos, verbose=0)

        self.samesentclassifier = KerasClassifier(build_fn=self.create_samesentmodel, epochs=epochs, batch_size=batch_size)
        self.samesentclassifier.fit(X, Y_samesent, verbose=0)

        #debugging section
        #predictions = sentposclassifier.predict(X)
        #for index in range(len(predictions)):
            #print('pred debug:', predictions[index], sentposlabels[index])

        print('WARNING: NOT STORING TRAINED CLASSIFIERS!')
        
        sentpos_json = sentposclassifier.model.to_json()
        with codecs.open(self.config['argumentExtractor']['sentposmodel'], 'w') as jsout:
            jsout.write(sentpos_json)
            sentposclassifier.model.save_weights(self.config['argumentExtractor']['sentposweights'], overwrite=True)
        sys.stdout.write('INFO: Saved sentence position classifier to %s\n' % self.config['argumentExtractor']['sentposmodel'])

        samesent_json = samesentclassifier.model.to_json()
        with codecs.open(self.config['argumentExtractor']['samesentmodel'], 'w') as jsout:
            jsout.write(samesent_json)
            samesentclassifier.model.save_weights(self.config['argumentExtractor']['samesentweights'], overwrite=True)
        sys.stdout.write('INFO: Saved same sentence classifier to %s\n' % self.config['argumentExtractor']['samesentmodel'])            
        
    """
        
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
