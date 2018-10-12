#!/usr/bin/python3

import utils
import os
import DimLexParser
import PCCParser
import configparser
import sys
import re
import random
import codecs
from nltk.parse import stanford
from nltk.tree import ParentedTree
import pandas
import numpy
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import dill as pickle

class ConnectiveClassifier:

    def __init__(self):

        self.dimension = 300

    #TODO: build in evaluate method

    def train(self, parser, debugmode=False):

        connectivefiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['standoffConnectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['syntax']))

        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectors')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # not using the gold syntax, but this layer is needed to extract full sentences, as it's (I think) the only layer that knows about this.
        parsermemorymap = {}
        if os.path.exists(parser.config['lexparser']['memorymap']):
            parsermemorymap = pickle.load(codecs.open(parser.config['lexparser']['memorymap'], 'rb'))
            sys.stdout.write('INFO: Loaded parse trees from %s\n' % parser.config['lexparser']['memorymap'])
        file2pccTokens = {}
        candidates = set()
        for basename in fdict:
            pccTokens, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fdict[basename]['connectors'])
            pccTokens = PCCParser.parseSyntaxFile(fdict[basename]['syntax'], pccTokens)
            file2pccTokens[basename] = pccTokens
            for pcct in pccTokens:
                if pcct.isConnective:
                    candidates.add(pcct.token)
        #TODO: current setup is single token based, which is stupid. Revisit this.
        matrix = []
        mid = 0
        for f, pccTokens in file2pccTokens.items():
            for index, pcct in enumerate(pccTokens):
                if pcct.token in candidates:
                    sentence = pcct.fullSentence
                    tokens = utils.filterTokens(sentence.split())
                    ptree = None
                    if sentence in parsermemorymap:
                        ptree = parsermemorymap[sentence]
                    else:
                        tree = parser.lexParser.parse(tokens)
                        ptreeiter = ParentedTree.convert(tree)
                        for t in ptreeiter:
                            ptree = t
                            break # always taking the first, assuming that this is the best scoring tree.
                        parsermemorymap[sentence] = ptree
                    features = self.getFeaturesFromTree(index, pccTokens, pcct, ptree)
                    row = [mid] + features + [pcct.isConnective]
                    row = [str(x) for x in row]
                    mid += 1
                    matrix.append(row)
        self.getf2ohvpos(matrix)
        nmatrix = []
        labels = []
        for row in matrix:
            nrow = []
            for fi, feat in enumerate(row[3:-1]):
                ohv = [0] * self.feature2ohvlength[fi+2]
                ohv[self.feature2ohvpos[fi+2][feat]] = 1
                nrow += ohv
            tok, pos = row[1], row[2]
            if tok in parser.embd:
                for item in parser.embd[tok]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dimension))):
                    nrow.append(item)
            if pos in parser.posembd:
                for item in parser.posembd[pos]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, self.dimension))):
                    nrow.append(item)
            self.rowdim = len(nrow)
            label = 0
            if row[-1] == 'True':
                label = 1
            nrow.append(label)
            labels.append(label) # redundancy to add it in nmatrix, but was struggling with dimensions. Check later if I can take it out of nrow
            nmatrix.append(nrow)

        df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
        ds = df.values
        X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
        Y = to_categorical(numpy.array(labels))
        self.keras_output_dim = len(Y[0])
        
        seed = 6
        batch_size = 5
        epochs = 100
        verbosity = 0
        if debugmode:
            epochs = 1
            verbosity = 1
            sys.stderr.write('WARNING: Setting epochs at %i (debug mode)\n' % epochs)
            
        self.classifier = KerasClassifier(build_fn=self.create_simple_model, epochs=epochs, batch_size=batch_size)
        self.classifier.fit(X, Y, verbose=verbosity)


    def create_simple_model(self):

        hidden_dims = 250
        hidden_dims2 = 64
        model = Sequential()
        model.add(Dense(hidden_dims, input_dim=self.rowdim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hidden_dims2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.keras_output_dim, activation='sigmoid')) # softmax/sigmoid
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def run(self, parser, sentences, memorymap):

        connectivePositions = []
        dimlexconnectives = [conn.word for conn in DimLexParser.parseXML(parser.config['dimlex']['dimlexlocation'])]
        dimlextokens = set()
        dimlextokens2fullversions = defaultdict(set)
        for dc in dimlexconnectives:
            for tok in dc.split():
                if not re.match('^\W+$', tok):
                    dimlextokens.add(tok)
                    dimlextokens2fullversions[tok].add(dc)

        # TODO: assuming pre-tokenized input (whitespace splittable, one sentence per line). Fix tokenization at some point
        for sid, sentence in enumerate(sentences):
            print('debugging sent:', sentence)
            tokens = sentence.split()
            ptree = None
            ptree = memorymap[sentence]
            sentenceFeatures = self.getVectorsForTree(ptree)
            for tid, token in enumerate(tokens):
                if token.lower() in dimlextokens:
                    fulltextfound = False
                    # this is the catch for multiword ones. singleword ones are found by definition, mwus not always
                    # for full accuracy, should also take pos of token in sent into account, but this should be close enough for now
                    # TODO: Does not work properly (or at all) for discontinuous ones
                    for fullconn in dimlextokens2fullversions[token]:
                        if re.search(fullconn.lower(), sentence):
                            fulltextfound = True
                    if fulltextfound:
                        features = ['dummyId']
                        features += [str(x) for x in sentenceFeatures[tid]]
                        nrow = []
                        for fi, feat in enumerate(features[3:]):
                            ohv = [0] * self.feature2ohvlength[fi+2]
                            if feat in self.feature2ohvpos[fi+2]:
                                ohv[self.feature2ohvpos[fi+2][feat]] = 1
                            else:
                                ohv[random.randint(0, len(ohv)-1)] = 1
                            nrow += ohv
                        tok, pos = features[1], features[2]
                        if tok in parser.embd:
                            for item in parser.embd[tok]:
                                nrow.append(item)
                        else:
                            for item in numpy.ndarray.flatten(numpy.random.random((1, self.dimension))):
                                nrow.append(item)
                        if pos in parser.posembd:
                            for item in parser.posembd[pos]:
                                nrow.append(item)
                        else:
                            for item in numpy.ndarray.flatten(numpy.random.random((1, self.dimension))):
                                nrow.append(item)

                        pred = self.classifier.predict(numpy.array([nrow, ]))

                        print('db tok:', tok)
                        print('db pred:', pred)
                        #print('db predval:', predval)
                        if pred[0] == 1:
                            connectivePositions.append((sid, tid))

        connectivePositions = utils.mergePhrasalConnectives(connectivePositions)
                
        return connectivePositions

    

    """
    def run(self, inp):

        connectivePositions = []
        dimlexconnectives = [conn.word for conn in DimLexParser.parseXML(self.config['dimlex']['dimlexlocation'])]
        dimlextokens = set()
        dimlextokens2fullversions = defaultdict(set)
        for dc in dimlexconnectives:
            for tok in dc.split():
                if not re.match('^\W+$', tok):
                    dimlextokens.add(tok)
                    dimlextokens2fullversions[tok].add(dc)

        # TODO: assuming pre-tokenized input (whitespace splittable, one sentence per line). Fix tokenization at some point
        for sid, sentence in enumerate(inp):
            #print('debugging sent:', sentence)
            tokens = sentence.split()
            ptree = None
            tree = self.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break # always taking the first, assuming that this is the best scoring tree.
            ptree = self.runtimeparsermemory[sentence]
            sentenceFeatures = self.getVectorsForTree(ptree)
            headers = ['id'] + [v for k, v in self.pos2column.items()][:-1]
            for tid, token in enumerate(tokens):
                if token in dimlextokens:
                    fulltextfound = False
                    # this is the catch for multiword ones. singleword ones are found by definition, mwus not always
                    # for full accuracy, should also take pos of token in sent into account, but this should be close enough for now
                    # TODO: Does not work properly (or at all) for discontinuous ones
                    for fullconn in dimlextokens2fullversions[token]:
                        if re.search(fullconn, sentence):
                            fulltextfound = True
                    if fulltextfound:
                        features = [str(j) for j in sentenceFeatures[tid]]
                        intfeatures = []
                        for feat in features:
                            if feat in self.customLabelEncoder: # doing this myself, instead of with sklearn LabelEncoder, since pickling didn't really work
                                intfeatures.append(self.customLabelEncoder[feat])
                            else:
                                self.clmid += 1
                                intfeatures.append(self.clmid)
                                self.customLabelEncoder[feat] = self.clmid
                        features = [0] + intfeatures
                        tdf = pandas.DataFrame([features], columns=headers)
                        X = tdf.iloc[:,1:len(headers)]
                        X = numpy.array(X)
                        pred = self.classifier.predict(X)
                        if pred[0] == 1:
                            connectivePositions.append((sid, tid))

        # filtering for multiword ones
        connectivePositions = utils.mergePhrasalConnectives(connectivePositions)
                            
        return connectivePositions
    """
    """
    def train(self):

        
        
        connectivefiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['standoffConnectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['syntax']))

        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectors')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # not using the gold syntax, but this layer is needed to extract full sentences, as it's (I think) the only layer that knows about this.
        parsermemorymap = {}
        if os.path.exists(self.config['lexparser']['memorymap']):
            parsermemorymap = pickle.load(codecs.open(self.config['lexparser']['memorymap'], 'rb'))
            sys.stdout.write('INFO: Loaded parse trees from %s\n' % self.config['lexparser']['memorymap'])
        file2pccTokens = {}
        candidates = set()
        for basename in fdict:
            pccTokens, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fdict[basename]['connectors'])
            pccTokens = PCCParser.parseSyntaxFile(fdict[basename]['syntax'], pccTokens)
            file2pccTokens[basename] = pccTokens
            for pcct in pccTokens:
                if pcct.isConnective:
                    candidates.add(pcct.token)
        #TODO: current setup is single token based, which is stupid. Revisit this.
        matrix = []
        headers = ['id'] + [v for k, v in self.pos2column.items()]
        #matrix.append(headers)
        matrix.append([0] * len(headers)) # dummy to get shapes right, look into this
        mid = 0
        customLabelEncoder = {True:1, False:0}
        clmid = 2
        for f, pccTokens in file2pccTokens.items():
            for index, pcct in enumerate(pccTokens):
                if pcct.token in candidates:
                    sentence = pcct.fullSentence
                    tokens = utils.filterTokens(sentence.split())
                    ptree = None
                    if sentence in parsermemorymap:
                        ptree = parsermemorymap[sentence]
                    else:
                        tree = self.lexParser.parse(tokens)
                        ptreeiter = ParentedTree.convert(tree)
                        for t in ptreeiter:
                            ptree = t
                            break # always taking the first, assuming that this is the best scoring tree.
                        parsermemorymap[sentence] = ptree
                    features = self.getFeaturesFromTree(index, pccTokens, pcct, ptree)
                    intfeatures = []
                    for feat in features:
                        if feat in customLabelEncoder: # doing this myself, instead of with sklearn LabelEncoder, since pickling didn't really work
                            intfeatures.append(customLabelEncoder[feat])
                        else:
                            clmid += 1
                            intfeatures.append(clmid)
                            customLabelEncoder[feat] = clmid
                    row = [mid] + intfeatures + [pcct.isConnective]
                    mid += 1
                    matrix.append(row)

        df = pandas.DataFrame(matrix, columns=headers)
        #d = defaultdict(LabelEncoder)
        #fit = df.apply(lambda x: d[x.name].fit_transform(x))
        #df = df.apply(lambda x: d[x.name].transform(x))
        Y = df.class_label
        l = list(set(Y))
        Y = numpy.array([l.index(x) for x in Y])
        X = df.iloc[:,1:len(headers)-1]
        X = numpy.array(X)

        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X, Y)
        
        joblib.dump(classifier, self.config['connectiveClassifier']['modelLocation']) 
        sys.stdout.write('INFO: Pickled classifier to %s\n' % self.config['connectiveClassifier']['modelLocation'])

        pickle.dump(customLabelEncoder, codecs.open(self.config['connectiveClassifier']['labelEncoder'], 'wb'))
        sys.stdout.write('INFO: Pickled encoder to %s\n' % self.config['connectiveClassifier']['labelEncoder'])
    """
        
    def getf2ohvpos(self, fmatrix):

        self.feature2ohvpos = defaultdict(lambda : defaultdict(int))
        f2 = defaultdict(set)
        self.feature2ohvlength = defaultdict()
        rowsize = 0
        for row in fmatrix:
            rowsize = len(row)
            for pos, val in enumerate(row):
                f2[pos].add(val)
        for i in f2:
            self.feature2ohvlength[i] = len(f2[i])
            for c, i2 in enumerate(f2[i]):
                self.feature2ohvpos[i][i2] = c
    
    def getFeaturesFromTree(self, index, pccTokens, pcct, pt):

        treevector = self.getVectorsForTree(pt)
        matches = [x for x in treevector if x[0] == pcct.token]
        if len(matches) > 1:
            matches = utils.narrowMatches(matches, pcct, pccTokens, index)
            return matches[0]
        elif len(matches) == 1:
            return matches[0]

    def getVectorsForTree(self, tree):

        treeVectors = []
        parentedTree = ParentedTree.convert(tree)
        for i, node in enumerate(parentedTree.pos()):
            features = []
            currWord = node[0]
            currPos = node[1]
            features.append(currWord)
            features.append(currPos)
            ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
            rn = "EOS" if i == len(parentedTree.pos())-1 else parentedTree.pos()[i+1]
            lpos = "_" if ln == "SOS" else ln[1]
            rpos = "_" if rn == "EOS" else rn[1]
            lstr = ln if ln == "SOS" else ln[0]
            rstr = rn if rn == "EOS" else rn[0]
            lbigram = lstr + '_' + currWord
            rbigram = currWord + '_' + rstr
            lposbigram = lpos + '_' + currPos
            rposbigram = currPos + '_' + rpos
            features.append(lbigram)
            features.append(lpos)
            features.append(lposbigram)
            features.append(rbigram)
            features.append(rpos)
            features.append(rposbigram)
            selfcat = currPos # always POS for single words
            features.append(selfcat)
            nodePosition = parentedTree.leaf_treeposition(i)
            parent = parentedTree[nodePosition[:-1]].parent()
            parentCategory = parent.label()
            features.append(parentCategory)
            ls = parent.left_sibling()
            lsCat = False if not ls else ls.label()
            rs = parent.right_sibling()
            rsCat = False if not rs else rs.label()
            features.append(lsCat)
            features.append(rsCat)
            rsContainsVP = False
            if rs:
                if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                    rsContainsVP = True
            features.append(rsContainsVP)
            rootRoute = utils.getPathToRoot(parent, [])
            features.append('_'.join(rootRoute))
            cRoute = utils.compressRoute([x for x in rootRoute])
            features.append('_'.join(cRoute))
            treeVectors.append(features)

        return treeVectors
