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
from sklearn.ensemble import RandomForestClassifier
import dill as pickle
import tensorflow as tf

class ConnectiveClassifier:

    mainclassdict = {'ADV':'adv', 'KON':'cco', 'APPR':'prep', 'KOUS':'csu', 'KOKOM':'cco', 'APPRART':'prep', 'KOUI':'csu'}
    
    def __init__(self):

        self.dimension = 300
        self.customEncoder = defaultdict(int)
        self.maxEncoderId = 1

    def getCategoricalLength(self, l):

        if l < 5:
            return 0
        elif l < 10:
            return 1
        elif l < 15:
            return 2
        elif l < 20:
            return 3
        elif l < 25:
            return 4
        else:
            return 5
        
    def isSInitial(self, token, ptree):

        for s in ptree.subtrees(lambda t: t.label().startswith('S')):
            if s.leaves()[0] == token: # sloppy coding; if a candidate occurs multiple times in a sentence, all its instances get True if this is true for one instance... (should not happen too often though)
                return True

        
    """
    # only needed for Keras classifier, currently using RandomForest one...
    def setGraph(self):
        self.graph = tf.get_default_graph()
    """

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
                    mainclass = self.mainclassdict[pcct.pos] if pcct.pos in self.mainclassdict else 'other'
                    catlen = self.getCategoricalLength(len(pcct.fullSentence.split())) # pcct.fullSentence is pre-tokenised (but one single string)
                    sinit = 1 if self.isSInitial(pcct.token, ptree) else 0
                    row = [mid] + features + [mainclass, catlen, sinit] + [pcct.isConnective]

                    #row = [mid] + features + [pcct.isConnective]
                    encoded = []
                    for x in row:
                        if x in self.customEncoder:
                            encoded.append(self.customEncoder[x])
                        else:
                            self.maxEncoderId += 1
                            encoded.append(self.maxEncoderId)
                            self.customEncoder[x] = self.maxEncoderId
                    matrix.append(encoded)
                    #row = [str(x) for x in row]
                    mid += 1
                    #matrix.append(row)
        #headers = ['id','token','pos','leftbigram','leftpos','leftposbigram','rightbigram','rightpos','rightposbigram','selfCategory','parentCategory','leftsiblingCategory','rightsiblingCategory','rightsiblingContainsVP','pathToRoot','compressedPath','class_label']
        headers = ['id','token','pos','leftbigram','leftpos','leftposbigram','rightbigram','rightpos','rightposbigram','selfCategory','parentCategory','leftsiblingCategory','rightsiblingCategory','rightsiblingContainsVP','pathToRoot','compressedPath', 'mainclass', 'sentencelength', 'sinitial','class_label']
        df = pandas.DataFrame(matrix, columns=headers)
        #self.d = defaultdict(LabelEncoder) # don't trust the LabelEncoder over different sessions, so implemented my own rudimentary one
        #fit = df.apply(lambda x: self.d[x.name].fit_transform(x))
        #df = df.apply(lambda x: self.d[x.name].transform(x))

        train_labels = df.class_label
        labels = list(set(train_labels))
        train_labels = numpy.array([labels.index(x) for x in train_labels])
        train_features = df.iloc[:,1:len(headers)-1]
        train_features = numpy.array(train_features)

        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(train_features, train_labels)
        sys.stderr.write('INFO: ConnectiveClassifier trained.\n')

                    
    """
    # this is the method to run for a Keras classifier. Didn't get that up and running in time for this DH demo. Revisit later.
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
        epochs = 10#100
        verbosity = 1#0
        if debugmode:
            epochs = 1
            verbosity = 1
            sys.stderr.write('WARNING: Setting epochs at %i (debug mode)\n' % epochs)
            
        self.classifier = KerasClassifier(build_fn=self.create_simple_model, epochs=epochs, batch_size=batch_size)
        self.classifier.fit(X, Y, verbose=verbosity)
    """

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
        
        # TODO: assuming pre-tokenized input (whitespace splittable, one sentence per line). Fix tokenization at some point. EDIT: fixed tokenisation at flaskcontroller level, because there I can ensure that tokenisation is the same as when the painting happens
        for sid, sentence in enumerate(sentences):
            tokens = sentence.split()
            ptree = None
            if sentence in memorymap:
                ptree = memorymap[sentence]
                sentenceFeatures = self.getVectorsForTree(ptree)
                for tid, token in enumerate(tokens):
                    if token.lower() in dimlextokens:
                        fulltextfound = False
                        # this is the catch for multiword ones. singleword ones are found by definition, mwus not always
                        # TODO: Does not work properly (or at all) for discontinuous ones
                        for fullconn in dimlextokens2fullversions[token.lower()]:
                            if re.search(fullconn.lower(), sentence.lower()):
                                fulltextfound = True
                        if fulltextfound:
                            features = ['0']
                            features.extend(sentenceFeatures[tid])
                            pos = features[2]
                            mainclass = self.mainclassdict[pos] if pos in self.mainclassdict else 'other'
                            catlen = self.getCategoricalLength(len(tokens))
                            sinit = 1 if self.isSInitial(token, ptree) else 0
                            features.append(mainclass)
                            features.append(catlen)
                            features.append(sinit)
                            encoded = []
                            for x in features:
                                if x in self.customEncoder:
                                    encoded.append(self.customEncoder[x])
                                else:
                                    self.maxEncoderId += 1
                                    encoded.append(self.maxEncoderId)
                                    self.customEncoder[x] = self.maxEncoderId
                            #features += [str(x) for x in sentenceFeatures[tid]]

                            df = pandas.DataFrame([encoded], columns=None)
                            #fit = df.apply(lambda x: self.d[x.name].fit_transform(x))
                            #df = df.apply(lambda x: self.d[x.name].transform(x))

                            test_features = df.iloc[:,1:]
                            test_features = numpy.array(test_features)

                            pred = self.classifier.predict(test_features)
                            #print('debug sent:', sentence)
                            #print('debug tok;', token)
                            #print('debug features:', features)
                            #print('debug pred:', pred)
                            if pred[0] == 1:
                                connectivePositions.append((sid, tid))

        connectivePositions = utils.mergePhrasalConnectives(connectivePositions)
    
        return connectivePositions
                        
    
    """
    # this is the method to run for a Keras classifier. Didn't get that up and running in time for this DH demo. Revisit later.
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
            #print('debugging sent:', sentence)
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
                    for fullconn in dimlextokens2fullversions[token.lower()]:
                        if re.search(fullconn.lower(), sentence.lower()):
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

                        pred = None
                        with self.graph.as_default():
                            pred = self.classifier.predict(numpy.array([nrow, ]))
                        print('debug cc sent:', sentence)
                        print('db tok:', tok)
                        print('db pred:', pred)
                        if pred[0] == 1:
                            connectivePositions.append((sid, tid))

        connectivePositions = utils.mergePhrasalConnectives(connectivePositions)
                
        return connectivePositions
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
