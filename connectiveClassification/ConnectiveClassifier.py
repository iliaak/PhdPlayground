#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from nltk.parse import stanford
from nltk import sent_tokenize
from nltk import Tree
from nltk.tree import ParentedTree
from nltk import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier
from nltk import classify
import csv
import importlib.util
import configparser
import pickle
import random

"""
Class for binary classification of words (connective or no connective), features based on Lin et al. 2014.
Contains several additional functions for training, pickling, unpickling a previously trained classifier and evaluation methods.
Arguments to this script (when ran stand-alone) should be pretty self-explanatory looking at the optionParser arguments, but are bascially:
- folder with annotated data, currently PCC (xml inline) and CONLL (typical tab-separated style) formats are supported.
- language, currently de and en are supported.
- verbosity flag for more information
- format of input data (conll or pcc)

To run this successfully, make sure that the required packages are installed and file paths are correct. This is taken from the settings.conf file, which should be in the same location as this script. 
TODO: document settings.conf a bit!

"""
verbose = False

memoryMap = defaultdict(list)

class ConnectiveClassifier():

    def __init__(self, alg, lang):

        self.matrix = defaultdict(list)
        self.maxId = 0
        self.sanityList = []
        self.classifier = None
        self.dictionary = set()
        self.alg = alg

        # read settings from config file
        Config = configparser.ConfigParser()
        scriptLocation = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        Config.read(os.path.join(scriptLocation, 'settings.conf'));
        os.environ['JAVAHOME'] = Config.get('JAVA', 'JAVAHOME')
        os.environ['STANFORD_PARSER'] = Config.get('CORENLP', 'STANFORD_PARSER')
        os.environ['STANFORD_MODELS'] = Config.get('CORENLP', 'STANFORD_MODELS')
        lexParserPath = ""
        if lang == 'de':
            lexParserPath = Config.get('CORENLP', 'LEXPARSER_DE')
        elif lang == 'en':
            lexParserPath = Config.get('CORENLP', 'LEXPARSER_EN')
        else:
            sys.stderr.write("ERROR: Language '%s' not supported. Please use one of the supported languages.\n" % lang)
            sys.exit(1)
        self.lexParser = stanford.StanfordParser(model_path= lexParserPath)
        pccPath = Config.get('MISC', 'PCCPARSER')
        pccLoc = importlib.util.spec_from_file_location("PCCParser", pccPath)
        self.PCCParser = importlib.util.module_from_spec(pccLoc)
        pccLoc.loader.exec_module(self.PCCParser)
        conllPath = Config.get('MISC', 'CONLLPARSER')
        conllLoc = importlib.util.spec_from_file_location('CONLLPARSER', conllPath)
        self.CONLLParser = importlib.util.module_from_spec(conllLoc)
        conllLoc.loader.exec_module(self.CONLLParser)
        self.index2FeatureName = {
            0:'id',
            1:'token',
            2:'pos',
            3:'leftbigram',
            4:'leftpos',
            5:'leftposbigram',
            6:'rightbigram',
            7:'rightpos',
            8:'rightposbigram',
            9:'parentCategory',
            10:'leftSiblingCategory',
            11:'rightSiblingCategory',
            12:'rightSiblingContainsVP',
            #TODO: rightSiblingContainsTrace
            13:'pathToRoot',
            14:'compressedPathToRoot',
            15:'isConnective'
        }

    def matrixRow2fDict(self, row):
        
        fDict = {}
        for i, feature in enumerate(row[:len(row)-1]):
            fDict[self.index2FeatureName[i+1]] = feature
        return fDict

    def buildFeatureMatrixFromCONLL(self, connectiveFiles):

        for i, conllFile in enumerate(connectiveFiles):
            if verbose:
                sys.stderr.write("INFO: Parsing file .../%s (%s/%s).\n" % (conllFile.split('/')[-1], str(i+1), str(len(connectiveFiles))))
            conllTokens = self.CONLLParser.parsePDTBFile(conllFile)
            self.getFeaturesForCONLLTokens(conllTokens)

        matrix = []
        for instance in self.matrix:
            row = self.matrix[instance]
            fDict = {}
            fDict = self.matrixRow2fDict(row)
            label = row[len(row)-1]
            matrix.append((fDict, label))

        if self.alg == 'NaiveBayes':
            self.classifier = NaiveBayesClassifier.train(matrix)
        elif self.alg == 'DecisionTree':
            self.classifier = DecisionTreeClassifier.train(matrix)
        elif self.alg == 'Maxent':
            self.classifier = MaxentClassifier.train(matrix)
        else:
            sys.stderr.write("ERROR: Algorithm '%s' not supported. Please pick one from 'NaiveBayes', 'DecisionTree' or 'Maxent'.\n")
    
    def buildFeatureMatrixFromPCC(self, connectiveFiles):

        for i, cxml in enumerate(connectiveFiles):
            if verbose:
                sys.stderr.write("INFO: Parsing file .../%s (%s/%s).\n" % (cxml.split('/')[-1], str(i+1), str(len(connectiveFiles))))
            pccTokens = self.PCCParser.parseConnectorFile(cxml)
            self.getFeaturesForPCCTokens(pccTokens)

        matrix = []
        for instance in self.matrix:
            row = self.matrix[instance]
            fDict = {}
            fDict = self.matrixRow2fDict(row)
            label = row[len(row)-1]
            matrix.append((fDict, label))

        if self.alg == 'NaiveBayes':
            self.classifier = NaiveBayesClassifier.train(matrix)
        elif self.alg == 'DecisionTree':
            self.classifier = DecisionTreeClassifier.train(matrix)
        elif self.alg == 'Maxent':
            self.classifier = MaxentClassifier.train(matrix)
        else:
            sys.stderr.write("ERROR: Algorithm '%s' not supported. Please pick one from 'NaiveBayes', 'DecisionTree' or 'Maxent'.\n")

    def getFeaturesForTokenList(self, tokenTupleList):

        # in this case, list is not assumed to be pccTokens, but a list of tuples, word first, true class second
       
        localId2class = defaultdict(bool)
        filteredTokens = self.filterTokenTuples(tokenTupleList)
        for i, t in enumerate(filteredTokens):
            if t[1]:
                localId2class[i] = True
            else:
                localId2class[i] = False

        sentences = sent_tokenize(' '.join([re.sub(r'[()]+', '', t[0]) for t in filteredTokens])) # have to do the same ugly fix here because nltk.tree messed up any word with parenthesis in it
        tokenizedSents = [sent.split() for sent in sentences]
        try:
            forest = self.lexParser.parse_sents(tokenizedSents)
            tokenId = 0
            for trees in forest:
                for tree in trees:
                    featureVectors = self.getVectorsForTree(tree)
                    for fv in featureVectors:
                        nfv = list(fv)
                        nfv.append(localId2class[tokenId])
                        if (localId2class[tokenId]):
                            self.sanityList.append(nfv[0])
                        self.matrix[self.maxId] = nfv
                        self.maxId += 1
                        tokenId += 1
                    
        except ValueError:
            sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")
            """
            # Got the following error for some file, coming from deep down the nltk parser somewhere...
            ValueError: Tree.read(): expected ')' but got 'end-of-string'
                at index 121.
                    "...JD 2.)))))"
            """


    def getFeaturesForCONLLTokens(self, conllTokens):

        localId2class = defaultdict(bool)
        # ugly fix: filtering for parenthesis, as these are skipped by nltk.Tree traversal (probably because this symbol is used in internal representation)
        conllTokens = self.filterTokens(conllTokens)
        for i, t in enumerate(conllTokens):
            if t.isConnective:
                localId2class[i] = True
                self.dictionary.add(t.token)
            else:
                localId2class[i] = False
        
        sentences = sent_tokenize(' '.join([re.sub(r'[()]+', '', t.token) for t in conllTokens]))
        tokenizedSents = [sent.split() for sent in sentences]
        try:
            forest = self.lexParser.parse_sents(tokenizedSents)
            tokenId = 0
            for trees in forest:
                for tree in trees:
                    featureVectors = self.getVectorsForTree(tree)
                    for fv in featureVectors:
                        nfv = list(fv)
                        nfv.append(localId2class[tokenId])
                        if (localId2class[tokenId]):
                            self.sanityList.append(nfv[0])
                        self.matrix[self.maxId] = nfv
                        self.maxId += 1
                        tokenId += 1
                    
        except ValueError:
            sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")
            """
            # Got the following error for some file, coming from deep down the nltk parser somewhere...
            ValueError: Tree.read(): expected ')' but got 'end-of-string'
                at index 121.
                    "...JD 2.)))))"
            """

        
            
    def getFeaturesForPCCTokens(self, pccTokens):        

        # NOTE: The bookkeeping here is a bit complicated, due to the conano inline xml format. Need to keep track of token id's to know if they are connective or not, and link this to the current node in the tree
        localId2class = defaultdict(bool)
        # ugly fix: filtering for parenthesis, as these are skipped by nltk.Tree traversal (probably because this symbol is used in internal representation)
        pccTokens = self.filterTokens(pccTokens)
        for i, t in enumerate(pccTokens):
            if t.isConnective:
                localId2class[i] = True
                self.dictionary.add(t.token)
            else:
                localId2class[i] = False
        
        sentences = sent_tokenize(' '.join([re.sub(r'[()]+', '', t.token) for t in pccTokens])) # have to do the same ugly fix here because nltk.tree messed up any word with parenthesis in it
        tokenizedSents = [sent.split() for sent in sentences]
        try:
            forest = self.lexParser.parse_sents(tokenizedSents)
            tokenId = 0
            for trees in forest:
                for tree in trees:
                    featureVectors = self.getVectorsForTree(tree)
                    for fv in featureVectors:
                        nfv = list(fv)
                        nfv.append(localId2class[tokenId])
                        if (localId2class[tokenId]):
                            self.sanityList.append(nfv[0])
                        self.matrix[self.maxId] = nfv
                        self.maxId += 1
                        tokenId += 1
                    
        except ValueError:
            sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")
            """
            # Got the following error for some file, coming from deep down the nltk parser somewhere...
            ValueError: Tree.read(): expected ')' but got 'end-of-string'
                at index 121.
                    "...JD 2.)))))"
            """

        
    
    def getVectorsForTree(self, tree):

        treeVectors = [] # returning a list of lists, as it takes a whole tree, and generates vectors for every word/node in the tree
        """
        Features per word (in order):
        - current word
        - pos tag of current word
        - previous word + current word
        - pos tag of previous word
        - previous word pos tag + current word pos tag
        - current word + next word
        - pos tag of next word
        - current word pos tag + next word pos tag
        - category of parent (in tree)
        - left sibling category (False if no left sibling)
        - right sibling category (False if no right sibling)
        - True if right sibling contains VP (False otherwise)
        # TODO: insert containsTrace here
        - path to root node (in categories)
        - compressed path to root node (in categories); identical adjacent categories compressed into one
        """

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

            # TODO: self-category is not clear to me, if not always POS (because we are dealing with single words here in binary classification), what else should it be?
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
            # TODO: Figure out how to check if rs contains a trace (given the tree/grammar)
            features.append(rsContainsVP)
            #featureList.append(rsContainsTrace) # TODO
            rootRoute = self.getPathToRoot(parent, [])
            features.append('_'.join(rootRoute))
            cRoute = self.compressRoute([x for x in rootRoute])
            features.append('_'.join(cRoute))

            treeVectors.append(features)

        return treeVectors

    def compressRoute(self, r): # filtering out adjacent identical tags
        
        delVal = "__DELETE__"
        for i in range(len(r)-1):
            if r[i] == r[i+1]:
                r[i+1] = delVal
        return [x for x in r if x != delVal]

    def getPathToRoot(self, ptree, route):

        if ptree.parent() == None:
            route.append(ptree.label())
            return route
        else:
            route.append(ptree.label())
            self.getPathToRoot(ptree.parent(), route)
        return route

    def filterTokens(self, tokens):
        skipSet = ['(', ')']
        return [t for t in tokens if not t.token in skipSet]

    def filterTokenTuples(self, tuples):
        skipSet = ['(', ')']
        return [t for t in tuples if not t[0] in skipSet]
        

    def writeMatrix(self, outputfilename):

        with open(outputfilename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames = [v for k, v in self.index2FeatureName.items()])
            #writer.writeheader()
            for instance in self.matrix:
                fVals = self.matrix[instance]
                writer.writerow({self.index2FeatureName[0]: instance,
                                 self.index2FeatureName[1]: fVals[0],
                                 self.index2FeatureName[2]: fVals[1],
                                 self.index2FeatureName[3]: fVals[2],
                                 self.index2FeatureName[4]: fVals[3],
                                 self.index2FeatureName[5]: fVals[4],
                                 self.index2FeatureName[6]: fVals[5],
                                 self.index2FeatureName[7]: fVals[6],
                                 self.index2FeatureName[8]: fVals[7],
                                 self.index2FeatureName[9]: fVals[8],
                                 self.index2FeatureName[10]: fVals[9],
                                 self.index2FeatureName[11]: fVals[10],
                                 self.index2FeatureName[12]: fVals[11],
                                 self.index2FeatureName[13]: fVals[12],
                                 self.index2FeatureName[14]: fVals[13],
                                 self.index2FeatureName[15]: fVals[14]})

        sys.stderr.write("INFO: csv output written to: %s.\n" % outputfilename)

    def readMatrix(self, csvFile):

        matrix =  []
        for line in codecs.open(csvFile, 'r').readlines():
            matrix.append(line.split(','))
        return matrix
                        

    def parseFeatureMatrixFromCSV(self, csvFile):

        matrix = []
        with codecs.open(csvFile, 'r') as fm:
            fieldnames = [v for k, v in self.index2FeatureName.items()]
            reader = csv.DictReader(fm, fieldnames=fieldnames)
            for row in reader:
                fDict = {}
                for fn in fieldnames:
                    fDict[fn] = row[fn]
                label = row['isConnective']
                matrix.append((fDict, label))

        return matrix
        
    def trainClassifierFromCSV(self, csvFile, alg):

        matrix = self.parseFeatureMatrixFromCSV(csvFile)
        if alg == 'NaiveBayes':
            self.classifier = NaiveBayesClassifier.train(matrix)
        elif alg == 'Maxent':
            self.classifier = MaxentClassifier.train(matrix)
        elif alg == 'DecisionTree':
            self.classifier = DecisionTreeClassifier.train(matrix)
        else:
            sys.stderr.write("ERROR: Algorithm '%s' not supported. Dying now.\n")
            sys.exit(1)

    def randomCrossValidate(self, csvFile, alg):

        import random
        matrix = self.parseFeatureMatrixFromCSV(csvFile)
        x = 10
        scores = []    
        for i in range(x):
            random.shuffle(matrix)
            si = len(matrix) / 10
            testSet = matrix[:int(si)]
            trainSet = matrix[int(si):]
            classifier = None
            if alg == 'NaiveBayes':
                classifier = NaiveBayesClassifier.train(trainset)
            elif alg == 'Maxent':
                classifier = MaxentClassifier.train(trainset)
            elif alg == 'DecisionTree':
                classifier = DecisionTreeClassifier.train(trainset)
            else:
                sys.stderr.write("ERROR: Algorithm '%s' not supported. Dying now.\n")
                sys.exit(1)

            accuracy = classify.accuracy(classifier, testSet)
            scores.append(accuracy)
            
        avg = float(sum(scores) / x)
        sys.stderr.write("INFO: Average score over random %i-fold validation: %f.\n" % (x, avg))



    def traditionalCrossValidate(self, csvFile, alg):

        matrix = self.parseFeatureMatrixFromCSV(csvFile)
        x = 10
        j = 0
        scores = []
        for i in range(x):
            si = len(matrix) / 10 + j
            testSet = []
            trainSet = []
            for k in range(len(matrix)):
                if k >= int(j) and k <= int(si):
                    testSet.append(matrix[k])
                else:
                    trainSet.append(matrix[k])
            classifier = None
            if alg == 'NaiveBayes':
                classifier = NaiveBayesClassifier.train(trainset)
            elif alg == 'Maxent':
                classifier = MaxentClassifier.train(trainset)
            elif alg == 'DecisionTree':
                classifier = DecisionTreeClassifier.train(trainset)
            else:
                sys.stderr.write("ERROR: Algorithm '%s' not supported. Dying now.\n")
                sys.exit(1)
            accuracy = classify.accuracy(classifier, testSet)
            scores.append(accuracy)
            j = si

        avg = float(sum(scores) / x)
        sys.stderr.write("INFO: Average score over traditional %i-fold validation: %f.\n" % (x, avg))

    def getTopNFeatures(self, n):

        self.classifier.show_most_informative_features(n)

    def explain(self):

        self.classifier.explain()
        
    def classifyText(self, text, classifier):

        global memoryMap
        
        cd = []
        if classifier == None and self.classifier == None:
            sys.stderr.write("ERROR: No classifier available. Please train one first. Dying now.\n")
            sys.exit(1)
        elif classifier == None: # if the current one has been trained already in a previous step, or unpickled
            classifier = self.classifier

        if memoryMap[text]:
            return memoryMap[text]
        #else...
        
        sentences = sent_tokenize(re.sub(r'[()]+', '', text)) # nltk.tree has a problem with parenthesis. look into this, maybe there is a way to escape them when parsing/traversing a tree.
        tokenizedSents = [sent.split() for sent in sentences]
        try:
            forest = self.lexParser.parse_sents(tokenizedSents)
            for trees in forest:
                for tree in trees:
                    featureVectors = self.getVectorsForTree(tree)
                    for fv in featureVectors:
                        isConnective = classifier.classify(self.matrixRow2fDict(fv))
                        if isConnective:# and fv[0] in self.dictionary: # something like this can be included if precision is too low
                            cd.append((fv[0], True))
                        else:
                            cd.append((fv[0], False))
                            
                        
        except ValueError:
            sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")
        memoryMap[text] = cd
            
        return cd

    def pickleClassifier(self, pickleFileName):

        pf = codecs.open(pickleFileName, 'wb')
        pickle.dump(self.classifier, pf, 1)
        pf.close()
        sys.stderr.write("INFO: Successfully stored trained classifier in %s.\n" % pickleFileName)

    def unpickleClassifier(self, pickleFileName):

        pf = codecs.open(pickleFileName, 'rb')
        self.classifier = pickle.load(pf)
        pf.close()
        sys.stderr.write("INFO: Successfully loaded trained classifier from %s.\n" % pickleFileName)

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

            
def customEvaluation(flist, alg, lang, inputFormat, preFilledMatrixBool):

    numIterations = 10
    accuracyScores = []
    precisionScores = []
    recallScores = []
    fScores = []
    trainingClassifier = ConnectiveClassifier(alg, lang)
    testClassifier = ConnectiveClassifier(alg, lang)
    p = int(len(flist) / 10)
    pl = [int(x) for x in range(0, len(flist), p)]
    pl.append(int(len(flist))) # think all this casting to int is a bit redundant, but too lazy to debug
    for i in range(numIterations):
        total = 0
        correct = 0
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        sys.stderr.write("INFO: Starting iteration %i of %i for algorithm '%s'.\n" % (i+1, numIterations, alg))
        # trying with every portion of the data set as test set once (for 10-fold cv), as deviations with random splits was huge (ranging from f of .77 to f of .96)
        testData = flist[pl[i]:pl[i+1]]
        trainingData = [f for f in flist if not f in testData]
        #random.shuffle(flist)
        #p = int(len(flist) / 10)
        #trainingData = flist[p:]
        #testData = flist[:p]

        if inputFormat.lower() == 'pcc':
            trainingClassifier.buildFeatureMatrixFromPCC(trainingData)
        elif inputFormat.lower() == 'conll':
            trainingClassifier.buildFeatureMatrixFromCONLL(trainingData)
        elif preFilledMatrixBool:
            matrix = trainingClassifier.readMatrix('conll2016SharedTaskDataMatrix.csv') # TODO: make argument!
            trainingData = matrix[p:]
            testData = matrix[:p]
            if alg == 'NaiveBayes':
                trainingClassifier = NaiveBayesClassifier.train(trainingData)
            elif alg == 'DecisionTree':
                trainingClassifier = DecisionTreeClassifier.train(trainingData)
            elif alg == 'Maxent':
                trainingClassifier = MaxentClassifier.train(trainingData)
            else:
                sys.stderr.write("ERROR: Algorithm '%s' not supported. Please pick one from 'NaiveBayes', 'DecisionTree' or 'Maxent'.\n")
                
        else:
            sys.stderr.write("ERROR: inputFormat '%s' not supported.\n" % inputFormat)
            sys.exit(1)
                             
        for q, f in enumerate(testData):
            if verbose:
                sys.stderr.write("INFO: Classifying file .../%s (%s/%s).\n" % (f.split('/')[-1], str(q+1), str(len(testData))))
            if inputFormat.lower() == 'pcc':
                pccTokens = testClassifier.PCCParser.parseConnectorFile(f)
                testClassifier.getFeaturesForPCCTokens(pccTokens)
            elif inputFormat.lower() == 'conll':
                conllTokens = testClassifier.CONLLParser.parsePDTBFile(f)
                testClassifier.getFeaturesForCONLLTokens(conllTokens)
            else:
                sys.stderr.write("ERROR: inputFormat '%s' not supported.\n" % inputFormat)
                sys.exit(1)
                
            # now the info is in testClassifier.matrix, next feed it the sentence and check
            td = defaultdict(str)
            flatString = ''
            for ii in testClassifier.matrix:
                td[ii] = testClassifier.matrix[ii][-1]
                flatString += ' ' + testClassifier.matrix[ii][0]
            flatString = flatString.strip()
            cd = trainingClassifier.classifyText(flatString, None)
            for l, tupl in enumerate(cd):
                w = tupl[0]
                classifiedClass = tupl[1]
                realClass = td[l]
                total += 1
                # redundancy below for readability...
                if realClass == True:
                    if classifiedClass == True:
                        correct += 1
                        tp += 1
                    elif classifiedClass == False:
                        fn += 1
                elif realClass == False:
                    if classifiedClass == False:
                        correct += 1
                        tn += 1
                    elif classifiedClass == True:
                        fp += 1
                        
        accuracy = correct / float(total)
        precision = 0
        recall = 0
        f1 = 0
        if not tp + fp == 0 and not tp + fn == 0: # division by zero probably only ever happens on small test set, but in any case...
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
        accuracyScores.append(accuracy)
        precisionScores.append(precision)
        recallScores.append(recall)
        fScores.append(f1)
        if verbose:
            print("INFO: accuracy for run %s: %s." % (str(i+1), str(accuracy)))
            print("INFO: precision for run %s: %s." % (str(i+1), str(precision)))
            print("INFO: recall for run %s: %s." % (str(i+1), str(recall)))
            print("INFO: f1 for run %s: %s." % (str(i+1), str(f1)))

            
    avgAccuracy = sum(accuracyScores) / float(numIterations)
    avgPrecision = sum(precisionScores) / float(numIterations)
    avgRecall = sum(recallScores) / float(numIterations)
    avgF = sum(fScores) / float(numIterations)
    print("INFO: Average accuracy over %i runs for '%s': %f." % (numIterations, alg, avgAccuracy))
    print("INFO: Average precision over %i runs for '%s': %f." % (numIterations, alg, avgPrecision))
    print("INFO: Average recall over %i runs for '%s': %f." % (numIterations, alg, avgRecall))
    print("INFO: Average f1 over %i runs for '%s': %f." % (numIterations, alg, avgF))

    if verbose:
        print("Obtaining most informative features:")
        trainingClassifier.getTopNFeatures(500)



def specialOneOffDebuggingMethod1():
        
    alg = 'Maxent'
    tcc = ConnectiveClassifier(alg, 'en')
    #tcc.buildFeatureMatrixFromCONLL(getInputfiles('/Users/peter/Desktop/osloCoop/conll16/en.train/conll_format/'))
    #tcc.writeMatrix('conllConnectiveMatrix.csv')
    #tcc.pickleClassifier('conllMaxentClassifier.pickle')
    numIterations = 1
    i = 0
    tcc.unpickleClassifier('conllMaxentClassifier.pickle')
    #tcc.unpickleClassifier('conllBayesClassifier.pickle')
    testData = getInputfiles('/Users/peter/Desktop/osloCoop/conll16/en.test/conll_format/')
    total = 0
    correct = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    accuracyScores = []
    precisionScores = []
    recallScores = []
    fScores = []
    for q, f in enumerate(testData):
        
        if verbose:
            sys.stderr.write("INFO: Classifying file .../%s (%s/%s).\n" % (f.split('/')[-1], str(q+1), str(len(testData))))
        conllTokens = tcc.CONLLParser.parsePDTBFile(f)
        tcc.getFeaturesForCONLLTokens(conllTokens)
        td = defaultdict(str)
        flatString = ''
        for ii in tcc.matrix:
            td[ii] = tcc.matrix[ii][-1]
            flatString += ' ' + tcc.matrix[ii][0]
        flatString = flatString.strip()
        cd = tcc.classifyText(flatString, None)
        for l, tupl in enumerate(cd):
            w = tupl[0]
            classifiedClass = tupl[1]
            realClass = td[l]
            total += 1
            # redundancy below for readability...
            if realClass == True:
                if classifiedClass == True:
                    correct += 1
                    tp += 1
                elif classifiedClass == False:
                    fn += 1
                    print('FALSE NEGATIVE: Did not recognise "%s" in context: "%s"\n' % (w, (cd[max(0, l-8):min(l+8, len(cd)-1)])))
            elif realClass == False:
                if classifiedClass == False:
                    correct += 1
                    tn += 1
                elif classifiedClass == True:
                    fp += 1
                    print('FALSE POSITIVE: Falsely recognised "%s" in context: "%s"\n' % (w, (cd[max(0, l-8):min(l+8, len(cd)-1)])))
                    
    accuracy = correct / float(total)
    precision = 0
    recall = 0
    f1 = 0
    if not tp + fp == 0 and not tp + fn == 0: # division by zero probably only ever happens on small test set, but in any case...
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
    accuracyScores.append(accuracy)
    precisionScores.append(precision)
    recallScores.append(recall)
    fScores.append(f1)
    
    print("INFO: accuracy for run %s: %s." % (str(i+1), str(accuracy)))
    print("INFO: precision for run %s: %s." % (str(i+1), str(precision)))
    print("INFO: recall for run %s: %s." % (str(i+1), str(recall)))
    print("INFO: f1 for run %s: %s." % (str(i+1), str(f1)))

    

def specialOneOffDebuggingMethod2():

    # the following is just a copy from: https://stackoverflow.com/questions/30991592/support-vector-machine-in-python-using-libsvm-example-of-features
    import pandas as pd
    import numpy as np
    from sklearn import svm

    matrixLines = codecs.open('pccConnectiveIntMatrix.csv', 'r').readlines()
    header = matrixLines[0]
    matrix = matrixLines[1:]

    accuracyScores = []
    precisionScores = []
    recallScores = []
    fScores = []
    
    
    numIterations = 10
    p = int(len(matrix) / numIterations)
    pl = [int(x) for x in range(0, len(matrix), p)]
    pl.append(int(len(matrix))) # think all this casting to int is a bit redundant, but too lazy to debug
    for i in range(numIterations):
        total = 0
        correct = 0
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        sys.stderr.write("INFO: Starting iteration %i of %i.\n" % (i+1, numIterations))
        # this is stupid (doing this with temp files instead of slicing the numpy array...), but one-off code anyway...
        testMatrix = matrix[pl[i]:pl[i+1]]
        trainMatrix = matrix[0:pl[i]] + matrix[pl[i+1]:]
        tempTest = codecs.open('tempTest.csv', 'w')
        tempTest.write(header)
        for line in testMatrix:
            tempTest.write(line)
        tempTest.close()
        tempTrain = codecs.open('tempTrain.csv', 'w')
        tempTrain.write(header)
        for line in trainMatrix:
            tempTrain.write(line)
        tempTrain.close()
        sys.stderr.write("INFO: Reading train matrix.\n")
        train_dataframe = pd.read_csv('tempTrain.csv')
        sys.stderr.write("INFO: Done.\n")
        sys.stderr.write("INFO: Preparing data.\n")
        train_labels = train_dataframe.class_label
        labels = list(set(train_labels))
        train_labels = np.array([labels.index(x) for x in train_labels])
        train_features = train_dataframe.iloc[:,1:]
        train_features = np.array(train_features)
        
    
        sys.stderr.write("INFO: Done.\n")
        classifier = svm.SVC()
        sys.stderr.write("INFO: Training classifier.\n")
        classifier.fit(train_features, train_labels)
        sys.stderr.write("INFO: Done.\n")

        sys.stderr.write("INFO: Reading test matrix.\n")
        test_dataframe = pd.read_csv('tempTest.csv')
        sys.stderr.write("INFO: Done.\n")
        sys.stderr.write("INFO: Preparing data.\n")
        test_labels = test_dataframe.class_label
        labels = list(set(test_labels))
        test_labels = np.array([labels.index(x) for x in test_labels])
        test_features = test_dataframe.iloc[:,1:]
        test_features = np.array(test_features)

        sys.stderr.write("INFO: Done.\n")
        sys.stderr.write("INFO: Proceeding with classification.\n")
        results = classifier.predict(test_features)
        sys.stderr.write("INFO: Done.\n")


        sys.stderr.write("INFO: Starting confusion matrix population.\n")
        for i, r in enumerate(results):
            total += 1
            classifiedClass = r
            realClass = test_labels[i]
            # redundancy below for readability...
            if realClass == True:
                if classifiedClass == True:
                    correct += 1
                    tp += 1
                elif classifiedClass == False:
                    fn += 1
                
            elif realClass == False:
                if classifiedClass == False:
                    correct += 1
                    tn += 1
                elif classifiedClass == True:
                    fp += 1
    

        accuracy = correct / float(total)
        precision = 0
        recall = 0
        f1 = 0
        if not tp + fp == 0 and not tp + fn == 0:
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
        accuracyScores.append(accuracy)
        precisionScores.append(precision)
        recallScores.append(recall)
        fScores.append(f1)
        verbose = True
        if verbose:
            print("INFO: accuracy for run %s: %s." % (str(i+1), str(accuracy)))
            print("INFO: precision for run %s: %s." % (str(i+1), str(precision)))
            print("INFO: recall for run %s: %s." % (str(i+1), str(recall)))
            print("INFO: f1 for run %s: %s." % (str(i+1), str(f1)))



        
    avgAccuracy = sum(accuracyScores) / float(numIterations)
    avgPrecision = sum(precisionScores) / float(numIterations)
    avgRecall = sum(recallScores) / float(numIterations)
    avgF = sum(fScores) / float(numIterations)
    print("INFO: Average accuracy over %i runs: %f." % (numIterations, avgAccuracy))
    print("INFO: Average precision over %i runs: %f." % (numIterations, avgPrecision))
    print("INFO: Average recall over %i runs: %f." % (numIterations, avgRecall))
    print("INFO: Average f1 over %i runs: %f." % (numIterations, avgF))
    

    """
    sys.stderr.write("INFO: Reading train matrix.\n")
    train_dataframe = pd.read_csv('conllIntTrainMatrix.csv')
    sys.stderr.write("INFO: Done.\n")
    sys.stderr.write("INFO: Preparing data.\n")
    train_labels = train_dataframe.class_label
    labels = list(set(train_labels))
    train_labels = np.array([labels.index(x) for x in train_labels])
    train_features = train_dataframe.iloc[:,1:]
    train_features = np.array(train_features)
    
    
    sys.stderr.write("INFO: Done.\n")
    classifier = svm.SVC()
    sys.stderr.write("INFO: Training classifier.\n")
    classifier.fit(train_features, train_labels)
    sys.stderr.write("INFO: Done.\n")

    sys.stderr.write("INFO: Reading test matrix.\n")
    test_dataframe = pd.read_csv('conllIntTestMatrix.csv')
    sys.stderr.write("INFO: Done.\n")
    sys.stderr.write("INFO: Preparing data.\n")
    test_labels = test_dataframe.class_label
    labels = list(set(test_labels))
    test_labels = np.array([labels.index(x) for x in test_labels])
    test_features = test_dataframe.iloc[:,1:]
    test_features = np.array(test_features)

    sys.stderr.write("INFO: Done.\n")
    sys.stderr.write("INFO: Proceeding with classification.\n")
    results = classifier.predict(test_features)
    sys.stderr.write("INFO: Done.\n")

    tp = 0
    fn = 0
    correct = 0
    tn = 0
    fp = 0
    for i, r in enumerate(results):
        classifiedClass = r
        realClass = test_labels[i]
        # redundancy below for readability...
        if realClass == True:
            if classifiedClass == True:
                correct += 1
                tp += 1
            elif classifiedClass == False:
                fn += 1
                
        elif realClass == False:
            if classifiedClass == False:
                correct += 1
                tn += 1
            elif classifiedClass == True:
                fp += 1
                
    accuracy = correct / float(len(test_labels))
    precision = 0
    recall = 0
    f1 = 0
    if not tp + fp == 0 and not tp + fn == 0:
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
    
    print("INFO: accuracy: %s." % (str(accuracy)))
    print("INFO: precision: %s." % (str(precision)))
    print("INFO: recall: %s." % (str(recall)))
    print("INFO: f1: %s." % (str(f1)))

    """
    """
    Results of one round (first 30k training, last 3.something k test for PCC:
    INFO: accuracy: 0.9645232815964523.
    INFO: precision: 1.0.
    INFO: recall: 0.1111111111111111.
    INFO: f1: 0.19999999999999998.
    TODO: proper k-fold cv...

    First results of one round for CONLL (first 839544 lines of conllConnectiveMatrix.csv as train, last 93282 lines as test (90-10 split)):
    
    
    """

        
        
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectivesFolder', dest='connectivesFolder', help='Specify PCC connectives folder to construct a feature matrix for training a classifier.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Include to get some debug info.')
    parser.add_option('-l', '--language', dest='language', help='Specify language. Currently supported languages: "de, en".')
    parser.add_option('-f', '--format', dest='inputFormat', help='Specify input format. Currently supported formats: "pcc, conll".')
    
    
    options, args = parser.parse_args()

    if not options.connectivesFolder or not options.language or not options.inputFormat:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if options.verbose:
        verbose = True
        
    #customEvaluation(getInputfiles(options.connectivesFolder), 'NaiveBayes')
    #customEvaluation(getInputfiles(options.connectivesFolder), 'DecisionTree')
    #customEvaluation(getInputfiles(options.connectivesFolder), 'Maxent', options.language, options.inputFormat, False)
    
    alg = 'Maxent'
    cc = ConnectiveClassifier(alg, options.language)
    if options.inputFormat.lower() == 'pcc':
        cc.buildFeatureMatrixFromPCC(getInputfiles(options.connectivesFolder))
        #cc.pickleClassifier(alg + 'Classifier.pickle')
    elif options.inputFormat.lower() == 'conll':
        cc.buildFeatureMatrixFromCONLL(getInputfiles(options.connectivesFolder))
    
    #cc.writeMatrix('pccConnectiveMatrix.csv')
    #cc.pickleClassifier('pccMaxentClassifier.pickle')
    #cc.randomCrossValidate('tempout.csv')
    #cc.traditionalCrossValidate('tempout.csv')

    #specialOneOffDebuggingMethod1()
    specialOneOffDebuggingMethod2()

    
    #cc.unpickleClassifier('conllMaxentClassifier.pickle')
    

    """
    cc.classifyText("Gejagt Wer es angesichts der Festlichkeiten zum 40. Stadtjubiläum vergessen haben sollte :", None)
    cc.classifyText("Gejagt Wer es angesichts der Festlichkeiten zum 40. Stadtjubiläum vergessen haben sollte :", None)
b    """
    
    """
    if options.verbose:
        for item in sanityList:
            sys.stderr.write("VERBOSE: Added \t'%s'\t as connective.\n" % item)
    """

    """
    TODO list:
    - evaluate on conll 2016 shared task data: revisit the f-score of around .86 I got there, as this is way too low.
    - 

    """
