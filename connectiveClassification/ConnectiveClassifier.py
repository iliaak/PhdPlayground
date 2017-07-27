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
from nltk import NaiveBayesClassifier
from nltk import classify
#import PCCParser
import csv
import importlib.util
import configparser


"""
TODO: <description here>
"""

class ConnectiveClassifier():

    def __init__(self):

        self.matrix = defaultdict(list)
        self.maxId = 0
        self.sanityList = []
        self.classifier = None

        # read settings from config file
        Config = configparser.ConfigParser()
        Config.read("settings.conf")
        os.environ['JAVAHOME'] = Config.get('JAVA', 'JAVAHOME')
        os.environ['STANFORD_PARSER'] = Config.get('CORENLP', 'STANFORD_PARSER')
        os.environ['STANFORD_MODELS'] = Config.get('CORENLP', 'STANFORD_MODELS')
        lexParserPath = Config.get('CORENLP', 'LEXPARSER')
        self.lexParser = stanford.StanfordParser(model_path= lexParserPath)
        pccPath = Config.get('MISC', 'PCCPARSER')
        pccLoc = importlib.util.spec_from_file_location("PCCParser", pccPath)
        self.PCCParser = importlib.util.module_from_spec(pccLoc)
        pccLoc.loader.exec_module(self.PCCParser)
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

    

    def buildFeatureMatrixFromPCC(self, connectiveFiles):

        for i, cxml in enumerate(connectiveFiles):
            sys.stderr.write("INFO: Parsing file .../%s (%s/%s).\n" % (cxml.split('/')[-1], str(i+1), str(len(connectiveFiles))))
            pccTokens = self.PCCParser.parseConnectorFile(cxml)
            self.getFeaturesForPCCTokens(pccTokens)
        
    def getFeaturesForPCCTokens(self, pccTokens):        

        # NOTE: The bookkeeping here is a bit complicated, due to the conano inline xml format. Need to keep track of token id's to know if they are connective or not, and link this to the current node in the tree
        localId2class = defaultdict(bool)
        # ugly fix: filtering for parenthesis, as these are skipped by nltk.Tree traversal (probably because this symbol is used in internal representation)
        pccTokens = self.filterTokens(pccTokens)
        for i, t in enumerate(pccTokens):
            if t.isConnective:
                localId2class[i] = True
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
        
    def trainClassifierFromCSV(self, csvFile):

        matrix = self.parseFeatureMatrixFromCSV(csvFile)
        alg = 'NaiveBayes' # TODO: may want to do other algorithms; plugin here
        if alg == 'NaiveBayes':
            self.classifier = NaiveBayesClassifier.train(matrix)
        

    def randomCrossValidate(self, csvFile):

        import random
        matrix = self.parseFeatureMatrixFromCSV(csvFile)
        x = 10
        scores = []    
        for i in range(x):
            random.shuffle(matrix)
            si = len(matrix) / 10
            testSet = matrix[:int(si)]
            trainSet = matrix[int(si):]
            classifier = NaiveBayesClassifier.train(trainSet)
            accuracy = classify.accuracy(classifier, testSet)
            scores.append(accuracy)
            
        avg = float(sum(scores) / x)
        sys.stderr.write("INFO: Average score over random %i-fold validation: %f.\n" % (x, avg))



    def traditionalCrossValidate(self, csvFile):

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
            classifier = NaiveBayesClassifier.train(trainSet)
            accuracy = classify.accuracy(classifier, testSet)
            scores.append(accuracy)
            j = si

        avg = float(sum(scores) / x)
        sys.stderr.write("INFO: Average score over traditional %i-fold validation: %f.\n" % (x, avg))


        


def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist




if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectivesFolder', dest='connectivesFolder', help='Specify PCC connectives folder to construct a feature matrix for training a classifier.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Include to get some debug info.')
    
    
    options, args = parser.parse_args()

    if not options.connectivesFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # MAJOR TODO: try this with the 2016 conll shared task code (on English) to see how this code compares to the scores on the shared task. For that, parse also other formats (write buildFeatureMatrixFromConll or something)

    
    cc = ConnectiveClassifier()
    #cc.buildFeatureMatrixFromPCC(getInputfiles(options.connectivesFolder))
    #cc.writeMatrix('tempout.csv')
    cc.randomCrossValidate('tempout.csv')
    cc.traditionalCrossValidate('tempout.csv')
    
    
    """
    if options.verbose:
        for item in sanityList:
            sys.stderr.write("VERBOSE: Added \t'%s'\t as connective.\n" % item)
    """