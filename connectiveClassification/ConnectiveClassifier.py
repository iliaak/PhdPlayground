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
import pickle

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
        scriptLocation = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        Config.read(os.path.join(scriptLocation, 'settings.conf'));
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

    def matrixRow2fDict(self, row):
        
        fDict = {}
        for i, feature in enumerate(row[:len(row)-1]):
            fDict[self.index2FeatureName[i+1]] = feature
        return fDict

    def buildFeatureMatrixFromPCC(self, connectiveFiles):

        for i, cxml in enumerate(connectiveFiles):
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

        self.classifier = NaiveBayesClassifier.train(matrix)
       
        
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


    def classifyText(self, text, classifier):

        if classifier == None and self.classifier == None:
            sys.stderr.write("ERROR: No classifier available. Please train one first. Dying now.\n")
            sys.exit(1)
        elif classifier == None: # if the current one has been trained already in a previous step, or unpickled
            classifier = self.classifier

        
        sentences = sent_tokenize(re.sub(r'[()]+', '', text)) # nltk.tree has a problem with parenthesis. look into this, maybe there is a way to escape them when parsing/traversing a tree.
        tokenizedSents = [sent.split() for sent in sentences]
        try:
            forest = self.lexParser.parse_sents(tokenizedSents)
            for trees in forest:
                for tree in trees:
                    featureVectors = self.getVectorsForTree(tree)
                    for fv in featureVectors:
                        isConnective = classifier.classify(self.matrixRow2fDict(fv))
                        if isConnective:
                            print(fv[0] + '\t' + 'connective')
                        else:
                            print(fv[0])

                        
        except ValueError:
            sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")



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




if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectivesFolder', dest='connectivesFolder', help='Specify PCC connectives folder to construct a feature matrix for training a classifier.')
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Include to get some debug info.')
    
    
    options, args = parser.parse_args()

    if not options.connectivesFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)


    
    cc = ConnectiveClassifier()
    #cc.buildFeatureMatrixFromPCC(getInputfiles(options.connectivesFolder))
    #cc.pickleClassifier('naiveBayesClassifier.pickle')
    
    #cc.writeMatrix('tempout.csv')
    #cc.randomCrossValidate('tempout.csv')
    #cc.traditionalCrossValidate('tempout.csv')

    cc.unpickleClassifier('naiveBayesClassifier.pickle')


    cc.classifyText("Gejagt Wer es angesichts der Festlichkeiten zum 40. Stadtjubiläum vergessen haben sollte :", None)
    print('\n')
    cc.classifyText("In Falkensee ist derzeit vor allem eines - Bürgermeister-Wahlkampf .", None)
    print('\n')
    cc.classifyText("Da kündigt Jürgen Bigalke ( SPD ) also an , die mit Finanzdezernentin Ruth Schulz besetzte Stelle der 1. Beigeordneten ( stellvertretende Bürgermeisterin ) nicht öffentlich ausschreiben und dies noch vor dem 11. November bestätigen lassen zu wollen .", None)
    print('\n')
    cc.classifyText("Acht weitere Jahre könnte die Sozialdemokratin dann amtieren - selbst dann wenn der Rathauschef abgewählt werden und damit die Ein-Stimmen-Mehrheit der SPD/FDP-Zählgemeinschaft in der Stadtverordenten-Versammlung zusammenbrechen sollte .", None)
    print('\n')
    cc.classifyText("Das ist der Knackpunkt :", None)
    print('\n')
    cc.classifyText("Grüne und Christdemokraten wären schlechte Herausforderer , würden sie dieses Ansinnen nicht als \" schlechten Stil \" und als \" Erbhof-Sicherung \" zerreißen , auch wenn die Sache rechtlich in Ordnung ginge .", None)
    print('\n')
    cc.classifyText("So bleiben Fragen :", None)
    print('\n')
    cc.classifyText("Sind Bigalke und die Seinen wirklich so naiv und denken , diese wichtige Personalie geräuschlos durchzukriegen ?", None)
    print('\n')
    cc.classifyText("Oder handeln sie kaltschnäuzig nach dem Motto : Augen zu und durch ?", None)
    print('\n')
    cc.classifyText("Wie auch immer .", None)
    print('\n')
    cc.classifyText("Der Bürgermeister hat aus einer Mücke einen Elefanten gemacht - und ihn in den Porzellanladen gejagt .", None)
    print('\n')
    
    
    """
    if options.verbose:
        for item in sanityList:
            sys.stderr.write("VERBOSE: Added \t'%s'\t as connective.\n" % item)
    """

    """
    TODO list:
    - write method to get f-score (in addition to accuracy)
    - evaluate on conll 2016 shared task data (which is probably english; sufficient to only change the lexparser pointer in settings.conf for this?)
    - 

    """
