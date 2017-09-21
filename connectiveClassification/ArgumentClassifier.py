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
TODO: <description here>
"""
verbose = False


class ArgumentClassifier():

    def __init__(self, alg, lang):

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


    def classifyText(self, text):# no classifier yet

        # to test and build procedure, using a dumb trigger happy, always fire classifier for now
        ad = []
        for word in text.split():
            ad.append((word, True))

        #TODO: have the procedure working. Now do connective classification. Whenever we have identified a connective, get the clause it is in (where the trick is to find the right granularity) and tag that whole clause as intArg.
        # think about this for a bit. Nice to have a setup where I can do everything in isolation/modular, but either I get a connectiveClassifier in here, re-use part of the code to do connective classification and argument classification in one. Alternative is to just pass the text to the connective classifier, get it back, stop at every connective, get its phrase. But his means I have to do the parsing, the most expensive part in the process, twice...
            
        return ad

    def evaluate(self, connectorfiles, alg, lang):

        # works for now, because I'm splitting on whitespace. Has to be more sophisticated when I'm doing proper parsing and everything (due to parenthesis problem for nltk)

        # TODO: think having the file loop inside here is a bit stupid, maybe take it out
        
        accuracyScores = []
        precisionScores = []
        recallScores = []
        fScores = []

        for f in connectorfiles:
            discoursetokenlist = self.PCCParser.parseConnectorFile(f)
            text = ' '.join([dt.token for dt in discoursetokenlist])
            ad = self.classifyText(text)
        
            correct = 0
            total = len(ad)
            tp = 0
            tn = 0
            fn = 0
            fp = 0
        
            for i, tupl in enumerate(ad):
                realClass = False
                if discoursetokenlist[i].segmentType == 'unit':
                    if discoursetokenlist[i].intOrExt == 'int':
                        realClass = True
                classClass = tupl[1]
                #sanity check:
                if discoursetokenlist[i].token == tupl[0]:
                    if realClass == True:
                        if classClass == True:
                            correct += 1
                            tp += 1
                        elif classClass == False:
                            fn += 1
                    elif realClass == False:
                        if classClass == False:
                            correct += 1
                            tn += 1
                        elif classClass == True:
                            fp += 1
                else:
                    print("We have a serious problem.")
                    sys.exit(1)
            accuracy = correct / float(total)
            precision = 0
            recall = 0
            f1 = 0
            print("DEBUG TOTAL:", total)
            print("DEBUG CORRECT:", correct)
            print("DEBUG TP:", tp)
            print("DEBUG fP:", fp)
            print("DEBUG Tn:", tn)
            print("DEBUG fn:", fn)
            if not tp + fp == 0 and not tp + fn == 0: # division by zero probably only ever happens on small test set, but in any case...
                precision = tp / float(tp + fp)
                recall = tp / float(tp + fn)
                f1 = 2 * ((precision * recall) / (precision + recall))
            accuracyScores.append(accuracy)
            precisionScores.append(precision)
            recallScores.append(recall)
            fScores.append(f1)
        
            
        avgAccuracy = sum(accuracyScores) / float(len(connectorfiles))
        avgPrecision = sum(precisionScores) / float(len(connectorfiles))
        avgRecall = sum(recallScores) / float(len(connectorfiles))
        avgF = sum(fScores) / float(len(connectorfiles))
        alg = 'dummy'
        print("INFO: Average accuracy for '%s': %f." % (alg, avgAccuracy))
        print("INFO: Average precision for '%s': %f." % (alg, avgPrecision))
        print("INFO: Average recall for '%s': %f." % (alg, avgRecall))
        print("INFO: Average f1 for '%s': %f." % (alg, avgF))

            
        
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
    parser.add_option('-l', '--language', dest='language', help='Specify language. Currently supported languages: "de, en".')
    parser.add_option('-f', '--format', dest='inputFormat', help='Specify input format. Currently supported formats: "pcc, conll".')
    
    
    options, args = parser.parse_args()

    if not options.connectivesFolder or not options.language or not options.inputFormat:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if options.verbose:
        verbose = True

    
    alg = 'dummy'
    AC = ArgumentClassifier(alg, options.language)
    connectorfiles = getInputfiles(options.connectivesFolder)
    AC.evaluate(connectorfiles, alg, options.language)
        
