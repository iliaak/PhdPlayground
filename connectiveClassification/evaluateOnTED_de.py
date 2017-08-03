#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import ConnectiveClassifier
import csv

"""
Probably a very temporary file, used to evaluate on the 7-document corput I got from Yulia (TED DE, annotated for connectives)
"""

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


def getRealClasses(fDict):

    classDict = defaultdict(list)
    for f in fDict:
        tokenTuples = []
        fullPath = fDict[f]["raw"]
        content = codecs.open(fullPath, 'r').readlines()
        annotations = fDict[f]["ann"]
        explicitConnectiveIndices = set()
        with codecs.open(annotations, 'r') as csvfile:
            annreader = csv.reader(csvfile, delimiter='|')
            for row in annreader:
                if row[0] == 'Explicit':
                    if re.search(';', row[1]): # connective is discontinuous
                        indexPairs = row[1].split(';')
                        for pair in indexPairs:
                            p = pair.split('..')
                            t = (int(p[0]), int(p[1]))
                            explicitConnectiveIndices.add(t)
                    else:
                        p = row[1].split('..')
                        t = (int(p[0]), int(p[1]))
                        explicitConnectiveIndices.add(t)
        i = 0
        for line in content:
            for token in line.split(): # assumption is that all is tokenised
                if (i, i+len(token)) in explicitConnectiveIndices:
                    t = (token, True)
                    tokenTuples.append(t)
                else:
                    t = (token, False)
                    tokenTuples.append(t)
                i += len(token) + 1 # +1 due to whitespace
            #i += 1
        classDict[f] = tokenTuples

    return classDict



def getFileDict(annFolder, rawFolder):

    annFiles = getInputfiles(annFolder)
    rawFiles = getInputfiles(rawFolder)

    if not len(annFiles) == len(rawFiles):
        sys.stderr.write("ERROR: Not same number of files for annotations and raw text. Dying now.\n")

    fDict = defaultdict(lambda : defaultdict(str))
    for f in annFiles:
        fDict[f.split('/')[-1]]["ann"] = f
    for f in rawFiles:
        fDict[f.split('/')[-1]]["raw"] = f

    return fDict
        
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-r', '--rawFolder', dest='rawFolder', help='specify input folder')
    parser.add_option('-a', '--annotationsFolder', dest='annotationsFolder', help='specify input folder')
    

    options, args = parser.parse_args()

    if not options.rawFolder or not options.annotationsFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    
    fDict = getFileDict(options.annotationsFolder, options.rawFolder)
    classDict = getRealClasses(fDict)
    
    alg = 'Maxent'
    cc = ConnectiveClassifier.ConnectiveClassifier(alg)
    cc.unpickleClassifier('MaxentClassifier.pickle')
    total = 0
    correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for f in classDict:
        testClassifier = ConnectiveClassifier.ConnectiveClassifier(alg) # this is only for building the feature matrix
        testClassifier.getFeaturesForTokenList(classDict[f])
        td = defaultdict(str)
        flatString = ''
        for ii in testClassifier.matrix:
            td[ii] = testClassifier.matrix[ii][-1]
            flatString += ' ' + testClassifier.matrix[ii][0]
        flatString = flatString.strip()
        cd = cc.classifyText(flatString, None)
        for l, tupl in enumerate(cd):
            w = tupl[0]
            classifiedClass = tupl[1]
            realClass = td[l]
            total += 1
            print("DEBUGGING Word, realClass, classClass:", w, realClass, classifiedClass)
            # redundancy below for readability...
            #print("realClass:", realClass)
            #print("classClass:", classifiedClass)
            if realClass == True:
                if classifiedClass == True:
                    correct += 1
                    tp += 1
                elif classifiedClass == False:
                    fn += 1
                    print("FALSE NEGATIVE!!!")
            elif realClass == False:
                if classifiedClass == False:
                    correct += 1
                    tn += 1
                elif classifiedClass == True:
                    fp += 1
                    print("FALSE POSITIVE!!!")
            

                    
    accuracy = correct / float(total)
    precision = 0
    recall = 0
    f1 = 0
    if not tp + fp == 0 and not tp + fn == 0: # division by zero probably only ever happens on small test set, but in any case...
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
