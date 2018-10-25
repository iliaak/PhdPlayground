#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import csv

"""
CONLL format Parser for connective classification (current code based on PDTB in CONLL format)
"""

class CONLLToken:
    def __init__(self, token):
        self.token = token

    # TODO: read up on proper way of doing this, think I'm mixing java and python too much
    def setGlobalTokenId(self, val):
        self.globalTokenId = val
    def setSentenceId(self, val):
        self.sentenceId = val
    def setSTokenId(self, val):
        self.sentenceTokenId = val
    def setCONLLPOSTag(self, val):
        self.conllPosTag = val
    def setConnectiveBoolean(self, val):
        self.isConnective = val
    def setDiscourseFunction(self, val):
        self.discourseFunction = val
    def setSecondaryAnnotation(self, val): # read docs: no idea what this field is doing
        self.secondaryAnnotation = val
    def setTerniaryAnnotation(self, val): # read docs: same here
        self.terniaryAnnotation = val
    def setFileId(self, val):
        self.fileId = val
        
        
def parsePDTBFile(conllFile):

    conllTokens = []
    with codecs.open(conllFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            # don't really understand why there are so many empty columns, and why the thing I'm looking for seems to be in another one every time...
            if len(row) > 5:
                globalId = row[0]
                sentenceId = row[1]
                sTokenId = row[2]
                token = row[3]
                pos = row[4]
                connectiveBoolean = False
                if re.search("conn", ' '.join(row[5:])):
                    connectiveBoolean = True # getting rid of the sense for now (only doing binary classifcation. Include sense at some point!)
                cToken = createConllToken(globalId, sentenceId, sTokenId, token, pos, connectiveBoolean, os.path.splitext(os.path.basename(conllFile))[0])
                conllTokens.append(cToken)
    return conllTokens

  
def createConllToken(globalId, sentenceId, sTokenId, token, pos, connectiveBoolean, fileId):

    ct = CONLLToken(token)
    ct.setGlobalTokenId(int(globalId))
    ct.setSentenceId(int(sentenceId))
    ct.setSTokenId(int(sTokenId))
    ct.setCONLLPOSTag(pos)
    ct.setConnectiveBoolean(connectiveBoolean)
    ct.setFileId(fileId)
    
    return ct



def printStats(conllTokens):

    cc = 0
    wc = 0
    for ct in conllTokens:
        wc += 1
        if ct.isConnective:
            cc += 1
            
    print('INFO: %i words.' % wc)
    print('INFO: %i connectors.' % cc)
    

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


            
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--conllFolder', dest='conllFolder', help='specify CONLL folder')

    options, args = parser.parse_args()

    if not options.conllFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    flist = getInputfiles(options.conllFolder)
    conllTokens = []
    for cf in flist:
        localTokens = parsePDTBFile(cf)
        conllTokens += localTokens
    
    
    printStats(conllTokens)
