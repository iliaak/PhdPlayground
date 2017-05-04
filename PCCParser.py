#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import lxml.etree
from itertools import chain

"""
Potsdam Commentary Corpus Parser for some initial playing around with the format
"""

class EDU:
    def __init__(self, text, isConnective):
        self.text = text
        self.isConnective = isConnective



def parseFile(pccxml, file2edulist):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(pccxml, parser = xmlParser)
    l = extractEDUs(tree.getroot(), [])
    file2edulist[pccxml] = l
    return file2edulist
    

def extractEDUs(node, l):

    if node.text:
        conn = False
        if node.tag == 'connective':
            conn = True
        textContent = node.text.strip()
        if textContent: # filtering out (empty) newlines
            edu = EDU(textContent, conn)
            l.append(edu)
            #conn = False
        conn = False
        if len(node):
            for subnode in node.getchildren():
                extractEDUs(subnode, l)
        
        if node.tail: # filtering out (empty) newlines
            if node.tail.strip():
                l.append(EDU(node.tail.strip(), conn))

    return l
    

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


def printPlaintext(file2edulist):

    for f in file2edulist:
        print('++++++++++ %s ++++++++++' % f)
        print(re.sub(r'\s+', ' ', ' '.join([x.text for x in file2edulist[f]])))
        print('\n')

def printConllGold(file2edulist):

    for f in file2edulist:
        print('++++++++++ %s ++++++++++' % f)
        # assuming that everything is (still) whitespace-tokenized
        for edu in file2edulist[f]:
            tokens = edu.text.split(' ')
            cb = edu.isConnective
            for i, t in enumerate(tokens):
                word, val = t, 'O'
                if cb:
                    if i == 0:
                        val = 'B-connective'
                    else:
                        val = 'I-connective'
                print('%s\t%s' % (word, val))
        print('\n')

def printStats(file2edulist):

    print('INFO: %i files processed.' % len(file2edulist))
    cn = 0
    ct = 0
    wc = 0
    educ = 0
    for f in file2edulist:
        eduList = file2edulist[f]
        educ += len(eduList)
        for edu in eduList:
            wc += len(edu.text.split(' '))
            if edu.isConnective:
                cn += 1
                ct += len(edu.text.split(' '))

    print('INFO: %i words.' % wc)
    print('INFO: %i discourse units.' % educ)
    print('INFO: %i connectors.' % cn)
    print('INFO: %i connector tokens.' % ct)
    
    
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-f', '--xmlFolder', dest='xmlFolder', help='specify PCC xml folder')

    options, args = parser.parse_args()

    if not options.xmlFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    inputfiles = getInputfiles(options.xmlFolder)
        
    file2edulist = defaultdict(list)
    for f in inputfiles:
        file2edulist = parseFile(f, file2edulist)

    
    #printPlaintext(file2edulist)

    #printConllGold(file2edulist) #TODO: debug this (to see if everything is correct)
    
    printStats(file2edulist)
