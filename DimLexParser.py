#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import lxml.etree

class DimLex:
    def __init__(self, entryId, word):
        self.entryId = entryId
        self.word = word
        self.alternativeSpellings = []
        self.syncats = []
        self.connectiveReadingProbability = 1
        self.sense2Probs = defaultdict(float)
        
    def addAlternativeSpelling(self, alt):
        self.alternativeSpellings.append(alt)
        
    def addSynCat(self, syncat):
        self.syncats.append(syncat)

    def setConnectiveReadingProbability(self, p):
        self.connectiveReadingProbability = p
        
    def addSense(self, sense, prob):
        self.sense2Probs[sense] = prob

def parseXML(dimlexml):
    
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(dimlexml, parser = xmlParser)
    l = []
    for entry in tree.getroot():
        dl = DimLex(entry.get('id'), entry.get('word'))
        for orth in entry.find('orths').findall('orth'):
            dl.addAlternativeSpelling(orth.find('part').text)
        
        ambiguity = entry.find('ambiguity')
        non_connNode = ambiguity.find('non_conn')
        
        if non_connNode.text == '1':
            if 'freq' in non_connNode.attrib and 'anno_N' in non_connNode.attrib: # this may not be the case for markers which have both readings, but there is no frequency info for one or the other
                p = 1 - (float(non_connNode.get('freq')) / float(non_connNode.get('anno_N')))
                dl.setConnectiveReadingProbability(p)

        syns = entry.findall('syn')
        for syn in syns:
            dl.addSynCat(syn.find('cat').text)
            for sem in syn.findall('sem'):
                for sense in sem:
                    freq = sense.get('freq')
                    anno = sense.get('anno_N')
                    if not freq == '0' and not freq == '' and not anno == '0' and not anno == '':
                        ##pdtb3sense = sense.get('pdtb3_relation sense') # bug in lxml due to whitespace in attrib name, or by design?
                        pdtb3sense = sense.get('sense')
                        prob = 1 - (float(freq) / float(anno))
                        dl.addSense(pdtb3sense, prob)

        l.append(dl)
            
    return l



if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-d', '--dimlexml', dest='dimlexml', help='specify dimlex xml file')

    options, args = parser.parse_args()

    if not options.dimlexml:
        parser.print_help(sys.stderr)
        sys.exit(1)

    connectiveList = parseXML(options.dimlexml)
    for conn in connectiveList:
        print(conn.word)
        
