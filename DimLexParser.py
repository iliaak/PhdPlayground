#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import lxml.etree

"""
Parser for the DiMLex (https://github.com/discourse-lab/dimlex) XML lexicon. Just for some initial playing around and extracting some random stuff.
"""

class DimLex:
    def __init__(self, entryId, word):
        self.entryId = entryId
        self.word = word
        self.alternativeSpellings = defaultdict(lambda : defaultdict(str))
        self.syncats = []
        self.connectiveReadingProbability = 1
        self.sense2Probs = defaultdict(float)
        
    def addAlternativeSpelling(self, alt, singleOrPhrasal, contOrDiscont):
        self.alternativeSpellings[alt][singleOrPhrasal] = contOrDiscont
        
    def addSynCat(self, syncat):
        self.syncats.append(syncat)

    def setConnectiveReadingProbability(self, p):
        self.connectiveReadingProbability = p
        
    def addSense(self, sense, prob):
        self.sense2Probs[sense] = prob

    def addFunctionalAmbiguityInfo(self, v):
        self.functionalAmbiguity = v # as opposed to sense ambiguity...
        #print('deb8ugging:', self.word, v)
        
def parseXML(dimlexml):
    
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments=True)
    tree = lxml.etree.parse(dimlexml, parser = xmlParser)
    l = []
    for entry in tree.getroot():
        #print('debugging entry:', entry.get('id'))
        dl = DimLex(entry.get('id'), entry.get('word'))
        """
        if len(entry.find('orths')) == 0:
            print('ERROR: no orths for:', entry.get('id'))
            sys.exit()
        """
        for orth in entry.find('orths').findall('orth'):
            text = orth.find('part').text
            t1 = orth.find('part').get('type') # single or phrasal
            t2 = orth.get('type') # cont or discont
            dl.addAlternativeSpelling(text, t1, t2)
            
        
        ambiguity = entry.find('ambiguity')
        non_connNode = None
        if ambiguity is not None:
            non_connNode = ambiguity.find('non_conn')
        """
        else:
            print('ERROR: no ambiguity node for:', entry.get('id'))
            sys.exit()
        
        if non_connNode == None:
            print('ERROR: no non conn node for:', entry.get('id'))
            sys.exit()
        if entry.find('non_conn_reading') is None:
            print('ERROR: no non_conn_reading node found:', entry.get('id'))
            sys.exit()
            non_conn_readingNode = lxml.etree.Element('non_conn_reading')
            non_conn_readingNode.text = '0'
            entry.append(non_conn_readingNode)
        """
        if non_connNode is not None:
            dl.addFunctionalAmbiguityInfo(non_connNode.text)
            if non_connNode.text == '1':
                if 'freq' in non_connNode.attrib and 'anno_N' in non_connNode.attrib: # this may not be the case for markers which have both readings, but there is no frequency info for one or the other
                    p = 1 - (float(non_connNode.get('freq')) / float(non_connNode.get('anno_N')))
                    dl.setConnectiveReadingProbability(p)
        
        syns = entry.findall('syn')
        """
        if len(syns) == 0:
            print('ERROR: No syn entries for:', entry.get('id'))
            sys.exit()
        """
        for syn in syns:
            dl.addSynCat(syn.find('cat').text)
            """
            if len(syn.findall('sem')) == 0:
                print('ERROR: no sem entry for:', entry.get('id'))
                sys.exit()
            """
            for sem in syn.findall('sem'):
                for sense in sem:
                    #print('sense val:', sense.get('sense'))
                    freq = sense.get('freq')
                    anno = sense.get('anno_N')
                    if not freq == '0' and not freq == '' and not anno == '0' and not anno == '' and not anno == None:
                        ##pdtb3sense = sense.get('pdtb3_relation sense') # bug in lxml due to whitespace in attrib name, or by design?
                        pdtb3sense = sense.get('sense')
                        prob = 1 - (float(freq) / float(anno))
                        dl.addSense(pdtb3sense, prob)
                    elif freq == None and anno == None:
                        dl.addSense(sense.get('sense'), 0)

        l.append(dl)

    # write if things are auto-fixed:
    fixed_name = os.path.splitext(dimlexml)[0] + '_auto-fixed.xml'
    writer = lxml.etree.XMLParser(remove_blank_text=True)
    tree.write(fixed_name, pretty_print=True)
    
            
    return l



if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-d', '--dimlexml', dest='dimlexml', help='specify dimlex xml file')

    options, args = parser.parse_args()

    if not options.dimlexml:
        parser.print_help(sys.stderr)
        sys.exit(1)

    connectiveList = parseXML(options.dimlexml)
    single = 0
    phrasal = 0
    discontinuous = 0
    continuous = 0
    alts = 0
    for conn in connectiveList:
        #print(conn.word)
        for alt in conn.alternativeSpellings:
            alts += 1
            t1 = list(conn.alternativeSpellings[alt].keys())[0]
            t2 = conn.alternativeSpellings[alt][t1]
            if t1 == 'single':
                single += 1
            elif t1 == 'phrasal':
                phrasal += 1
                
            if t2 == 'cont':
                continuous += 1
            elif t2 == 'discont':
                discontinuous += 1    

    print('INFO: %s single, %s phrasal, (%s/%s).' % (str(single), str(phrasal), str(phrasal + single), str(alts)))
    print('INFO: %s continuous, %s discontinuous, (%s/%s).' % (str(continuous), str(discontinuous), str(continuous + discontinuous), str(alts)))
    

    # validate against dtd:
    #"""
    parser = lxml.etree.XMLParser(dtd_validation=False, remove_blank_text=True) # dtd has to be in same dir as dimlexml
    tree = lxml.etree.parse(options.dimlexml, parser)
    for entry in tree.getroot():
        oid = 1
        # add canonical, type and on attributes to orths
        for orth in entry.find('orths'):
            if orth.get('canonical') is None:
                orth.set('canonical', "0")
            orth.set('type', 'cont')
            orth.set('onr', '%so%i' % (entry.get('id'), oid))
            oid += 1
            for part in orth:
                if re.search('\s', part.text):
                    part.set('type', 'phrasal')
                else:
                    part.set('type', 'single')
        # add focuspart if not there
        if not entry.find('focuspart'):
            fp = lxml.etree.Element('focuspart')
            fp.text = '_'
            entry.insert(2, fp)
        # add non_conn_reading node if not there
        ncr = entry.find('non_conn_reading')
        if ncr is None:
            ncr_node = lxml.etree.Element('non_conn_reading')
            entry.insert(3, ncr_node)
        else:
            for example in entry.find('non_conn_reading'):
                t = example.get('type')
                new_type = re.sub('\s', '-', t)
                example.set('type', new_type)
        # whitespaces in attribute values not allowed
        for snode in entry:
            if snode.tag == 'syn':
                for ssnode in snode:
                    if ssnode.tag == 'sem':
                        for ex in ssnode:
                            if ex.tag == 'example':
                                t = ex.get('type')
                                new_type = re.sub('\s', '-', t)
                                ex.set('type', new_type)
        # add ambiguity node if not there
        if entry.find('ambiguity') is None:
            amb = lxml.etree.Element('ambiguity')
            sa = lxml.etree.Element('sem_ambiguity')
            amb.insert(1, sa)
            na = lxml.etree.Element('non_conn')
            amb.insert(0, na)
            entry.insert(1, amb)
        # ensure stts is at correct position
        stts_index = entry.index(entry.find('stts'))
        if not stts_index == 4:
            stts = entry.find('stts')
            entry.remove(entry.find('stts'))
            entry.insert(4, stts)
        

    doc = lxml.etree.ElementTree(tree.getroot())
    doc.write('temp_czech.xml', xml_declaration=True, encoding='utf-8', pretty_print=True)
    #"""
    parser = lxml.etree.XMLParser(dtd_validation=True)
    tree = lxml.etree.parse('temp_czech.xml', parser)
