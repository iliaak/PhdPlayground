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
connectorTokenId = 1
syntaxTokenId = 0 # bit ugly, but because of zero-index list...

"""
class EDU:
    def __init__(self, text, isConnective):
        self.text = text
        self.isConnective = isConnective
"""

class DiscourseToken:
    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token
    # TODO: read up on proper way of doing this, think I'm mixing java and python too much
    def setIntOrExt(self, val):
        self.intOrExt = val
    def setUnitId(self, val):
        self.unitId = val
    def setRelation(self, val):
        self.relation = val
    def setType(self, val):
        self.segmentType = val
    def setLemma(self, val):
        self.lemma = val
    def setPOS(self, val):
        self.pos = val
    def setTerminalsId(self, val):
        self.terminalsId = val
    def setMorph(self, val):
        self.morph = val

        
def parseConnectorFile(connectorxml):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(connectorxml, parser = xmlParser)
    discourseTokens = extractTokens(tree.getroot(), [])
    global connectorTokenId
    connectorTokenId = 1 # reset at end of file
    return discourseTokens


def parseSyntaxFile(syntaxxml, tokenlist):

    global syntaxTokenId
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(syntaxxml, parser = xmlParser)
    for body in tree.getroot():
        for sentence in body:
            sid = sentence.get('id') # may want to add art_id and some others too here
            graph = sentence.getchildren()[0]
            terminalsNode = graph.find('.//terminals') # could also just take first child, but playing around with lxml a bit :)
            for t in terminalsNode:
                sToken = t.get('word')
                dt = tokenlist[syntaxTokenId]
                print('dt token:', dt.token, '\t', 'sToken:', sToken)
                if not sToken == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (syntaxxml, sToken, str(syntaxTokenId), tokenlist[syntaxTokenId].token, str(tokenlist[syntaxTokenId].tokenId))) # TODO debug this for more files
                    sys.exit(1)

                # TODO: continue here. This crashes on maz-10175 because of a nested structure, fix connector text extraction
                dt.setLemma(t.get('lemma'))
                dt.setPOS(t.get('pos'))
                dt.setTerminalsId(t.get('id'))
                dt.setMorph(t.get('morph')) # whatever this is supposed to contain (empty in examples I've seen so far)
                
                # may want to go on with nonterminals/deps here
                syntaxTokenId += 1
    syntaxTokenId = 0 # reset at end of file          
    return tokenlist
            
"""
def parseConnectorFile(pccxml, file2edulist):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(pccxml, parser = xmlParser)
    l = extractEDUs(tree.getroot(), [])
    file2edulist[pccxml] = l
    return file2edulist
"""
def createDiscourseToken(token, node, conn):

    global connectorTokenId
    dt = DiscourseToken(connectorTokenId, token)
    segmentType = ''
    if conn:
        segmentType = 'connective'
    else:
        segmentType = 'unit'
    dt.setType(segmentType)
    dt.setUnitId(node.get('id'))
    if segmentType == 'unit':
        pn = node.getparent()
        segmentType = node.get('type')
        if not segmentType: # this elaborate stuff is for tail cases, in which case lxml returns None for node.get('type')
            # TODO: debug if this procedure is correct!!!
            pn = node.getparent()
            if pn is not None and pn.tag == 'unit':
                segmentType = node.getparent().get('type')
        dt.setIntOrExt(segmentType)
    elif segmentType == 'connective':
        dt.setRelation(node.get('relation'))
    connectorTokenId += 1
    return dt
    

def extractTokens(node, l):

    if node.text:
        conn = False
        if node.tag == 'connective':
            conn = True
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            for token in tokens:
                dt = createDiscourseToken(token, node, conn)
                l.append(dt)
        conn = False
        if len(node):
            for subnode in node.getchildren():
                extractTokens(subnode, l)

        if node.tail:
            if node.tail.strip():
                tokens = node.tail.strip().split()
                for token in tokens:
                    dt = createDiscourseToken(token, node, conn)
                    l.append(dt)

    return l

"""                    
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
""" 

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
    

def getFileVersionsDict(connectorFiles, syntaxFiles):

    d = defaultdict(lambda : defaultdict(str))
    for f in connectorFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['connectors'] = f
    for f in syntaxFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['syntax'] = f

    # sanity check:
    die = False
    for name in d:
        if not len(d[name]['connectors']):
            sys.stderr.write('ERROR: Could not find annotation layer <connectors> for %s.\n' % name)
            die = True
        if not len(d[name]['syntax']):
            sys.stderr.write('ERROR: Could not find annotation layer <syntax> for %s.\n' % name)
            die = True
    if die:
        sys.stderr.write('FATAL ERROR: Please investigate annotation layers.\n')
        sys.exit(1)

    return d
        
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectorsFolder', dest='connectorsFolder', help='specify PCC connectors folder')
    parser.add_option('-s', '--syntaxFolder', dest='syntaxFolder', help='specify PCC syntax folder')

    options, args = parser.parse_args()

    if not options.connectorsFolder or not options.syntaxFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    connectorfiles = getInputfiles(options.connectorsFolder)
    syntaxfiles = getInputfiles(options.syntaxFolder)

    fileversions = getFileVersionsDict(connectorfiles, syntaxfiles) # makes a dict with filename 2 connectors, syntax, etc.
    
    
    file2tokens = defaultdict(list)
    for name in fileversions:
        tokenlist = parseConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        
        file2tokens[name] = tokenlist
        print('DEBUG: parsed %s' % name)
        """
        for dt in tokenlist:
            print('token id:', dt.tokenId)
            print('token:', dt.token)
            print('type:', dt.segmentType)
            print('unitId:', dt.unitId)
            if dt.segmentType == 'unit':
                print('intOrExt:', dt.intOrExt)
            elif dt.segmentType == 'connective':
                print('relation:', dt.relation)
            print('\n')
        """
        #print(' '.join([t.token for t in tokenlist]))
        
    #for fname in file2tokens:
        #print(fname)
        #for dt in file2tokens[fname]:
            #print('\t%s %s %s %s' % (dt.token, dt.segmentType, dt.pos, dt.lemma))
    
    # not sure if these printing functions still work after recent changes              
    #printPlaintext(file2edulist)

    #printConllGold(file2edulist) #TODO: debug this (to see if everything is correct)
    
    ##printStats(file2edulist)

    #TODO: combine this with other annotation layers, to print (for learning/inspiration purposes)
    # conll format output with features (i.e. pos-tag of connective, dep parse info, etc.)
    # TODO: fix structure based on EDU ID, isInternal, isExternal, etc...
    # battle plan: assuming that everything is tokenised (such that whitespace splitting results in proper tokens)
    # go through the connector file and the syntax file (rst file is the one we want to construct eventually, so maybe look into that at
    # a later point) , and map every token to either its unit id (connector file) or graph and t(erminal) id (syntax file)
    # the, going through the list of tokens top to bottom, I can link units (connectors) and postags, lemmas, etc (syntax)
