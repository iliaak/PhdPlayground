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
rstTokenId = 0 # idem



class DiscourseToken:
    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token
    # TODO: read up on proper way of doing this, think I'm mixing java and python too much
    def setIntOrExt(self, val):
        self.intOrExt = val
    def setMultiWordBoolean(self, val):
        self.isMultiWord = val
    def setConnectiveBoolean(self, val):
        self.isConnective = val
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
    def setRstSegmentId(self, val):
        self.rstSegmentId = val
    def setRstParent(self, val):
        self.rstParent = val
    def setRelname(self, val):
        self.relname = val

        
        
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
                #print('dt token:', dt.token, '\t', 'sToken:', sToken)
                if not sToken == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (syntaxxml, sToken, str(syntaxTokenId), tokenlist[syntaxTokenId].token, str(tokenlist[syntaxTokenId].tokenId)))
                    sys.exit(1)

                dt.setLemma(t.get('lemma'))
                dt.setPOS(t.get('pos'))
                dt.setTerminalsId(t.get('id'))
                dt.setMorph(t.get('morph')) # whatever this is supposed to contain (empty in examples I've seen so far)
                
                # may want to go on with nonterminals/deps here
                syntaxTokenId += 1
    syntaxTokenId = 0 # reset at end of file          
    return tokenlist

def parseRSTFile(rstxml, tokenlist):

    global rstTokenId # not sure why I made this global anymore, think it's not needed in this case
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(rstxml, parser = xmlParser)
    body = tree.getroot().find('.//body')
    for node in body:
        if node.tag == 'segment':
            sId = node.get('id')
            sParent = node.get('parent')
            sRelname = node.get('relname')
            tokens = node.text.split()
            for token in tokens:
                dt = tokenlist[rstTokenId]
                if not token == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (rstxml, sToken, str(rstTokenId), tokenlist[rstTokenId].token, str(tokenlist[rstTokenId].tokenId)))
                    sys.exit(1)
                dt.setRstSegmentId(sId)
                dt.setRstParent(sParent)
                dt.setRelname(sRelname)
                rstTokenId += 1
            # may want to add non-terminal rst tree info here
            
    rstTokenId = 0
    return tokenlist


  
def createDiscourseToken(token, node, conn, multiWordBool):

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
    dt.setMultiWordBoolean(multiWordBool)
    dt.setConnectiveBoolean(conn) # a bit redundant, because the type also contains this info, but this is easier for if checks...
    connectorTokenId += 1
    return dt
    

def extractTokens(node, l):

    conn = False
    if node.text:
        if node.tag == 'connective':
            conn = True
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            multiWordBool = False
            if len(tokens) > 1:
                multiWordBool = True
            for token in tokens:
                dt = createDiscourseToken(token, node, conn, multiWordBool)
                l.append(dt)
        conn = False
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, l)

    if node.tail:
        if node.tail.strip():
            tokens = node.tail.strip().split()
            multiWordBool = False
            if len(tokens) > 1:
                multiWordBool = True
            for token in tokens:
                dt = createDiscourseToken(token, node, conn, multiWordBool)
                l.append(dt)
        
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
    

def getFileVersionsDict(connectorFiles, syntaxFiles, rstFiles):

    d = defaultdict(lambda : defaultdict(str))
    for f in connectorFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['connectors'] = f
    for f in syntaxFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['syntax'] = f
    for f in rstFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['rst'] = f

    # sanity check:
    die = False
    for name in d:
        if not len(d[name]['connectors']):
            sys.stderr.write('ERROR: Could not find annotation layer <connectors> for %s.\n' % name)
            die = True
        if not len(d[name]['syntax']):
            sys.stderr.write('ERROR: Could not find annotation layer <syntax> for %s.\n' % name)
            die = True
        if not len(d[name]['rst']):
            sys.stderr.write('ERROR: Could not find annotation layer <rst> for %s.\n' % name)
            die = True
    if die:
        sys.stderr.write('FATAL ERROR: Please investigate annotation layers.\n')
        sys.exit(1)

    return d

def debugPrint(file2tokens):

    for f in file2tokens:
        print(f)
        for dt in file2tokens[f]:
            tokenId = dt.tokenId
            token = dt.token
            intOrExt = '___'
            if hasattr(dt, 'intOrExt'): # intOrExt is only specified for EDUs (not for connectives)
                intOrExt = dt.intOrExt
            relation = '___'    
            if (hasattr(dt, 'relation')): # relation is only specified for connectives
                relation = dt.relation
            sType = dt.segmentType
            lemma = dt.lemma
            pos = dt.pos
            rstParent = dt.rstParent
            rstRelation = dt.relname
            
            print('\t%s %s %s %s %s %s %s %s' % (str(tokenId), token, sType, pos, intOrExt, relation, lemma, rstRelation))
            

def printConnectiveStats(file2tokens):

    singleWordConnectives = defaultdict(int)
    multiWordConnectives = defaultdict(int)

    for f in file2tokens:
        for dt in file2tokens[f]:
            if dt.isConnective:
                if not dt.isMultiWord:
                    singleWordConnectives[dt.token] += 1
                else:
                    pass # do multiword stuff here (getting i+1 etc.)


    connpostags = defaultdict(lambda : defaultdict(float))
    nonconnpostags = defaultdict(lambda : defaultdict(float))
    for swc in singleWordConnectives:
        nonconnreading = 0
        connreading = 0
    
        for f in file2tokens:
            for dt in file2tokens[f]:
                if dt.token == swc:
                   if dt.isConnective:
                       connreading += 1
                   else:
                       nonconnreading += 1
                # TODO: include pos tag, lemma, rst-relation etc here
                
        print('Single Word Connective:', swc)
        print('\tconnective ratio: %i of %i' % (connreading, connreading + nonconnreading))  
        print('\n')
        
        
            
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectorsFolder', dest='connectorsFolder', help='specify PCC connectors folder')
    parser.add_option('-s', '--syntaxFolder', dest='syntaxFolder', help='specify PCC syntax folder')
    parser.add_option('-r', '--rstFolder', dest='rstFolder', help='specify PCC RST folder')

    options, args = parser.parse_args()

    if not options.connectorsFolder or not options.syntaxFolder or not options.rstFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    connectorfiles = getInputfiles(options.connectorsFolder)
    syntaxfiles = getInputfiles(options.syntaxFolder)
    rstfiles = getInputfiles(options.rstFolder)
    
    fileversions = getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles) # makes a dict with filename 2 connectors, syntax, etc.
    
    file2tokens = defaultdict(list)
    for name in fileversions:
        tokenlist = parseConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = parseRSTFile(fileversions[name]['rst'], tokenlist)
        file2tokens[name] = tokenlist


    debugPrint(file2tokens)
    
    #printConnectiveStats(file2tokens)

    
            
            
    # not sure if these printing functions still work after recent changes              
    #printPlaintext(file2edulist)

    #printConllGold(file2edulist) #TODO: debug this (to see if everything is correct)
    
    ##printStats(file2edulist)

    
    
