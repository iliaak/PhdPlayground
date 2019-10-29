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

syntaxTokenId = 0

class DiscourseRelation(object):

    def __init__(self, relationId):
        self.relationId = relationId
        self.connectiveTokens = []
        self.intArgTokens = []
        self.extArgTokens = []
        self.sense = None

    def addConnectiveToken(self, tid):
        self.connectiveTokens.append(tid)
    def addIntArgToken(self, tid):
        self.intArgTokens.append(tid)
    def addExtArgToken(self, tid):
        self.extArgTokens.append(tid)
    def setSense(self, sense):
        self.sense = sense
    def addExtArgSynfo(self, label, hasfullcoverage, spanningNodeText):
        self.extArgSynLabel = label
        self.extArgSynLabelIsExactMatch = hasfullcoverage
        self.extArgSpanningNodeText = spanningNodeText
    def addIntArgSynfo(self, label, hasfullcoverage, spanningNodeText):
        self.intArgSynLabel = label
        self.intArgSynLabelIsExactMatch = hasfullcoverage
        self.intArgSpanningNodeText = spanningNodeText

    def filterIntArgForConnectiveTokens(self):
        self.intArgTokens = [x for x in self.intArgTokens if not x in self.connectiveTokens]

class DiscourseToken:
    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token

    def setConnectiveBoolean(self, val):
        self.isConnective = val
    def setUnitId(self, val):
        self.unitId = val
    def setRelation(self, val):
        self.relation = val
    def setLemma(self, val):
        self.lemma = val
    def setPOS(self, val):
        self.pos = val
    def setIsIntArg(self, val):
        self.isIntArg = val
    def setIsExtArg(self, val):
        self.isExtArg = val
    def setSentenceId(self, val):
        self.sentenceId = val
    def addFullSentence(self, val):
        self.fullSentence = val
    def setConnectiveId(self, val):
        self.connectiveId = val
    def setSentencePosition(self, val):
        self.sentencePosition = val
    def setSyntaxSentenceId(self, val):
        self.syntaxSentenceId = val
    def setTerminalsId(self, val):
        self.terminalsId = val
    def setMorph(self, val):
        self.morph = val
    def setSyntaxNodeId(self, val):
        self.syntaxId = val
    def setStartIndex(self, val):
        self.characterStartIndex = val
    def setEndIndex(self, val):
        self.characterEndIndex = val



def parseStandoffConnectorFile(connectorxml):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(connectorxml, parser = xmlParser)
    tokenlist = []
    discourseRelations = []
    tid2dt = {}
    tokenoffset = 0
    for node in tree.getroot():
        if node.tag == 'tokens':
            for subnode in node:
                dt = DiscourseToken(subnode.get('id'), subnode.text)
                dt.setConnectiveBoolean(False)
                dt.setStartIndex(tokenoffset)
                dt.setEndIndex(tokenoffset + len(subnode.text))
                tokenoffset += len(subnode.text)+1
                tokenlist.append(dt)
                tid2dt[subnode.get('id')] = dt
        elif node.tag == 'relations':
            for subnode in node:
                dr = DiscourseRelation(subnode.get('id'))
                dr.setSense(subnode.get('sense'))
                for elem in subnode:
                    if elem.tag == 'connective_tokens':
                        for ct in elem:
                            dr.addConnectiveToken(ct.get('id'))
                            tid2dt[ct.get('id')].setConnectiveBoolean(True)
                    if elem.tag == 'int_arg_tokens':
                        for iat in elem:
                            dr.addIntArgToken(iat.get('id'))
                            tid2dt[iat.get('id')].setIsIntArg(True)
                    if elem.tag == 'ext_arg_tokens':
                        for eat in elem:
                            dr.addExtArgToken(eat.get('id'))
                            tid2dt[eat.get('id')].setIsExtArg(True)
                dr.filterIntArgForConnectiveTokens()
                discourseRelations.append(dr)
                
    return tokenlist, discourseRelations, tid2dt


def parseSyntaxFile(syntaxxml, tokenlist):

    global syntaxTokenId
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(syntaxxml, parser = xmlParser)
    sentencedict = defaultdict(str)
    for body in tree.getroot():
        for elemid, sentence in enumerate(body):
            sid = sentence.get('id')
            graph = sentence.getchildren()[0]
            terminalsNode = graph.find('.//terminals')
            nonterminalNodes = graph.find('.//nonterminals')
            tokenisedSentence = ' '.join([x.get('word') for x in terminalsNode])
            sentencedict[elemid] = tokenisedSentence
            subdict, catdict = getSubDict(nonterminalNodes)
            terminalnodeids = [x.get('id') for x in terminalsNode]
            maxId = max([int(re.sub('\D', '', re.sub(r'[^_]+_', '', x))) for x in terminalnodeids])
            for sentencePosition, t in enumerate(terminalsNode):
                sToken = t.get('word')
                dt = tokenlist[syntaxTokenId]
                dt.setSyntaxSentenceId(sid)
                if not sToken == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (syntaxxml, sToken, str(syntaxTokenId), tokenlist[syntaxTokenId].token, str(tokenlist[syntaxTokenId].tokenId)))
                    sys.exit(1)
                    
                dt.setLemma(t.get('lemma'))
                dt.setPOS(t.get('pos'))
                dt.setTerminalsId(t.get('id'))
                dt.setMorph(t.get('morph')) # whatever this is supposed to contain (empty in examples I've seen so far)
                dt.setSyntaxNodeId(t.get('id'))
                dt.addFullSentence(tokenisedSentence)
                dt.setSentenceId(elemid)
                dt.setSentencePosition(sentencePosition)
                
                syntaxTokenId += 1
    syntaxTokenId = 0 # reset at end of file          
    return tokenlist


def getSubDict(nonterminalnodes):

    d = {}
    cat = {}
    for nt in nonterminalnodes:
        edges = []
        for edge in nt:
            if edge.tag == 'edge': # some have secedges, which cause nesting/loops
                edges.append(edge.get('idref'))
        d[nt.get('id')] = edges
        cat[nt.get('id')] = nt.get('cat')
    return d, cat

def wrapTokensInSentences(pccTokens):

    sid = defaultdict(list)
    for pcct in pccTokens:
        sid[pcct.sentenceId].append(pcct)
    return sid
