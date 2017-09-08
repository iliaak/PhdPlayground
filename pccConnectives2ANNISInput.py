#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import PCCParser
import lxml


"""
<markable id="markable_12" span="word_11..word_17" anaphor_antecedent="markable_4"  type="intArg"  />
<markable id="markable_4" span="word_11,word_12" anaphor_antecedent="empty"  type="connective" discourseSense="consequence"/>
<markable id="markable_7" span="word_18..word_32" anaphor_antecedent="markable_4"  type="extArg"/>

"""

wordCount = 0
markableCount = 1

class AnnisToken:
    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token
    def setType(self, _type):
        self._type = _type
    def setSegmentId(self, sid):
        self.sid = sid
    def setRelation(self, reln):
        self.relation = reln


def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

def createPrimmark(tokens, node, conn, _type, sid, relation):

    primmarkNode = lxml.etree.Element('markable')
    primmarkNode.attrib['id'] = "markable_" + str(markableCount)
    spanValue = ""
    if len(tokens) == 1:
        spanValue = "word_%s" % str(wordCount)
    elif len(tokens) == 2:
        spanValue = "word_%s,word_%s" % (str(wordCount), str(wordCount+1))
    else:
        spanValue = "word_%s..word_%s" % (str(wordCount), str(wordCount+len(tokens)))
    primmarkNode.attrib['span'] = spanValue

    anaphor_antecedentValue = 'empty'
    if node.tag == 'unit':
        #print("debugging node here:", lxml.etree.tostring(node))
        anaphor_antecedentValue = node.attrib['id']
    primmarkNode.attrib['anaphor_antecedentValue'] = anaphor_antecedentValue

    if _type:
        primmarkNode.attrib['type'] = _type
    
    if relation:
        primmarkNode.attrib['discourseSense'] = relation

    # adding the text here, only to not have to retrieve it through word spans later on... (alternative would be making a dictionary for bookkeeping...)
    primmarkNode.text = ' '.join(tokens)
        
    return primmarkNode

def createAnnisToken(token, node, conn, _type, sid, relation):

    global _id
    at = AnnisToken(_id, token)
    at.setType(_type)
    at.setSegmentId(sid)
    if conn:
        at.setRelation(relation)
    else:
        at.setRelation(None)
    _id += 1
    return at

def extractTokensBackup(node, l):

    conn = False
    relation = False
    if node.text:
        if node.tag == 'connective':
            conn = True
            relation = node.get('relation')
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            for token in tokens:
                at = createAnnisToken(token, node, conn, node.get('type'), node.get('id'), relation)
                l.append(at)
        conn = False
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, l)

    if node.tail:
        if node.tail.strip():
            tokens = node.tail.strip().split()
            for token in tokens:
                at = createAnnisToken(token, node, conn, node.getparent().get('type'), node.getparent().get('id'), relation)
                l.append(at)
        
    return l



def extractTokens(node, l):

    global wordCount
    global markableCount
    conn = False
    relation = False
    if node.text:
        if node.tag == 'connective':
            conn = True
            relation = node.get('relation')
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            pm = createPrimmark(tokens, node, conn, node.get('type'), node.get('id'), relation)
            wordCount += len(tokens)
            markableCount += 1
            #for token in tokens:
                #at = createAnnisToken(token, node, conn, node.get('type'), node.get('id'), relation)
            l.append(pm)
            
            
        conn = False
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, l)

    if node.tail:
        if node.tail.strip():
            tokens = node.tail.strip().split()
            #for token in tokens:
                #at = createAnnisToken(token, node, conn, node.getparent().get('type'), node.getparent().get('id'), relation)
            pm = createPrimmark(tokens, node, conn, node.getparent().get('type'), node.getparent().get('id'), relation)
            wordCount += len(tokens)
            markableCount += 1
            l.append(pm)
            
        
    return l

def createAnnisXML(annisTokens):

    #start = '<?xml version="1.0" encoding="UTF-8"?><?relations relSet="Martin1992" lexURL="jar:file:/home/lisa/Arbeit/Conano-Distrib/conano.jar!/de/uni_potsdam/ling/coli/resources/ConnectorLexicon.xml"?>'
    """
    <markable id="markable_12" span="word_11..word_17" anaphor_antecedent="markable_4"  type="intArg"  />
    <markable id="markable_4" span="word_11,word_12" anaphor_antecedent="empty"  type="connective" discourseSense="consequence"/>
    <markable id="markable_7" span="word_18..word_32" anaphor_antecedent="markable_4"  type="extArg"/>
    """
    
    discourseNode = lxml.etree.Element("discourse")
    for at in annisTokens:
        markableNode = lxml.etree.Element('markable')
        markableNode.attrib['span'] = at.get('span')
        ### DEBGUGGING MEASURE: (delete)
        markableNode.text = at.text
        ###
        markableNode.attrib['anaphor_antecedent'] = at.get('anaphor_antecedentValue')
        add = False
        if at.get('type') == 'int':
            markableNode.attrib['discInfo'] = '%s (%s)' % ('intArg', at.get('id'))
            add = True
        elif at.get('type') == 'ext':
            markableNode.attrib['discInfo'] = '%s (%s)' % ('extArg', at.get('id'))
            add = True
        elif at.get('discourseSense'):
            markableNode.attrib['discInfo'] = 'Connective: %s (%s)' % (at.attrib['discourseSense'], at.get('id'))
            add = True
        if add:
            discourseNode.append(markableNode)

    root = lxml.etree.ElementTree(discourseNode)

    #TODO: Span ids seem to be ok now. anaphor_antecedent however is not. As is the markable_ id that is used as referrer in discInfo... (extArg and intArg and conn should have the same one, now it just keeps on incrementing (guess this corresponds to the bug in anaphor antecedent markable id...
    
    return root

    
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-f', '--inputfolder', dest='inputfolder', help='specify input folder')
    parser.add_option('-o', '--outputfolder', dest='outputfolder', help='specify output folder')
    
    options, args = parser.parse_args()

    if not options.inputfolder or not options.outputfolder:
        parser.print_help(sys.stderr)
        sys.exit(1)


    flist = getInputfiles(options.inputfolder)

    global wordCount
    global markableCount
    for f in flist:
        xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
        tree = lxml.etree.parse(f, parser = xmlParser)
        wordCount = 0
        markableCount = 1 # reset both after every file (TODO; check if wordcount spans should be zero based
        annisTokens = extractTokens(tree.getroot(), [])
        xml = createAnnisXML(annisTokens)
        xml.write(os.path.join(options.outputfolder, os.path.basename(f)), pretty_print=True, encoding='UTF-8')
        #_id = 1 # reset after every doc
    
    
    
