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

_id = 1

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

def extractTokens(node, l):

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


def createAnnisXML(annisTokens):

    #start = '<?xml version="1.0" encoding="UTF-8"?><?relations relSet="Martin1992" lexURL="jar:file:/home/lisa/Arbeit/Conano-Distrib/conano.jar!/de/uni_potsdam/ling/coli/resources/ConnectorLexicon.xml"?>'
    
    discourseNode = lxml.etree.Element("discourse")
    for at in annisTokens:
        tokenNode = lxml.etree.Element('token')
        tokenNode.text = at.token
        #tokenNode.attrib['id'] = str(at.tokenId)
        if at._type:
            argType = ''
            if at._type == 'int':
                argType = '%s (%s)' % ('intArg', str(at.sid))
            elif at._type == 'ext':
                argType = '%s (%s)' % ('extArg', str(at.sid))
            tokenNode.attrib['discInfo'] = argType
        elif at.relation:
            tokenNode.attrib['discInfo'] = 'Connective: %s (%s)' % (at.relation, str(at.sid))
        #if at.sid:
            #tokenNode.attrib['discourseUnitId'] = at.sid
        discourseNode.append(tokenNode)

    root = lxml.etree.ElementTree(discourseNode)

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

    for f in flist:
        xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
        tree = lxml.etree.parse(f, parser = xmlParser)
        annisTokens = extractTokens(tree.getroot(), [])
        xml = createAnnisXML(annisTokens)
        xml.write(os.path.join(options.outputfolder, os.path.basename(f)), pretty_print=True, encoding='UTF-8')
        _id = 1 # reset after every doc
    
    
    
