#!/usr/bin/env python3

import sys
from optparse import OptionParser
import os
import codecs
import lxml.etree
from collections import defaultdict
import re

"""
Converts the Conano inline xml format to standoff xml. Can be run on single files, or on folder of conano output files (see run options).
"""

"""
Known PCC issues at time of coding this (29-05-2018):

- maz-7690.xml relation id 4 contains only ext arg (no connective and int arg)
- maz-9207.xml relation id 14 contains only int arg
- maz-17242.xml relation id 30 contains only ext arg
- maz-1423.xml relation id 11 has empty sense

"""

import treetaggerwrapper
ttagger = treetaggerwrapper.TreeTagger(TAGLANG='de') # DANGER! DANGER! HIGH VOLTAGE! ELECTRIC SIX! (hard-coded to german, watch out)
# http://treetaggerwrapper.readthedocs.io/en/latest/



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
        
tokenId = 1

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


def convert_latin1_to_utf8(f):

    utf8_version_filename = os.path.join('/tmp/', os.path.basename(f) + '_utf-8.xml') # causes this thing to work only in linux
    with codecs.open(f, "r", "iso-8859-15") as sourceFile:
        with codecs.open(utf8_version_filename, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.readlines()
                if not contents:
                    break
                if re.match('<\?xml version="1\.0" encoding="ISO-8859-15"\?>\n', contents[0]):
                    contents[0] = '<?xml version="1.0" encoding="UTF-8"?>\n'
                targetFile.write(''.join(contents))
    return utf8_version_filename


def parseConnectorFile(connectorxml, tokenizeFlag):

    try: # some files were ISO-8859-15
        codecs.open(connectorxml, 'r', 'utf-8').readlines()
    except UnicodeDecodeError:
        connectorxml = convert_latin1_to_utf8(connectorxml)
        
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(connectorxml, parser = xmlParser)
    relations = defaultdict(DiscourseRelation)
    tokens = extractTokens(tree.getroot(), defaultdict(str), relations, tokenizeFlag)
    global tokenId
    tokenId = 1
    return tokens, relations

    
        
def appendToRelation(tokenId, conn, node, relations, nodeTag, nodeType, nodeId):

    # passing on nodeTag and nodeType from the caller here, because getting it at this point does not work with tailing text...
    if nodeTag == 'discourse': # at toplevel, nothing to add (not part of a relation)
        return
    rel = relations[nodeId]
    if conn:
        rel.addConnectiveToken(tokenId)
        sense = None
        if node.get('relation') == None: # this is the modifier case (maz-5007)
            sense = node.getparent().get('relation')
        else:
            sense = node.get('relation')
        rel.setSense(node.get('relation'))
    elif nodeTag == 'unit':
        if nodeType == 'int':
            rel.addIntArgToken(tokenId)
        elif nodeType == 'ext':
            rel.addExtArgToken(tokenId)
            
    ou = getOverarchingUnits(node, defaultdict(set))
    for unit in ou:
        for _type in ou[unit]:
            if _type == 'int':
                if not tokenId in relations[unit].intArgTokens: # not sure if this was due to an error in PCC (in maz-00001.xml relation id 11 for example, where the int arg is embedded in the ext arg, which does not seem to make sense), or a bug in my code. Fixed duplicates and should not do any harm in any case.
                    relations[unit].addIntArgToken(tokenId)
            elif _type == 'ext':
                if not tokenId in relations[unit].extArgTokens: # same here (see above)
                    relations[unit].addExtArgToken(tokenId)
            elif _type == 'conn':
                if not tokenId in relations[unit].connectiveTokens:
                    relations[unit].addConnectiveToken(tokenId)
                
def getOverarchingUnits(node, d):

    if node.getparent() is not None:
        parent = node.getparent()
        if parent.get('id'):
            _type = None
            if parent.get('type'):
                _type = parent.get('type')
            elif parent.tag == 'connective':
                _type = 'conn'
            d[parent.get('id')].add(_type)
        getOverarchingUnits(parent, d)
    return d

        
def extractTokens(node, d, relations, tokenizeFlag):

    global tokenId
    conn = False
    if node.get('id'):
        if not node.get('id') in relations:
            relations[node.get('id')] = DiscourseRelation(node.get('id'))
    nodeTag = node.tag
    nodeType = node.get('type')
    nodeId = node.get('id')
    if node.text:
        if node.tag == 'connective':
            conn = True
        # PCC (maz-5007.xml for ex.) can also contain modifier tags
        if node.tag == 'modifier':
            conn = True
            nodeTag = node.getparent().tag
            nodeType = node.getparent().get('type')
            nodeId = node.getparent().get('id')
        textContent = node.text.strip()
        if textContent:
            tokens = []
            if tokenizeFlag:
                # we marked connectives with ** in plain text before Robin started annotating, so removing them here:
                textContent = re.sub('\*\*', '', textContent)
                tagged = ttagger.tag_text(textContent) # tagging is a bit overkill, but this was the first python wrapper for the treetagger I found. And since in the PCC for tokenization probably the treetagger is used (at least for tagging, as it says in the 2004 paper, so I assume for tokenization as well), decided to just go with this.
                for tt in tagged:
                    if re.search('\t', tt):
                        word, tag, lemma = tt.split('\t')
                        tokens.append(word)
                    elif re.match(r'<repdns text="[^"]+"\s+/>', tt): # this was the case for a url at some point. 
                        word =  re.match(r'<repdns text="([^"]+)"\s+/>', tt).groups()[0]
                        tokens.append(word)
                    tokens.append(word)
            if not tokenizeFlag:
                tokens = textContent.split()
            for token in tokens:
                appendToRelation(tokenId, conn, node, relations, nodeTag, nodeType, nodeId)
                d[tokenId] = token
                tokenId += 1
        conn = False
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, d, relations, tokenizeFlag)
    if node.tail:
        nodeTag = node.getparent().tag
        nodeType = node.getparent().get('type')
        if node.tail.strip():
            tokens = []
            if tokenizeFlag:
                textContent = re.sub('\*\*', '', node.tail.strip())
                tagged = ttagger.tag_text(textContent)
                for tt in tagged:
                    if re.search('\t', tt):
                        word, tag, lemma = tt.split('\t')
                        tokens.append(word)
                    elif re.match(r'<repdns text="[^"]+"\s+/>', tt): # this was the case for a url at some point. 
                        word =  re.match(r'<repdns text="([^"]+)"\s+/>', tt).groups()[0]
                        tokens.append(word)
            if not tokenizeFlag:
                tokens = node.tail.strip().split()
            for token in tokens:
                appendToRelation(tokenId, conn, node, relations, nodeTag, nodeType, nodeId)
                d[tokenId] = token
                tokenId += 1
    return d

def relationIsComplete(relation):

    if relation.sense == None:
        return False
    if relation.connectiveTokens == []:
        return False
    if relation.intArgTokens == []:
        return False
    if relation.extArgTokens == []:
        return False
    return True

def extractProcessingInstructions(xmlfile):

    try: # some files were ISO-8859-15
        codecs.open(xmlfile, 'r', 'utf-8').readlines()
    except UnicodeDecodeError:
        xmlfile = convert_latin1_to_utf8(xmlfile)
    
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(xmlfile, parser = xmlParser)
    
    return tree.xpath("//processing-instruction()")

def convertStrKeysToInts(d):

    nd = defaultdict()
    for k, v in d.items():
        nd[int(k)] = v
        
    return nd

    
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('--folder', dest='folder', help='specify Folder with Conano output files')
    parser.add_option('--file', dest='_file', help='specify single Conano output file')
    parser.add_option('--outputfolder', dest='outputfolder', help='specify folder (created if it does not exist) to dump standoff xml')
    parser.add_option('-t', '--tokenize', dest='tokenize', action='store_true', help='Boolean option. If true, text is (re)tokenized.')
    
    options, args = parser.parse_args()

    if not options._file and not options.folder or not options.outputfolder:
        parser.print_help(sys.stderr)
        sys.exit(0)
        

    inputfiles = []
    if options._file:
        inputfiles.append(os.path.abspath(options._file))
    elif options.folder:
        inputfiles = getInputfiles(options.folder)

    outputfolder = os.path.abspath(options.outputfolder)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
        
    for f in inputfiles:
        tokens, relations = parseConnectorFile(f, options.tokenize)
        root = lxml.etree.Element('discourse')
        tokensnode = lxml.etree.Element('tokens')
        relationsnode = lxml.etree.Element('relations')
        for tid in tokens:
            tokennode = lxml.etree.Element('token')
            tokennode.set('id', str(tid))
            tokennode.text = tokens[tid]
            tokensnode.append(tokennode)

        # first convert relations keys to ints for proper sorting
        relations = convertStrKeysToInts(relations)

        for relId, rel in sorted(relations.items()):
            # at the moment of coding this, there were a few issues in PCC (see on top), which is why this check is necessary
            if relationIsComplete(rel):
                relationnode = lxml.etree.Element('relation')
                relationnode.set('relation_id', rel.relationId) # dubbed 'relationId' to not confuse it with token ids...
                relationnode.set('sense', rel.sense)
                connectivetokens = lxml.etree.Element('connective_tokens')
                for tid in sorted(rel.connectiveTokens):
                    conntoken = lxml.etree.Element('connective_token') # perhaps a bit verbose to generate node for every token, could also work with spans here (like MMAX)
                    conntoken.set('id', str(tid))
                    connectivetokens.append(conntoken)
                relationnode.append(connectivetokens)
                intargtokens = lxml.etree.Element('int_arg_tokens')
                for tid in rel.intArgTokens:
                    intargtoken = lxml.etree.Element('int_arg_token')
                    intargtoken.set('id', str(tid))
                    intargtokens.append(intargtoken)
                relationnode.append(intargtokens)
                extargtokens = lxml.etree.Element('ext_arg_tokens')
                for tid in rel.extArgTokens:
                    extargtoken = lxml.etree.Element('ext_arg_token')
                    extargtoken.set('id', str(tid))
                    extargtokens.append(extargtoken)
                relationnode.append(extargtokens)
                relationsnode.append(relationnode)

        root.append(tokensnode)
        root.append(relationsnode)

        doc = lxml.etree.ElementTree(root)
        pis = extractProcessingInstructions(f)
        # must be a better way to add processing instruction to a doc, but I couldn't find it...
        for pi in pis:
            piText = []
            for k, v in pi.attrib.items():
                piText.append('%s="%s"' % (k, v))
            doc.getroot().addprevious(lxml.etree.ProcessingInstruction('relations', ' '.join(piText)))

        outname = os.path.join(outputfolder, os.path.basename(f))
        doc.write(outname, xml_declaration=True, encoding='utf-8', pretty_print=True) 

