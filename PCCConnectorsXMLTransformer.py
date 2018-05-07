#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from lxml import etree
from lxml import builder
from xml.sax.saxutils import escape


def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

    

def getIdChain(node, l):

    if node.get('id'):
        tp = None
        if node.tag == 'connective':
            tp = 'connective:%s' % node.get('relation')
        elif node.tag == 'unit':
            tp = node.get('type')
        l.append((node.get('id'), tp))
    if node.getparent() is not None and not node.getparent().get('id') == node.get('id'): # connectives are always nested inside a unit tag
        getIdChain(node.getparent(), l)
    return l
                
    
def processRelations(idchain, tid):

    for rel in idchain:
        if rel[1].startswith('connective'):
            rid2conn[rel[0]].append(tid)
            rid2sense[rel[0]] = re.sub('connective:', '', rel[1])
        elif rel[1] == 'int':
            rid2int[rel[0]].append(tid)
        elif rel[1] == 'ext':
            rid2ext[rel[0]].append(tid)
        
def extractTokens(node, l):

    global tid
    if node.text:
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            idchain = getIdChain(node, [])
            for token in tokens:
                tid2token[tid] = token
                l.append((token, idchain))
                processRelations(idchain, tid)
                tid += 1

        
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, l)

    if node.tail:
        if node.tail.strip():
            tokens = node.tail.strip().split()
            idchain = getIdChain(node.getparent(), []) # this is an annoying property of lxml; text in the tail. When processing this actual node (instead of the parent node) it takes the attributes of the embedded connective
            for token in tokens:
                tid2token[tid] = token
                l.append((token, idchain))
                processRelations(idchain, tid)
                tid += 1

    return l
"""
def createTextXML(f, tokens):

    textXMLroot = etree.Element('paula', version='1.0')
    header = etree.Element('header', paula_id=os.path.basename(f)+'.text.xml', type='text')
    textXMLroot.append(header)
    body = etree.Element('body')
    body.text = ' '.join([x[0] for x in tokens])
    textXMLroot.append(body)

    return textXMLroot
"""
"""
# these were attempts at generating paula xml. Would need to read up a bit more on paula structure to get that (but would in principe not be much more difficult from here...
def createTokenXML(f, tokens):

    charOffset = 1
    tokenXMLroot = etree.Element('paula', version='1.0')
    header = etree.Element('header', paula_id=os.path.basename(f)+'.tok.xml')
    tokenXMLroot.append(header)

    marklistmap = {
        'xlink': 'http://www.w3.org/1999/xlink',
        'base': os.path.basename(f)+'.text.xml'
    }
    
    markList = etree.Element('markList', nsmap=marklistmap, type='tok') # this makes both NSes xmlns, whereas in the orig paula, one is xml only... hmm, let's proceed for now and see if this works as well for pepper...
    
    for tid, token in enumerate(tokens):
        mark = etree.Element('mark', id='tok_%i' % tid)
        # TODO: debug if this charrOffset thing works well with newlines
        start = charOffset
        end = start + len(token)
        mark.attrib['href'] = '#xpointers(string-range(//body,'',%i,%i))' % (start, end)
        markList.append(mark)
        markList.append(etree.Comment(re.sub('-', '_', token[0])))
        charOffset += len(token) + 1
    tokenXMLroot.append(markList)
    
    #print(etree.tostring(markList))
    return tokenXMLroot
"""
"""
def createRelationXML(f, tid2token, rid2conn, rid2int, rid2ext, rid2sense):

    relationXMLroot = etree.Element('paula', version='1.0')
    header = etree.Element('header', paula_id=os.path.basename(f)+'.relations')
    relationXMLroot.append(header)
    structList = etree.Element('structList', type='struct', nsmap={'xlink':'http://www.w3.org/1999/xlink'}) # this is modelled on rst.maz-4282.struct.xml
    
    for rel in rid2sense:
        sense = rid2sense[rel]
        intargtokens = rid2int[rel]
        extargtokens = rid2ext[rel]
        conntokens = rid2conn[rel]
        
        relnode = etree.Element('struct', id='rel_%s' % rel)
        for token in conntokens:
            pass#rel = etree.Element('rel', id='conntoken_%i' % token, 'href'='%s.tok.xml#tok_%i' % (os.path.basename(f), i))
            #find out why this goes wrong!!!!
"""
def getmmaxspan(reltokens):
    
    span = None
    if len(reltokens) == 1: # highly unlikely, but you never know
        span = 'word_%i' % reltokens[0]
    elif len(reltokens) == 2: # also quite unlikely
        span = 'word_%i,word_%i' % (reltokens[0], reltokens[1])
    else:
        if iscontinuous(reltokens):
            span = 'word_%i..word_%i' % (reltokens[0], reltokens[len(reltokens)-1])
        else:
            # hope the following works (i.e. is valid mmax)
            span =','.join(['word_%i' % x for x in reltokens])

    return span
    
            
def createPrimmarkXML(f, tokens):

    root = etree.Element('markables', xmlns='www.eml.org/NameSpaces/primmark')
    # markable for complete relation, and markables for parts, let's see if that works...
    for rid in rid2sense:
        reltokens = sorted(rid2conn[rid] + rid2int[rid] + rid2ext[rid])
        relspan = getmmaxspan(reltokens)
        relationmarkable = etree.Element('markable', span=relspan, id=str(rid), sense=rid2sense[rid], type='discourse_relation')
        root.append(relationmarkable)
        # also create separate int, ext and conn markables (note that there is some redundancy...):
        if rid2conn[rid]:
            connmarkable = etree.Element('markable', span=getmmaxspan(rid2conn[rid]), id=str(rid)+'_connective', sense=rid2sense[rid], type='connective')
            root.append(connmarkable)
        if rid2int[rid]:
            intmarkable = etree.Element('markable', span=getmmaxspan(rid2int[rid]), id=str(rid)+'_intArg', type='intArg')
            root.append(intmarkable)
        if rid2ext[rid]:
            extmarkable = etree.Element('markable', span=getmmaxspan(rid2ext[rid]), id=str(rid)+'_extArg', type='extArg')
            root.append(extmarkable)
    return root

        
def iscontinuous(l):

    for i in range(len(l)-1):
        if l[i+1] - l[i] == 1:
            pass
        else:
            return False
    return True

def createWordsXML(f, tokens):

    root = etree.Element('words')
    for i, token in enumerate(tokens):
        word = etree.Element('word', id='word_%s' % (i+1))
        word.text = token[0]
        root.append(word)
    return root

def createSentenceXML(f, tokens):

    root = etree.Element('markables', xmlns='www.eml.org/NameSpaces/primmark')
    for rid in rid2sense:
        reltokens = sorted(rid2conn[rid] + rid2int[rid] + rid2ext[rid])
        relspan = getmmaxspan(reltokens)
        relationmarkable = etree.Element('markable', span=relspan, id=str(rid))
        root.append(relationmarkable)
        # also create separate int, ext and conn markables (note that there is some redundancy...):
        if rid2conn[rid]:
            connmarkable = etree.Element('markable', span=getmmaxspan(rid2conn[rid]), id=str(rid)+'_connective')
            root.append(connmarkable)
        if rid2int[rid]:
            intmarkable = etree.Element('markable', span=getmmaxspan(rid2int[rid]), id=str(rid)+'_intArg')
            root.append(intmarkable)
        if rid2ext[rid]:
            extmarkable = etree.Element('markable', span=getmmaxspan(rid2ext[rid]), id=str(rid)+'_extArg')
            root.append(extmarkable)
        
    return root

    
    
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-f', '--inputfolder', dest='inputfolder', help='specify input folder')
    parser.add_option('-o', '--outputfolder', dest='outputfolder', help='specify output folder')
    parser.add_option('--format', dest='outputformat', help='specify output format. Currently supported: "generic", "mmax2"')
    
    options, args = parser.parse_args()

    if not options.inputfolder or not options.outputfolder or not options.outputformat:
        parser.print_help(sys.stderr)
        sys.exit(1)


    flist = getInputfiles(options.inputfolder)
    basedatafolder = os.path.join(options.outputfolder, 'basedata')
    markablesfolder = os.path.join(options.outputfolder, 'markables')
    
    if options.outputformat == 'mmax2':
        if not os.path.exists(basedatafolder):
            os.makedirs(basedatafolder)
        if not os.path.exists(markablesfolder):
            os.makedirs(markablesfolder)

    xmlParser = etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
        
    for f in flist:
        tid = 1
        rid2conn = defaultdict(list)
        rid2int = defaultdict(list)
        rid2ext = defaultdict(list)
        tid2token = defaultdict(str)
        rid2sense = defaultdict(str)
        
        tree = etree.parse(f, parser = xmlParser)
        tokens = extractTokens(tree.getroot(), [])

        # generic is supposed to be what I assume as pretty intuitive standoff XML. Pepper didn't agree though (GenericXMLImporter didn't accept it)
        if options.outputformat == 'generic':
    
            standoffXMLRootNode = etree.Element("discourse")
            for tid in sorted(tid2token):
                tokenNode = etree.Element('token')
                tokenNode.attrib['id'] = str(tid)
                tokenNode.text = tid2token[tid]
                standoffXMLRootNode.append(tokenNode)
            for rid in sorted(rid2conn):
                relationNode = etree.Element('relation')
                relationNode.attrib['id'] = str(rid)
                relationNode.attrib['sense'] = rid2sense[rid]
                connNode = etree.Element('connectiveTokens')
                intNode = etree.Element('intArgTokens')
                extNode = etree.Element('extArgTokens')
                for tid in rid2conn[rid]:
                    connTokenNode = etree.Element('token')
                    connTokenNode.attrib['id'] = str(tid)
                    connNode.append(connTokenNode)
                for tid in rid2int[rid]:
                    intTokenNode = etree.Element('token')
                    intTokenNode.attrib['id'] = str(tid)
                    intNode.append(intTokenNode)
                for tid in rid2ext[rid]:
                    extTokenNode = etree.Element('token')
                    extTokenNode.attrib['id'] = str(tid)
                    extNode.append(extTokenNode)
                relationNode.append(connNode)
                relationNode.append(intNode)
                relationNode.append(extNode)
                standoffXMLRootNode.append(relationNode)
            standofftree = etree.ElementTree(standoffXMLRootNode)
            standofftree.write(os.path.join(options.outputfolder, os.path.basename(f)), pretty_print=True, encoding='UTF-8', xml_declaration = True)

        elif options.outputformat == 'mmax2':

            # this generates only a minimal set of files and assumes that styles, dtds, schemes etc are all there already
            wordsXMLcontent = createWordsXML(f, tokens)
            primmarkXMLcontent = createPrimmarkXML(f, tokens)
            sentenceXMLcontent = createSentenceXML(f, tokens)

            wordsXML = etree.ElementTree(wordsXMLcontent)
            wordsXML.write(os.path.join(basedatafolder, os.path.basename(f))[:-4]+'_words.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)

            primmarkXML = etree.ElementTree(primmarkXMLcontent)
            primmarkXML.write(os.path.join(markablesfolder, os.path.basename(f))[:-4]+'_primmark_level.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)

            sentenceXML = etree.ElementTree(sentenceXMLcontent)
            sentenceXML.write(os.path.join(markablesfolder, os.path.basename(f))[:-4]+'_sentence_level.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)


        else:
            sys.stderr.write('ERORR: format %s not supported. Dying now.\n' % options.outputformat)
            sys.exit(1)
