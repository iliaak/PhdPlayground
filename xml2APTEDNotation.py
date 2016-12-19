#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from pyparsing import nestedExpr
from lxml import etree

def parseXML(fh):

    parser = etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding='utf-8')
    tree = etree.parse(fh, parser=parser)
    root = tree.getroot()

    # strip off the text and the values of span and leaf, because I want to calculate distance based on structure only
    traverseTree(root)
    
    return root


def traverseTree(node):

    if node.get('span'):
        node.set('span', '0')
    elif node.get('leaf'):
        node.set('leaf', '0')
    if node.text:
        node.text = ''
    for subnode in node:
        traverseTree(subnode)

    

if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-x", "--xml", dest="xmlin", help="Specify file to get input xml from.")
    parser.add_option("-a", "--aptedForm", dest="aptedForm", help="Specify file to print APTED output format to.")
    
   
    options, args = parser.parse_args()
    
    if not options.xmlin or not options.aptedForm:
        parser.print_help(sys.stderr)
        sys.exit(1)

    newTree = parseXML(options.xmlin)
    treestring = str(etree.tostring(newTree), 'utf-8')
    treestring = re.sub(r'</\w+>', '}', treestring)
    treestring = re.sub(r'<', '{', treestring)
    treestring = re.sub(r'>', '', treestring)
    treestring = re.sub(r'\\n\s*', '', treestring)

    ofh = codecs.open(options.aptedForm, 'w', 'utf-8')
    ofh.write(treestring)
    ofh.close()
    
    
