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

def parseRST(rstring):

    rstList = nestedExpr('(', ')').parseString(rstring).asList()
    xmldoc = constructXMLFromNextedList(rstList)

    return xmldoc


def constructXMLFromNextedList(rstList):

    xml = etree.Element("root")
    for root in rstList:
        for item in root:
            if type(item) == list:
                if item[0] == "span":
                    t = "%s,%s" % (item[1], item[2])
                    xml.attrib['span'] = t
                else:    
                    traverseTree(item, xml)

    doc = etree.ElementTree(xml)

    return doc


def traverseTree(t, xml):

    currentType = t[0]
    granularity = t[1][0]
    rel2par = t[2][0]
    node = etree.Element(currentType)
    if granularity == "span":
        # there is a deeper lever, add this one and then proceed
        node.set('span', "%s,%s" % (t[1][1], t[1][2]))
        node.set('rel2par', t[2][1])
        xml.append(node)
        for item in t[3:]:
            traverseTree(item, node)
        
    
    elif granularity == "leaf":
        # we are at terminal level
        node.set('leaf', t[1][1])
        node.set('rel2par', t[2][1])
        node.text = re.sub(r'_!$', '', re.sub(r'^_!', '', ' '.join(t[3][1:])))
        xml.append(node)

            

if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-t", "--rsttree", dest="rsttree", help="Specify file containing an rst tree.")
    parser.add_option("-x", "--xml", dest="xmlout", help="Specify file to print output xml to.")
   
    options, args = parser.parse_args()
    
    if not options.rsttree or not options.xmlout:
        parser.print_help(sys.stderr)
        sys.exit(1)

    xmldoc = parseRST(codecs.open(options.rsttree, 'r', 'utf-8').read())
    xmldoc.write(options.xmlout, xml_declaration=True, encoding="utf-8", pretty_print=True)

    #TODO: maybe add a flag to not print the actual text content, to be sure that at later sage/when comparing, the text is not used for clustering...
