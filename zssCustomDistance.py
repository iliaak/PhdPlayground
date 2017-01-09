#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from lxml import etree
import zss
import treedist # this script has to be in the same folder as this one

TAXONOMYXML = ""
DEFAULT_DISTANCE = 0.0
KNOWNNODES = set()

def weird_dist(A, B):

    # this is where I can do the taxonomy-based distance, something like:
    if A == B:
        return 0
    else:
        if len(A) == 2 and len(B) == 2:
            if A[1] in KNOWNNODES and B[1] in KNOWNNODES:
                return treedist.getTreeDistance(TAXONOMY, A[1], B[1])
            else:
                return DEFAULT_DISTANCE
    return 0 # to cover all the other cases
        
class WeirdNode(object):

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node.my_children

    @staticmethod
    def get_label(node):
        return node.my_label

    def addkid(self, node, before=False):
        if before:  self.my_children.insert(0, node)
        else:   self.my_children.append(node)
        return self


def xml2zss(fh):

    parser = etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding='utf-8')
    tree = etree.parse(fh, parser=parser)
    root = tree.getroot()

    zssTree = (WeirdNode(root.tag))
    for subnode in root:
        traverseTree(zssTree, subnode)

    return zssTree

def traverseTree(zssTree, node):
    
    for subnode in node:
        NodeTuple = (subnode.tag, None)
        if subnode.get('rel2par'):
            NodeTuple = (subnode.tag, subnode.get('rel2par'))
        zssTree.addkid(WeirdNode(NodeTuple))
        traverseTree(zssTree, subnode)

        
def setTaxXML(fh):
    global TAXONOMYXML
    TAXONOMYXML = fh

def setDefaultDistance(v):
    global DEFAULT_DISTANCE
    DEFAULT_DISTANCE = v

def setKnownNodesList(l):
    global KNOWNNODES
    KNOWNNODES = l
        
if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-a", "--aTree", dest="aTreeXML", help="Specify xml file with a tree.")
    parser.add_option("-b", "--bTree", dest="bTreeXML", help="Specify xml file with another tree for comparison.")
    parser.add_option("-x", "--taxonomyXML", dest="taxonomyXML", help="XML file with taxonomy of rhetorical relations.")
   
    options, args = parser.parse_args()
    
    if not options.aTreeXML or not options.bTreeXML or not options.taxonomyXML:
        parser.print_help(sys.stderr)
        sys.exit(1)

    A = xml2zss(options.aTreeXML)
    B = xml2zss(options.bTreeXML)
    
    setTaxXML(options.taxonomyXML)
    setKnownNodesList(treedist.getAllNodes(TAXONOMYXML))
    setDefaultDistance(treedist.getAverageDistance(TAXONOMYXML, KNOWNNODES)) # this is the average distance between two nodes given the tree

    print("debug a:", A)
    print("debug b:", B)
    
    dist = zss.simple_distance(A, B, WeirdNode.get_children, WeirdNode.get_label, weird_dist)

    print("dist:", dist)
    #TODO: debug distance measure!
