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
   

def weird_dist(A, B):

    # this is where I can do the taxonomy-based distance, something like:
    if A == B:
        return 0
    else:
        # do the magic here. Proposal: both A and B are tuples here. Get A[1] and B[1], get taxonomy distance from treedist script (also in this dir), and return that value (TODO: create taxonomy XML from Hovy paper and take it from there). For this to work, make sure that the relations in the taxonomy and those in the parsed text match (this is not the case for the CODRA stuff and the Hovy taxonomy)
        return 99
    
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

        

if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-a", "--aTree", dest="aTreeXML", help="Specify xml file with a tree.")
    parser.add_option("-b", "--bTree", dest="bTreeXML", help="Specify xml file with another tree for comparison.")
   
    options, args = parser.parse_args()
    
    if not options.aTreeXML or not options.bTreeXML:
        parser.print_help(sys.stderr)
        sys.exit(1)

    A = xml2zss(options.aTreeXML)
    B = xml2zss(options.bTreeXML)

    
    
    dist = zss.simple_distance(A, B, WeirdNode.get_children, WeirdNode.get_label, weird_dist)

    print("dist:", dist)
