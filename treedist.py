#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from lxml import etree
from zss import Node, simple_distance


def debug(fh):

    xmlString = "<root><a><b/><c><d/></c></a></root>"
    root = etree.fromstring(xmlString)
    l = ['root', 'a', 'b', 'c', 'd']
    for i in l:
        for j in l:
            ancestorChain = traverse(root, i, j, [])
            lca = ancestorChain[len(ancestorChain)-1]
            dist = getDistance(lca, i, 0, False) + getDistance(lca, j, 0, False)
            print("Distance between %s and %s: %s" % (i, j, dist))


def getTreeDistance(taxonomyXML, n1, n2):

    parser = etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding='utf-8')
    tree = etree.parse(taxonomyXML, parser=parser)
    root = tree.getroot()
    ancestorChain = traverse(root, n1, n2, [])
    lca = ancestorChain[len(ancestorChain)-1]
    dist = getDistance(lca, n1, 0, False) + getDistance(lca, n2, 0, False)

    return dist
    
            

def nodeInChildren(node, target, f):

    if node.tag == target:
        f = True
        return f
    else:
        for subnode in node:
            f = nodeInChildren(subnode, target, f)
    return f
    
def traverse(node, n1, n2, ancestorChain):

    n1IsChild = nodeInChildren(node, n1, False)
    n2IsChild = nodeInChildren(node, n2, False)
    if n1IsChild and n2IsChild:
        ancestorChain.append(node)
        for subnode in node:
            traverse(subnode, n1, n2, ancestorChain)

    return ancestorChain


def getDistance(node, target, dist, f):

    if node.tag == target:
        return dist
    dist += 1
    childList = []
    for subnode in node:
        childList.append(subnode.tag)
    if target in childList:
        f = True
        return dist
    else:
        for subnode in node:
            f = getDistance(subnode, target, dist, f)
    return f


#debug(sys.argv[1])
#print(getTreeDistance(sys.argv[1], 'b', 'd'))


          
