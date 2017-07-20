#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from nltk.parse import stanford
from nltk import sent_tokenize
from nltk import Tree
from nltk.tree import ParentedTree
import PCCParser
import csv

os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2017-06-09' # TODO: set this in some config file
os.environ['STANFORD_MODELS'] = 'stanford-parser-full-2017-06-09'# same here
lexParser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz") # TODO: set this in some config file
os.environ['JAVAHOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64' # same here!

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

def compressRoute(r): # filtering out adjacent identical tags

    delVal = "__DELETE__"
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            r[i+1] = delVal
    return [x for x in r if x != delVal]
    
            
def getPathToRoot(ptree, route):

    if ptree.parent() == None:
        route.append(ptree.label())
        return route
    else:
        route.append(ptree.label())
        getPathToRoot(ptree.parent(), route)
    return route
                     
            
def constructFeatureMatrix(connectorxmls):

    instanceIndex = 0
    instance2Values = defaultdict(list)
    
    for i, cxml in enumerate(connectorxmls):
        sys.stderr.write("INFO: Parsing file .../%s (%s/%s).\n" % (cxml.split('/')[-1], str(i+1), str(len(connectorxmls))))
        # extract real classes from PCC
        fileTokenList = PCCParser.parseConnectorFile(cxml)
        instance2Values = getFeaturesForFileTokens(fileTokenList, instance2Values, instanceIndex)
    
    for inst in instance2Values:
        print(inst, instance2Values[inst])

    return instance2Values

def getFeaturesForFileTokens(ftokens, instance2Values, instanceIndex):        

    # bookkeeping
    id2class = defaultdict(bool)
    for i, t in enumerate(ftokens):
        if t.isConnective:
            id2class[i + instanceIndex] = True
        else:
            id2class[i + instanceIndex] = False
    
    
    sentences = sent_tokenize(' '.join([t.token for t in ftokens]))
    tokenizedSents = [sent.split() for sent in sentences]
    try:
        forest = lexParser.parse_sents(tokenizedSents)
        for trees in forest:
            for tree in trees:
                instanceIndex, instance2Values = getFeaturesForTree(tree, instanceIndex, instance2Values)

        # add connective booleans (as last column)
        for instance in instance2Values:
            newl = instance2Values[instance]
            newl.append(id2class[instance])
            instance2Values[instance] = newl
    except ValueError:
        sys.stderr.write("WARNING: Failed to parse file. Skipping.\n")
        """
        # Got the following error for some file, coming from deep down the nltk parser somewhere...
        ValueError: Tree.read(): expected ')' but got 'end-of-string'
            at index 121.
                "...JD 2.)))))"
        """
        
    return instance2Values

        
def getFeaturesForTree(tree, instanceIndex, instance2Values):

    parentedTree = ParentedTree.convert(tree)
    for i, node in enumerate(parentedTree.pos()):
        featureList = []
        currWord = node[0]
        currPos = node[1]
        featureList.append(currWord)
        featureList.append(currPos)
        ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
        rn = "EOS" if i == len(parentedTree.pos())-1 else parentedTree.pos()[i+1]
        lpos = "_" if ln == "SOS" else ln[1]
        rpos = "_" if rn == "EOS" else rn[1]
        lstr = ln if ln == "SOS" else ln[0]
        rstr = rn if rn == "EOS" else rn[0]
        lbigram = lstr + '_' + currWord
        rbigram = currWord + '_' + rstr
        lposbigram = lpos + '_' + currPos
        rposbigram = currPos + '_' + rpos
        featureList.append(lbigram)
        featureList.append(lpos)
        featureList.append(lposbigram)
        featureList.append(rbigram)
        featureList.append(rpos)
        featureList.append(rposbigram)
        
        
        # TODO: self-category is not clear to me, if not always POS (because we are dealing with single words here in binary classification), what else should it be?
        nodePosition = parentedTree.leaf_treeposition(i)
        parent = parentedTree[nodePosition[:-1]].parent()
        parentCategory = parent.label()
        featureList.append(parentCategory)
        
        ls = parent.left_sibling()
        lsCat = False if not ls else ls.label()
        rs = parent.right_sibling()
        rsCat = False if not rs else rs.label()
        featureList.append(lsCat)
        featureList.append(rsCat)
        rsContainsVP = False
        if rs:
            if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                rsContainsVP = True
        # TODO: Figure out how to check if rs contains a trace (given the tree/grammar)
        featureList.append(rsContainsVP)
        #featureList.append(rsContainsTrace) # TODO
        rootRoute = getPathToRoot(parent, []) # to make things a bit more general, not including the pos tag of the node itself here, as it is in plenty of other features already
        featureList.append(rootRoute)
        cRoute = compressRoute([x for x in rootRoute])
        featureList.append(cRoute)
        
        instance2Values[instanceIndex] = featureList
        instanceIndex += 1
        
    
    return instanceIndex, instance2Values
    
                

if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectivesFolder', dest='connectivesFolder', help='Specify PCC connectives folder to construct a feature matrix for training a classifier.')
    
    
    options, args = parser.parse_args()

    if not options.connectivesFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # MAJOR TODO: Find out why it seems to fail on more data (looks good based on testFolder containing .../maz-00001.xml and .../maz-00002.xml
    fm = constructFeatureMatrix(getInputfiles(options.connectivesFolder))

    # TODO: DEBUG THIS WRITING (writing some lists somewhere where there should be flat strings I think
    with open('tempout.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=['id', 'token', 'pos', 'leftbigram', 'leftpos', 'leftposbigram', 'rightbigram', 'rightpos', 'rightposbigram', 'parentCategory', 'leftSiblingCategory', 'rightSiblingCategory', 'rightSiblingContainsVP', 'pathToRoot', 'compressedPathToRoot', 'isConnective'])
        #writer.writeheader()
        for instance in fm:
            writer.writerow({'id': instance, 'token': fm[0], 'pos': fm[1], 'leftbigram': fm[2], 'leftpos': fm[3], 'leftposbigram': fm[4], 'rightbigram': fm[5], 'rightpos': fm[6], 'rightposbigram': fm[7], 'parentCategory': fm[8], 'leftSiblingCategory': fm[9], 'rightSiblingCategory': fm[10], 'rightSiblingContainsVP': fm[11], 'pathToRoot': fm[12], 'compressedPathToRoot': fm[13], 'isConnective': fm[14]})
