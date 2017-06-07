#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import spacy
import networkx as nx
import time
#import nltk
import urllib

"""
Script taking a list of connectives and iteratively looking for other-language version (first from German to English, then back to German, and so forth). Assumption is that the folders/files are aligned.

NOTE: 
- install spacy ('pip install spacy')
- download de and en models for spacy: 'python3 -m spacy download en' (and 'de' afterwards)
- install goslate ('pip install goslate')
"""

de_pipeline = None
en_pipeline = None

POSTAGMAP = {'CONJ':['CCONJ'], 'ADP':['ADP']} # the german pos tags (as assigned by spacy) and their english equivalents. TODO: complete this with more types/tags

"""
NOTES: 

DIFFICULTIES:
- one connective in german can correspond to a phrase in english (and the other way round)
- a german connective is not necessarily overtly realised in english (and the other way round)
- a connective/adverb in the seed list may have a non-connective reading in the particular sentence (leading to false hits on the english side)
- 
"""

class Connective:
    def __init__(self, word, lang): # may want to add more info here
        self.word = word
        self.lang = lang
        self.translations2 = {}
        self.translations = defaultdict(int)


    def addTranslation(self, translation):
        self.translations[str(translation)] += 1
        print('debug newdict:', self.translations)
        

def loadSpacyModels():
    
    global de_pipeline
    global en_pipeline
    #print('INFO: Loading spacy models...')
    t1 = time.time()
    de_pipeline = spacy.load('de')
    t2 = time.time()
    en_pipeline = spacy.load('en')
    t3 = time.time()
    #print('INFO: Done loading spacy models (%s seconds).' % str(int(t3 - t1)))
    


def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


def getList(fh):

    l = []
    for line in codecs.open(fh, 'r').readlines():
        if re.search('\w+', line):
            l.append(line.strip())
    return l


def dictAlign(deFiles, enFiles):

    deDict = defaultdict(list)
    enDict = defaultdict(list)

    for f in deFiles:
        if f.endswith('.txt'):
            fid = re.sub(r'de_', '', os.path.basename(f))
            deDict[fid] = codecs.open(f, 'r').readlines()
    for f in enFiles:
        if f.endswith('.txt'):
            fid = re.sub(r'en_', '', os.path.basename(f))
            enDict[fid] = codecs.open(f, 'r').readlines()
    
    #sanity check:
    for fid in deDict:
        if not len(deDict[fid]) == len(enDict[fid]):
            sys.stderr.write('ERROR: sentences do not match for file:%s.\nDying now.\n' % fid)

    return deDict, enDict

def getRootNode(dep_graph):

    rootNode = None
    for word in dep_graph:
        if word.dep_ == 'ROOT':
            rootNode = word.text + '-' + str(word.i)
    return rootNode
    
def getNXGraph(dep_graph):

    edges = []
    for word in dep_graph:
        for child in word.children:
            edges.append(('{0}-{1}'.format(word.lower_,word.i),
                          '{0}-{1}'.format(child.lower_,child.i)))
    return nx.Graph(edges)

def getRootDist(graph, connectorNode, rootNode):

    if graph.has_node(rootNode) and graph.has_node(connectorNode) and nx.has_path(graph, connectorNode, rootNode): # I think the input format (at least the CORBON data) is supposed to be tokenised (hence also sentence-splitted) already. This was not the case everywhere (so it crashes if the connector is in sentence B and it is looking for the rootNode of sentence A...)
        return nx.shortest_path_length(graph, source = connectorNode, target = rootNode)
    

def iterate(gc, ec, deDict, enDict):

    deMemoryMap = defaultdict(spacy.tokens.doc.Doc)
    enMemoryMap = defaultdict(spacy.tokens.doc.Doc)

    #TODO: debug if memory maps are actually being used (still quite slow on more data)
    #TODO: check pdtb sense and check for overlap
    #TODO: the code below only deals with single words (both on the german and english site). Explore discovery as phrases for single words and in reverse

    #DEBUG only! Remove!
    for entry in gc:
        if entry.word == 'und' or entry.word == 'aber':
            gc.remove(entry)
            
    
    for gConnItem in gc: #TODO: change loop structure, looping through the larger list (i.e. deDict) first, then though connective list (should save some time, even though I'm using the memoryMap)
        gConn = gConnItem.word
        if len(gConn.split()) == 1: # connective is a single word
            for fid in deDict:
                for i, sentence in enumerate(deDict[fid]):
                    if re.search(r'\b%s\b' % gConn, sentence, re.IGNORECASE):
                        print('DEBUGGING FOuND CONNECTIVE:', gConn)
                        doc = None
                        if sentence in deMemoryMap:
                            doc = deMemoryMap[sentence]
                        else:
                            doc = de_pipeline(sentence, parse=True)
                            deMemoryMap[sentence] = doc

                        
                        rootNode = getRootNode(doc)
                        graph = getNXGraph(doc)

                        
                        for word in doc:
                            if word.text.lower() == gConn.lower():
                                for word2 in doc:
                                    print(word2.text, word2.pos_, word2.dep_, word2.head.text)
                                    
                                print('connective found:', word.text)
                                #print('pos:', word.pos_)
                                connectorNode = word.text + '-' + str(word.i)
                                print('debug connectirNode:', connectorNode)
                                rootdist = getRootDist(graph, connectorNode, rootNode)
                                
                                # constituency trees not supported in spacy, instead: looking for the nounphrases the connectives appears in between, and checking the same for english...? (get some MT thing going for this)
                                
                                
                                candidates, enMemoryMap = checkEnglishEquivalent(enDict[fid][i], enMemoryMap, gConn, word.pos_, rootdist)
                                print('DEBG candidates:', candidates)
                                for c in candidates:
                                    print('Found\n"%s"\nas english option for\n"%s"\nSource sentence pair:\n"%s"\n+++++++\n"%s"\n' % (c, gConn, sentence.strip(), enDict[fid][i].strip()))
                                    gConnItem.addTranslation(c)
                                        
                        
        else:
            pass #TODO: write code for when connective is discontinuous or a phrase!

    for gConn in gc:
        if gConn.translations:
            print('RESULT: Translations found for %s: %s.' % (gConn.word, str(gConn.translations)))


def checkEnglishEquivalent(enSent, enMemoryMap, deConnectiveToken, dePOSTag, deRootdist):

    #print('Processing sent:', enSent)
    candidates = []
    
    doc = None
    if enSent in enMemoryMap:
        doc = enMemoryMap[enSent]
    else:
        doc = en_pipeline(enSent)
        enMemoryMap[enSent] = doc

    rootNode = getRootNode(doc)
    graph = getNXGraph(doc)
    
    for word in doc:
        print(word.text, word.pos_)#, word.dep_, word.head.text)
        enRootdist = getRootDist(graph, word.text + '-' + str(word.i), rootNode)
        if dePOSTag in POSTAGMAP and word.pos_ in POSTAGMAP[dePOSTag]:
            if deRootdist and enRootdist: # can also be None
                if abs(deRootdist - enRootdist) <  2 + graph.size() / 20: # may want to play with the threshold
                    #print('DEBUG found likely candidate!!!!!!!!!:', word.text)
                    candidates.append(word)
                    
    return candidates, enMemoryMap


    

def translateConnectivesLiterally(fh):

    # all translation APIs I've tried so far (goslate, pytranslate, some others, and the one below) are shut off/behind a paywall. Perhaps setup Moses connection myself?
    import pprint
    from apiclient.discovery import build
    api_key = 'AIzaSyD8IaW3gxk8wN0cgrzOszFecpOWXK4mD5Y'
    service = build('translate', 'v2', developerKey=api_key)

    print(service.translations().list(
        source='en',
        target='fr',
        q=['flower', 'car']
    ).execute())

    
    
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-d', '--de', dest='deFolder', help='German input folder. Assumption is that all files in German and English folder have the same name, except for the prefix (which is either de or en)')
    parser.add_option('-e', '--en', dest='enFolder', help='English input folder. See option deFolder')
    parser.add_option('-c', '--connectiveList', dest='connectiveList', help='specify file with a list of connectives that need to be looked up')
    
    options, args = parser.parse_args()

    if not options.deFolder or not options.enFolder or not options.connectiveList:
        parser.print_help(sys.stderr)
        sys.exit(1)

    deFiles = []
    enFiles = []
    if os.path.isdir(options.deFolder) and os.path.isdir(options.enFolder):
        deFiles = getInputfiles(options.deFolder)
        enFiles = getInputfiles(options.enFolder)
    else:
        sys.stderr.write('ERROR: Could not find input folder, please check path.\nDying now.\n')


    loadSpacyModels()
    
    deDict, enDict = dictAlign(deFiles, enFiles)

    connectives = getList(options.connectiveList)

    germanConnectives = []
    englishConnectives = []
    for c in connectives:
        conn = Connective(c, 'de')
        germanConnectives.append(conn)
        
    iterate(germanConnectives, englishConnectives, deDict, enDict)
