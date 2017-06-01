#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize

"""
Script taking a list of connectives and iteratively looking for other-language version (first from German to English, then back to German, and so forth)
"""


EN_TAGGER = StanfordPOSTagger('english-bidirectional-distsim.tagger')
DE_TAGGER = StanfordPOSTagger('german-fast.tagger')



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

def iterate(gc, ec, deDict, enDict):

    tagMemoryMap = defaultdict(str)
    
    for gConn in gc: #note: programmatically more efficient to loop through the larger list (i.e. deDict) first, but this is intuitively better to grasp (for me, at least :)
        if len(gConn.split()) == 1:
            for fid in deDict:
                for i, s in enumerate(deDict[fid]):
                    if re.search(r'\b%s\b' % gConn, s, re.IGNORECASE):
                        deTags = ''
                        if len(tagMemoryMap[s]) > 1:
                            deTags = tagMemoryMap[s]
                        else:
                            deTags = DE_TAGGER.tag(word_tokenize(s))
                            tagMemoryMap[s] = deTags
                        de_tag = ''
                        for wt_pair in deTags:
                            if wt_pair[0].lower() == gConn:
                                de_tag = wt_pair[1]
                        print('DEBUG conn:', gConn)
                        print('DEBUG tag:', de_tag)
                        print('DEBUG de sent:', s)
                        checkEnglishEquivalent(enDict[fid][i])

        else:
            pass #TODO: write code for when connective is discontinuous


def checkEnglishEquivalent(enSent):

    print('Processing sent:', enSent)
    #TODO: parse, pos-tag, whatever, to get a list of candidates for connectives here... (get pos-tag of german word, position, maybe some other measures to decide upon the most likely english counterpart)


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

    deDict, enDict = dictAlign(deFiles, enFiles)

    connectives = getList(options.connectiveList)

    germanConnectives = defaultdict(int)
    englishConnectives = defaultdict(int)
    for c in connectives:
        germanConnectives[c] = 0 # value is the number of iterations that this connective was found in

    iterate(germanConnectives, englishConnectives, deDict, enDict)
