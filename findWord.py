#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import numpy


def parseCorpusToDict(corpusFile):

    src2trg = defaultdict(lambda : defaultdict(float))
    flines = codecs.open(corpusFile, "r").readlines()
    for i, line in enumerate(flines):
        line = line.strip()
        if line.startswith("#"):
            enSent = flines[i+1]
            deSent = flines[i+2] # this and the line above should not lead to OutOfIndexErrors if the syntax is correct
            end = defaultdict(str)
            for i2, word in enumerate(enSent.split()):
                end[i2+1] = word # alignemt is not zero-based
            for m in re.finditer(r"(\w+) \(\{([\d\s]+)\}\)", deSent): # maybe I'll also want to do something with the NULL at the start of the line (for words not literally found in the translation I guess)
                deWord = m.groups()[0]
                enIndex = m.groups()[1].split()
                enWord = " ".join([end[int(i3)] for i3 in enIndex])
                src2trg[deWord][enWord] += 1
            
    
    # normalize dict to probabilities
    for w in src2trg:
        total = 0
        for sw in src2trg[w]:
            total += src2trg[w][sw]
        for sw in src2trg[w]:
            src2trg[w][sw] = src2trg[w][sw] / total

    return src2trg


def findSingleWord(word, src2trg, threshold):

    if threshold == None:
        threshold = 0.5

    out = defaultdict(float)
    for trg in src2trg[word]:
        if src2trg[word][trg] > threshold:
            out[trg] = src2trg[word][trg]
    
    return out



            
if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-w", "--word", dest="word", help="specify source word to look for in alignment corpus (de-en)...")
    parser.add_option("-c", "--corpus", dest="corpus", help="aligned corpus, de-en")

    options, args = parser.parse_args()
    
    if not options.word or not options.corpus:
        parser.print_help(sys.stderr)
        sys.exit(1)

    src2trg = parseCorpusToDict(options.corpus)
    outdict = None
    inputLength = len(options.word.split())
    if inputLength == 1:
        outdict = findSingleWord(options.word, src2trg, 0.1)
    else:
        sys.stderr.write("ERROR: multiword units not yet supported. Dying now.\n")
        # TODO: think about this some more. Not straightforward: do the words have to be consecutive in the input? If so, should make another pass over the aligned file to get this right. Discontinuous markers are a different case still...

    for key in sorted(outdict, key=outdict.get, reverse=True):
        print("%s\t%s" % (key, str(outdict[key])))
