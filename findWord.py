#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import numpy
import itertools


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
                end[i2+1] = word # alignment is not zero-based
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

def findPhrase(phrase, corpusFile, threshold):

    out = defaultdict(float)
    words = phrase.split()
    flines = codecs.open(corpusFile, "r").readlines()
    t = 0
    for i, line in enumerate(flines):
        line = line.strip()
        if line.startswith("#"):
            enSent = flines[i+1]
            deSent = flines[i+2] # this and the line above should not lead to OutOfIndexErrors if the syntax is correct
            if all(word.lower() in deSent.lower().split() for word in words):
                end = defaultdict(str)
                for i2, word in enumerate(enSent.split()):
                    end[i2+1] = word # alignment is not zero-based
                phraseTranslation = defaultdict(float)
                for m in re.finditer(r"(\w+) \(\{([\d\s]+)\}\)", deSent):
                    deWord = m.groups()[0]
                    if deWord in phrase:
                        enIndex = m.groups()[1].split()
                        enWord = " ".join([end[int(i3)] for i3 in enIndex])
                        phraseTranslation[enWord] = enIndex
                pt = tuple([k for k in sorted(phraseTranslation, key=phraseTranslation.get)])
                out[pt] += 1
                t += 1

    # note that normalisation is not straightforward here, normalising only over current items in dict now
    nd = defaultdict(float)
    for o in out:
        v = out[o] / t
        if v > threshold: # not sure if using the same threshold as above here is a good idea
            nd[o] = v
    out = nd
    
    return out
                    

def findSingleWord(word, src2trg, threshold):

    if threshold == None:
        threshold = 0.5

    out = defaultdict(float)
    for trg in src2trg[word]:
        if src2trg[word][trg] > threshold:
            out[tuple(trg)] = src2trg[word][trg]
    
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

    # NOTE: for sinle words, it's faster/more efficient to first build the alignment dict (on word level basis) and then do simple lookups. For phrases, the original alignment file is needed to check for each individual case if/how every word in the phrase is translated (in the particlar context).
    
    if inputLength == 1:
        outdict = findSingleWord(options.word, src2trg, 0.2)
    else:
        outdict = findPhrase(options.word, options.corpus, 0.2)

    for key in sorted(outdict, key=outdict.get, reverse=True):
        print("%s\t%s" % (key, str(outdict[key])))
