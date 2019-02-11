#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import utils



def data2tokens(data):

    tokens = []
    if os.path.isdir(data):
        for f in utils.getInputfiles(data):
            for line in codecs.open(f).readlines():
                for token in line.split():
                    tokens.append(token)
    elif os.path.isfile(data):
        for line in codecs.open(data).readlines():
            for token in line.split():
                tokens.append(token)
    return set(tokens)


def filterEmbd(embd, tokens):

    d = {}
    for e in embd:
        if e in tokens:
            d[e] = embd[e]
    return d


if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-v", "--embeddings", dest="embeddings", help="specify embeddings file (ending in .vec, or at least white-space separated, where first column is the wor)...")
    parser.add_option("-d", "--data", dest="data", help="specify data folder, with pre-tokenized input (white space splitting returns tokens)...")

    options, args = parser.parse_args()
    
    if not options.embeddings or not options.data:
        parser.print_help(sys.stderr)
        sys.exit(1)

    embd = utils.loadExternalEmbeddings(options.embeddings)
    tokens = data2tokens(options.data)
        
    filtered = filterEmbd(embd, tokens)
    
    outf = codecs.open(os.path.splitext(options.embeddings)[0] + '.filtered.vec', 'w')

    dimensions = codecs.open(options.embeddings, 'r').readline()
    outf.write(dimensions)
    for e in filtered:
        outf.write('%s %s\n' % (e, ' '.join([str(x) for x in filtered[e]])))
        
    outf.close()
