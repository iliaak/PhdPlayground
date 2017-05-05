#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs


def conll20022opennlp(fileLines, nerType):

    outputLines = []
    sent = []
    fileLength = len(fileLines)
    for index, line in enumerate(fileLines):
        line = line.strip()
        tokens = line.split('\t')
        if tokens[-1].startswith('B-' + nerType):
            foundType = re.sub(r'^B-', '', tokens[-1])
            sent.append('<START:%s>' % foundType)
            sent.append(tokens[0])
            if index < fileLength and fileLines[index+1].split('\t')[-1].startswith('I-' + nerType):
                pass
            else:
                sent.append('<END>')
        elif tokens[-1].startswith('I-' + nerType):
            sent.append(tokens[0])
            if index < fileLength and fileLines[index+1].split('\t')[-1].startswith('I-' + nerType):
                pass
            else:
                sent.append('<END>')
        else:
            sent.append(tokens[0])

        if re.match(r'^\s*$', line):
            outputLines.append(' '.join(sent))
            sent = []

    return outputLines




if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-c", "--conll", dest="conll", help="specify file in conll format...")
    
    options, args = parser.parse_args()
    
    if not options.conll:
        parser.print_help(sys.stderr)
        sys.exit(1)


    outputlines = conll20022opennlp(codecs.open(options.conll, 'r').readlines(), "connective")
    if not options.conll.endswith(".conll"):
        sys.stderr.write("ERROR: please specify conll file ending in .conll, or change this line in script.")
    outf = codecs.open(os.path.splitext(options.conll)[0] + ".opennlp", 'w')
    for line in outputlines:
        if line:
            outf.write(line + "\n")
        
    outf.close()
