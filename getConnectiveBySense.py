#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import DimLexParser

"""
Script to provide a dimlex sense and get a list of all connectives with this sense
"""


if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-d', '--dimlex', dest='dimlex', help='specify dimlex xml')
    parser.add_option('-o', '--output', dest='output', help='specify output file')
    parser.add_option('-t', '--threshold', dest='threshold', help='threshold controlling the minimum probability of the particular reading for this connective', type=float)
    parser.add_option('-s', '--sense', dest='sense', help='(pdtb)sense to look for')
    
    options, args = parser.parse_args()

    if not options.dimlex or not options.output or not options.sense:
        parser.print_help(sys.stderr)
        sys.exit(1)

    outputlist = []
    connectiveList = DimLexParser.parseXML(options.dimlex)
    for conn in connectiveList:
        for sense in conn.sense2Probs:
            if sense == options.sense:
                if options.threshold:
                    if float(conn.sense2Probs[sense]) > options.threshold:
                        outputlist.append(conn)
                else:
                    outputlist.append(conn)
    outf = codecs.open(options.output, 'w')
    for conn in outputlist:
        outf.write(conn.word + '\n')
    outf.close()
