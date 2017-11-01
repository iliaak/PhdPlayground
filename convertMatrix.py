#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import csv

if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-m", "--matrix", dest="matrix", help="specify matrix...")

    options, args = parser.parse_args()
    
    if not options.matrix:
        parser.print_help(sys.stderr)
        sys.exit(1)


    fh = codecs.open(options.matrix, 'r')
    reader = csv.reader(fh, delimiter=',')

    headers = []
    column2features = defaultdict(set)
    for i, row in enumerate(reader):
        if i == 0:
            headers = row
        else:
            for j, k in enumerate(row):
                column2features[j].add(k)

    column2str2float = defaultdict(lambda : defaultdict(float))
    # convert...
    for i in range(1, len(column2features)):
        uniqueValues = column2features[i]
        lastId = 0
        for uv in uniqueValues:
            # it is a set, so no need to check for uniqueness here...
            column2str2float[i][uv] = lastId
            lastId += 1

    newMatrix = []
    fh = codecs.open(options.matrix, 'r')
    reader = csv.reader(fh, delimiter=',')
    for i, row in enumerate(reader):
        if i == 0:
            newMatrix.append(row)
        else:
            newRow = []
            for i, j in enumerate(row):
                if i == 0:
                    newRow.append(int(j))
                else:
                    newRow.append(column2str2float[i][j])
            newMatrix.append(newRow)

    outf = codecs.open('floatMatrix.csv', 'w')
    writer = csv.writer(outf, delimiter=',')
    for row in newMatrix:
        writer.writerow(row)
    outf.close()
