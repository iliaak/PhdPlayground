#!/usr/bin/env python3


import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import numpy
import subprocess

class DistanceMatrix(object):

    def __init__(self, size):
        self.size = size
        self.matrix = numpy.zeros((size, size))

    def setValue(self, x, y, val):
        self.matrix[x][y] = val

    def printMatrix(self):
        print(numpy.matrix(self.matrix))


def computeDistance(fi, fj, jarLocation):

    distance = 0.0
    output = subprocess.check_output(['java', '-jar', jarLocation, '-t', codecs.open(fi, 'r').read(), codecs.open(fj, 'r').read()])
    try:
        distance = float(output.strip())
    except:
        sys.stderr.write("ERROR: Value for %s and %s is not a double. Please investigate. Dying now.\n")
        sys.exit(1)
    
    return distance


if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-f", "--inputfolder", dest="inputfolder", help="Specify folder with APTED form trees.")
    parser.add_option("-j", "--jarLocation", dest="jarLocation", help="Location of APTER jar file (see: http://tree-edit-distance.dbresearch.uni-salzburg.at/#download).")
   
    options, args = parser.parse_args()
    
    if not options.inputfolder or not options.jarLocation:
        parser.print_help(sys.stderr)
        sys.exit(1)


    inputfiles = []
    for f in os.listdir(options.inputfolder):
        inputfiles.append(os.path.abspath(os.path.join(os.path.join(os.getcwd(), options.inputfolder), f)))

    DM = DistanceMatrix(len(inputfiles))

    for i in range(len(inputfiles)):
        for j in range(len(inputfiles)):
            if i == j:
                DM.setValue(i, j, 0.0)
            else:
                # do subprocess call to get distance and set it
                dist = computeDistance(inputfiles[i], inputfiles[j], options.jarLocation)
                DM.setValue(i, j, dist)

    # TODO: now print the matrix, or save in some way

    # TODO: also print the current order of the inputfiles, as perhaps they change or something. In any case it would be good to know what the x and y vars are...
    
