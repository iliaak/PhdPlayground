import os
import sys
from collections import defaultdict
import re
import numpy
import codecs
import time


def loadExternalEmbeddings(embfile):

    starttime = time.time()
    sys.stderr.write('INFO: Loading external embeddings from %s.\n' % embfile)
    ed = defaultdict()
    #"""
    with codecs.open(embfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            values = line.split()
            ed[values[0]] = numpy.array([float(x) for x in values[1:]])
    #"""
    endtime = time.time()
    sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))
    return ed


def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist
