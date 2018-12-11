import os
import sys
import re
import codecs
import lxml.etree
import string
from collections import defaultdict

class DiscourseToken():

    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token

    def setRstSegmentId(self, val):
        self.rstSegmentId = val
    def setRstParent(self, val):
        self.rstParent = val
    def setRelname(self, val):
        self.relname = val
    def setSegmentStart(self, i):
        if i == 0:
            self.segmentStart = True
        else:
            self.segmentStart = False
        

def parseRSTfile(rstxml):

    tokenlist = []
    rstTokenId = 0
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(rstxml, parser = xmlParser)
    body = tree.getroot().find('.//body')
    for node in body:
        if node.tag == 'segment':
            sId = node.get('id')
            sParent = node.get('parent')
            sRelname = node.get('relname')
            tokens = node.text.split()
            for i, token in enumerate(tokens):
                dt = DiscourseToken(rstTokenId, token)
                dt.setRstSegmentId(sId)
                dt.setRstParent(sParent)
                dt.setRelname(sRelname)
                dt.setSegmentStart(i)
                tokenlist.append(dt)
                rstTokenId += 1
    return tokenlist

def parseDepsfile(depconll, rsttokenlist, outputbasename):

    depTokenId = 0
    # making one iteration first to get index to line
    index2line = defaultdict(str)
    for line in codecs.open(depconll, 'r').readlines():
        if re.search('\t', line):
            index2line[depTokenId] = line
            depTokenId += 1

    lastSentId = 0
    out = codecs.open(outputbasename + '.segmented.conll', 'w')
    for k, v in index2line.items():
        elems = v.split('\t')
        token = elems[1]
        sentId = elems[0]
        rsttoken = rsttokenlist[k].token

        if not rsttoken == token:
            print('MISMATCH!')
            print('rst:', rsttoken)
            print('dep:', token)
            print('in file:', depconll)
            print('DYING NOW!')
            sys.exit(1)
        
        if int(sentId) <= int(lastSentId):
            out.write('\n')
        if rsttokenlist[k].segmentStart:
            v = v.strip() + '\tB-Segment\n'
        out.write(v)
        lastSentId = sentId
    
def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

def basename2fullpath(a, b):

    d = defaultdict(lambda : defaultdict(str))
    for f in a:
        d[os.path.splitext(os.path.basename(f))[0]]['deps'] = f
    for f2 in b:
        d[os.path.splitext(os.path.basename(f2))[0]]['rst'] = f2
    return d
        
if __name__ == '__main__':

    #dev run
    #devdeps = getInputfiles('dev/deps')
    devdeps = getInputfiles('dev/udeps')
    devrst = getInputfiles('dev/rst')
    d = basename2fullpath(devdeps, devrst)
    for f in d:
        rsttokenlist = parseRSTfile(d[f]['rst'])
        parseDepsfile(d[f]['deps'], rsttokenlist, 'dev/out/'+f)
    
    #test run
    #testdeps = getInputfiles('test/deps')
    testdeps = getInputfiles('test/udeps')
    testrst = getInputfiles('test/rst')
    d = basename2fullpath(testdeps, testrst)
    for f in d:
        rsttokenlist = parseRSTfile(d[f]['rst'])
        parseDepsfile(d[f]['deps'], rsttokenlist, 'test/out/'+f)
    
    #train run
    #traindeps = getInputfiles('train/deps')
    traindeps = getInputfiles('train/udeps')
    trainrst = getInputfiles('train/rst')
    d = basename2fullpath(traindeps, trainrst)
    for f in d:
        rsttokenlist = parseRSTfile(d[f]['rst'])
        parseDepsfile(d[f]['deps'], rsttokenlist, 'train/out/'+f)
    
    
    #rsttokenlist = parseRSTfile('train/rst/maz-14654.rs3')
    #parseDepsfile('train/deps/maz-14654.conll', rsttokenlist)
