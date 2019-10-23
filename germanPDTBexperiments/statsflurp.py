#!/usr/bin/env python3
# this one differs from other CONLLParsers in the sense that it not just picks out connectives (if re.search("conn", ' '.join(row[5:])):, but also gets the corresponding arguments
# should be better in every way, so can use this one over CONLLParser.py everywhere, probably

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import csv
from tqdm import tqdm


class Relation:

    def __init__(self, connectiveTokens, arg1Tokens, arg2Tokens, sense, rid, _type):
        self.connectiveTokens = connectiveTokens
        self.arg1Tokens = arg1Tokens
        self.arg2Tokens = arg2Tokens
        self.sense = sense
        self.relationId = rid
        self._type = _type
    

class Token:

    def __init__(self, tokenId, sentenceId, sentencePosition, token):
        self.token = token
        self.relationIds = set()
        self.tokenId = tokenId
        self.sentenceId = sentenceId
        self.sentencePosition = sentencePosition
        self._types = defaultdict(str)
        self.senses = defaultdict(str)

    def addSentence(self, sentence):
        self.fullSentence = sentence
    def addAnnotation(self, cid, _type, sense=False):
        self.relationIds.add(cid)
        self._types[cid] = _type
        if sense:
            self.senses[cid] = sense
    def setIndices(self, i, j):
        self.characterStartIndex = i
        self.characterEndIndex = j

            
def parsePDTBFile(conllFile):

    file2conllTokens = defaultdict(list)
    fname = None
    conllTokens = []
    tokenoffset = 0
    f2sid2tokens = defaultdict(lambda : defaultdict(list))
    f2tid2token = defaultdict(lambda : defaultdict(Token))
    with codecs.open(conllFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if row:
                if row[0].startswith('#'):
                    if conllTokens:
                        file2conllTokens[fname] = conllTokens
                    fname = row[0][1:]
                    conllTokens = []
                    tokenoffset = 0
                else:
                    tokenId, sentenceId, sentencePosition, token = row[:4]
                    t = Token(tokenId, sentenceId, sentencePosition, token)
                    t.setIndices(tokenoffset, tokenoffset + len(token))
                    tokenoffset += len(token)+1
                    f2sid2tokens[fname][sentenceId].append(tokenId)
                    for i, col in enumerate(row[5:]):
                        if col.startswith('arg1'):
                            t.addAnnotation(i, 'arg1')
                        elif col.startswith('arg2'):
                            sense = None
                            if re.search('^arg2\|', col):
                                sense = re.match('^arg2\|(.*)$', col).groups()[0]
                            #t.addAnnotation(i, 'arg2', sense)
                            t.addAnnotation(i, col, sense) # instead of arg2, passing on full specs (|connective|sense may follow for implicits, for ex)
                        elif col.startswith('conn'):
                            sense = re.match('^conn\|(.*)$', col).groups()[0]
                            t.addAnnotation(i, 'conn', sense)

                    conllTokens.append(t)
                    f2tid2token[fname][tokenId] = t
        file2conllTokens[fname] = conllTokens

    for f in f2sid2tokens:
        sid2fullsent = defaultdict(str)
        for sid in f2sid2tokens[f]:
            fullSentence = ' '.join([f2tid2token[f][tid].token for tid in f2sid2tokens[f][sid]])
            sid2fullsent[sid] = fullSentence
        for ct in file2conllTokens[f]:
            ct.addSentence(sid2fullsent[ct.sentenceId])

    
    file2relations = defaultdict(lambda : defaultdict(Relation))
    for f in file2conllTokens:
        rid = 1
        for ct in file2conllTokens[f]:
            for cid in ct._types:
                if ct._types[cid] == 'conn':
                    conntokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct._types[cid] == ct2._types[cid]]
                    arg1Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid] == 'arg1']
                    arg2Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid] == 'arg2']
                    sense = ct.senses[cid]
                    erel = Relation(conntokens, arg1Tokens, arg2Tokens, sense, '%s_%s' % (f, rid), 'Explicit')
                    file2relations[f]['%s_%s' % (f, rid)] = erel # this overwrites it for multi-token connectives. Not very elegant/optimally efficient, but who cares (I don't)
                    rid += 1

                elif ct._types[cid].lower().startswith('arg2|altlex'):
                    conntokens = []
                    arg1Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid] == 'arg1']
                    arg2Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid].startswith('arg2')]
                    sense = ct.senses[cid]
                    altlex = Relation(conntokens, arg1Tokens, arg2Tokens, sense, '%s_%s' % (f, rid), 'AltLex')
                    file2relations[f]['%s_%s' % (f, rid)] = altlex
                    rid += 1

                elif ct._types[cid].lower().startswith('arg2|entrel'):
                    conntokens = []
                    arg1Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid] == 'arg1']
                    arg2Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid].startswith('arg2')]
                    sense = ct.senses[cid]
                    entrel = Relation(conntokens, arg1Tokens, arg2Tokens, sense, '%s_%s' % (f, rid), 'EntRel')
                    file2relations[f]['%s_%s' % (f, rid)] = entrel
                    rid += 1

                elif ct._types[cid].lower().startswith('arg2|'):
                    conntokens = []
                    arg1Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid] == 'arg1']
                    arg2Tokens = [ct2 for ct2 in file2conllTokens[f] if ct2.relationIds.intersection(ct.relationIds) and ct2._types[cid].startswith('arg2')]
                    sense = ct.senses[cid]
                    irel = Relation(conntokens, arg1Tokens, arg2Tokens, sense, '%s_%s' % (f, rid), 'Implicit')
                    file2relations[f]['%s_%s' % (f, rid)] = irel
                    rid += 1
            
                
    return file2conllTokens, file2relations

            
if __name__ == '__main__':
    
    f2conlltokens, f2relations = parsePDTBFile('/home/peter/various/german_pdtb/corpus/german_version/german_pdtb_full_converted.conll')
    #f2conlltokens, f2relations = parsePDTBFile('dummy.conll')
    

    for f in f2relations:
        print(f)
        for rel in f2relations[f]:
            print(f2relations[f][rel].relationId)
            print(f2relations[f][rel]._type)
            print([x.token for x in f2relations[f][rel].connectiveTokens])
            print([x.token for x in f2relations[f][rel].arg1Tokens])
            print([x.token for x in f2relations[f][rel].arg2Tokens])
            print(f2relations[f][rel].sense)
            print('\n\n\n')
    
