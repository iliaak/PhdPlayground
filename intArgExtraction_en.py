import os
import sys
import re
from collections import defaultdict
import codecs
import json
import CONLLParser
from nltk.tree import ParentedTree
from nltk.parse import stanford
import spacy
import string
import pickle

def checkAlignment(conllfiles, relations):

    for f in conllfiles:
        docId = os.path.splitext(os.path.basename(f))[0]
        tokens = CONLLParser.parsePDTBFile(f)
        for rid in relations[docId]:
            conntokens = relations[docId][rid]['conn']
            connrawtext = relations[docId][rid]['conntext']
            for ct in tokens:
                if int(ct.globalTokenId) in conntokens:
                    print('Found:', ct.token)
                    print('as part of raw text:', connrawtext)
                    print('with rel id:', rid)
                    print('in docId:', docId)
                    print('with intarg:', relations[docId][rid]['intargtext'])
                    print()

                    
def parseRelationsJson(reljs):

    relations = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for line in codecs.open(reljs, 'r').readlines():
        js = json.loads(line)
        fileId = js['DocID']
        rid = js['ID']
        # check this, but my current understanding of the TokenList is that [a, b, c, d, e] is that a is char start, b is char end, c is global (file) tokenid, d is sentenceid and e is sentence tokenid
        if js['Type'] == 'Explicit': # only taking explicits here
            conn = js['Connective']
            conntokens = [x[2] for x in conn['TokenList']]
            intarg = js['Arg2']
            intargtokens = [x[2] for x in intarg['TokenList']]
            extarg = js['Arg1']
            extargtokens = [x[2] for x in extarg['TokenList']]
            relations[fileId][rid]['conn'] = conntokens
            relations[fileId][rid]['intarg'] = intargtokens
            relations[fileId][rid]['extarg'] = extargtokens
            relations[fileId][rid]['conntext'] = conn['RawText'] # only for debugging purposes
            relations[fileId][rid]['intargtext'] = intarg['RawText'] # only for debugging purposes
            relations[fileId][rid]['extargtext'] = extarg['RawText'] # only for debugging purposes
            

    return relations
        

def flipdict(tokenlist): # not actually a dict as input.  # and also adding ct.fullsent in this function

    d = defaultdict()
    sid2tokens = defaultdict(list)
    for ct in tokenlist:
        sid2tokens[ct.sentenceId].append(ct.globalTokenId)
    
    for ct in tokenlist:
        d[int(ct.globalTokenId)] = ct

    return d, sid2tokens
        
def convertIds(sentIds, refconsentid, conlltokens):

    ids = []
    for ct in conlltokens:
        if ct.sentenceId == refconsentid and ct.sentenceTokenId in sentIds:
            ids.append(ct.globalTokenId)
    return sorted(ids)

def get_right_sibling(tree, pos, ct):

    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodepos = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            rs = pt[nodepos[:-1]].right_sibling()
            if rs:
                if rs.label() == 'S': # the conn is connecting one or two S-es, take the right sibling S as int arg
                    return rs.leaves()
                else:
                    parent = pt[nodepos[:-1]].parent()
                    # assuming that there are no duplicates of the connective anymore at this level of detail:
                    leaves = parent.leaves()
                    connindex = leaves.index(ct.token)
                    remainder = [xj for xi, xj in enumerate(leaves) if xi >= connindex]
                    return remainder
            else: # it's on the same level with its arg, which is not an S-clause
                parent = pt[nodepos[:-1]].parent()
                right_sibling = parent.right_sibling()
                leaves = parent.leaves()
                leaves = leaves + right_sibling.leaves() # in this case, it may well be at the end of the clause, in which case the right sibling should probably also be included
                connindex = leaves.index(ct.token)
                remainder = [xj for xi, xj in enumerate(leaves) if xi >= connindex]
                return remainder

def get_parent_phrase_plus_phrase_after_comma(tree, pos, labels, ct):

    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodePosition = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            children = pt[nodePosition[:1]]
            labelnode = climb_tree(tree, nodePosition, labels)
            predictedIntArgTokens = labelnode.leaves()
            rs = labelnode.right_sibling()
            if rs:
                if rs.label() == '$,':
                    predictedIntArgTokens += rs.right_sibling().leaves()
            return predictedIntArgTokens

        
def get_parent_phrase(tree, pos, labels, ct):

    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodePosition = tree.leaf_treeposition(i)
            pt = ParentedTree.convert(tree)
            children = pt[nodePosition[:1]]
            labelnode = climb_tree(tree, nodePosition, labels)
            predictedIntArgTokens = labelnode.leaves()
            return predictedIntArgTokens
            
def climb_tree(tree, nodePosition, labels):
    
    pTree = ParentedTree.convert(tree)
    parent = pTree[nodePosition[:-1]].parent()
    if parent.label() in labels or parent.label() == 'ROOT': # second condition in case the label I'm looking for is not there
        return parent
        
    else:
        return climb_tree(tree, nodePosition[:-1], labels)


def getSentenceId2Token(fullsent, ct):

    id2token = {}
    sp = ct.sentenceTokenId
    tokens = fullsent.split()
    for i, j in enumerate(tokens):
        diff = sp - i
        tokenId = int(ct.globalTokenId) - diff
        id2token[tokenId] = j
        
    return id2token

def matchPlainTokensWithIds(plain_tokens, id2Token):

    _sorted = sorted(id2Token.items(), key = lambda x: x[0])
    for i, pair in enumerate(_sorted):
        if plain_tokens[0] == pair[1]:
            if len(plain_tokens) > 1:
                if plain_tokens[1] == _sorted[i+1][1] or plain_tokens[1] == _sorted[i+2][1] and _sorted[i+1][1] in '()': # second condition to accomodate stupid bracket deletion requirement
                    anchor = _sorted[0][0]
                    if len(plain_tokens) == 2:
                        anchor = _sorted[i][0]
                    elif plain_tokens[2] == _sorted[i+2][1] or plain_tokens[2] == _sorted[i+3][1] and plain_tokens[1] in '()':
                        anchor = _sorted[i][0]
                    else:
                        pass#print('Taking default anchor!')
                    #sys.exit(1)
                    ret = []
                    for i in range(anchor, anchor + len(plain_tokens)):
                        ret.append(i)
                    return ret
            else: # in case of VP args it can sometimes be only 1 or 2 tokens I guess
                anchor = _sorted[0][0]
                ret = []
                for i in range(anchor, anchor + len(plain_tokens)):
                    ret.append(i)
                return ret        


def const_approach(conllfiles, relations):

    JAVAHOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64'
    STANFORD_PARSER='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
    STANFORD_MODELS='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
    os.environ['JAVAHOME'] = JAVAHOME
    os.environ['STANFORD_PARSER'] = STANFORD_PARSER
    os.environ['STANFORD_MODELS'] = STANFORD_MODELS
    import nltk
    nltk.internals.config_java(options='-xmx4G')
    
    lexParserPath = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
    lexParser = stanford.StanfordParser(model_path=lexParserPath)

    parsermemorymap = {}
    pmname = 'englishPDTBparsermemory.pickle'
    if os.path.exists(pmname):
        parsermemorymap = pickle.load(codecs.open(pmname, 'rb'))
        sys.stderr.write('INFO: Loaded parse trees from pickled dict.\n')


    tp = 0
    fp = 0
    fn = 0

    for _if, fname in enumerate(conllfiles):
        #if _if % 100 == 0:
        sys.stderr.write('Processing file %s (%i of %i).\n' % (fname, _if+1, len(conllfiles)))
        docId = os.path.splitext(os.path.basename(fname))[0]
        discourserelations = relations[docId]
        conlltokens = CONLLParser.parsePDTBFile(fname)
        gtid2ct, sid2tokens = flipdict(conlltokens)
        for rid in discourserelations:
            conntokens = discourserelations[rid]['conn']
            actualIntArgTokens = discourserelations[rid]['intarg']
            connsentids = set()
            for gtid in conntokens:
                connsentids.add(gtid2ct[gtid].sentenceId)
            if len(connsentids) > 1:
                print('skipping due to conn spread out over multiple sentences!')
            else:
                refcon = conntokens[-1]
                ctrefcon = gtid2ct[refcon]
                fullsent = ' '.join([gtid2ct[x].token for x in sid2tokens[gtid2ct[refcon].sentenceId]])
                #print('refcon:', gtid2ct[refcon].token)
                #print('fullsent:', fullsent)
                tree = None
                if fullsent in parsermemorymap:
                    tree = parsermemorymap[fullsent]
                else:
                    tree = lexParser.parse(fullsent.split())
                    parentedTreeIterator = ParentedTree.convert(tree)
                    for t in parentedTreeIterator:
                        tree = t
                        break # always taking the first, assuming that this is the best scoring tree...
                    parsermemorymap[fullsent] = tree


                """
                    
                # now locate conn in tree and check for some parent nodes (S, relS, VP, etc.) which gives the best results
                leaveno = 0
                if tree.leaves()[ctrefcon.sentenceTokenId] == ctrefcon.token:
                    leaveno = ctrefcon.sentenceTokenId
                else:
                    # probably due to round brackets that I removed because the nltk parser crashes on them
                    bracketno = 0
                    restored = False
                    if re.search('[\(\)]', fullsent):
                        for char in ' '.join(fullsent.split()[:ctrefcon.sentenceTokenId]):
                            if char == '(' or char == ')':
                                bracketno += 1
                    if bracketno:
                        if tree.leaves()[ctrefcon.sentenceTokenId-bracketno] == ctrefcon.token:
                            restored = True
                            leaveno = ctrefcon.sentenceTokenId-bracketno

                    # this is not supposed to happen
                    if not restored:
                        print('positions do not match')
                        print('tree leaves:', tree.leaves())
                        print('conntoken:', ctrefcon.token)
                        print('pos:', ctrefcon.sentenceTokenId)
                        print('fullsent:', fullsent)
                        print('DYING NOW!')
                        sys.exit()


                # now have the tree position, extract first S/VP/experiment going up from that node (ultimately taking the ROOT node if the specified label is not encountered in the way up in the tree)
                
                refconPosition = tree.leaf_treeposition(leaveno)
                pt = ParentedTree.convert(tree)
                connnode = pt[refconPosition[:-1]]
                refcontype = connnode.label()

                plain_tokens = []
                if refcontype == 'CC':
                    plain_tokens = get_right_sibling(tree, leaveno, ctrefcon)
                #elif refcontype == 'PROAV' or refcontype.startswith('A') or refcontype.startswith('K'):
                    #plain_tokens = get_parent_phrase_plus_phrase_after_comma(tree, leaveno, ['S', 'CS', 'VP'], ctrefcon)
                else:
                    plain_tokens = get_parent_phrase(tree, leaveno, ['S', 'CS', 'VP'], ctrefcon)


                # now convert the list of plain tokens (leaves) to token ids, since actualIntArgTokens are ids, and it's safer than plain tokens (could be duplicates)
                id2Token = getSentenceId2Token(fullsent, ctrefcon)
                predictedIntArgTokens = matchPlainTokensWithIds(plain_tokens, id2Token)

                predictedIntArgTokens = [x for x in predictedIntArgTokens if not x in conntokens]

                # stupid fix; re-inserting the comma after the last word if there is one (tree-based approach excludes this, PCC includes it...)
                if predictedIntArgTokens[-1]+1 in gtid2ct: # will otherwise crash if intArg is at end of file
                    if gtid2ct[predictedIntArgTokens[-1]+1].token in string.punctuation:#[',', '.']:
                        predictedIntArgTokens.append(predictedIntArgTokens[-1]+1)


                for tokenId in set(actualIntArgTokens + predictedIntArgTokens):
                    if tokenId in actualIntArgTokens and tokenId in predictedIntArgTokens:
                        tp += 1
                    elif tokenId in actualIntArgTokens:
                        fn += 1
                    else:
                        fp += 1
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
                """
    with open(pmname, 'wb') as handle:
        pickle.dump(parsermemorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write('INFO: Pickled parse trees to %s.\n' % pmname)

    return 1,2,3#precision, recall, f1


    

def dep_approach(conllfiles, relations):

    tp = 0
    fp = 0
    fn = 0

    sys.stderr.write('INFO: Loading English spacy stuff...\n')
    nlp = spacy.load('en')
    sys.stderr.write('INFO: Done.\n')

    for _if, fname in enumerate(conllfiles):
        if _if % 100 == 0:
            sys.stderr.write('Processing file %s (%i of %i).\n' % (fname, _if+1, len(conllfiles)))
        docId = os.path.splitext(os.path.basename(fname))[0]
        discourserelations = relations[docId]
        conlltokens = CONLLParser.parsePDTBFile(fname)
        gtid2ct, sid2tokens = flipdict(conlltokens)
        for rid in discourserelations:
            conntokens = discourserelations[rid]['conn']
            actualIntArgTokens = discourserelations[rid]['intarg']
            connsentids = set()
            for gtid in conntokens:
                connsentids.add(gtid2ct[gtid].sentenceId)
            if len(connsentids) > 1:
                print('skipping due to conn spread out over multiple sentences!')
            else:
                refcon = conntokens[-1]
                fullsent = ' '.join([gtid2ct[x].token for x in sid2tokens[gtid2ct[refcon].sentenceId]])
                #print('refcon:', gtid2ct[refcon].token)
                #print('fullsent:', fullsent)
                
                tree = nlp(fullsent)
                for index, token in enumerate(tree):
                    if index == gtid2ct[refcon].sentenceTokenId and gtid2ct[refcon].token == token.text: # not always the case, due to some tokenisation differences (spacy is not very configurable...)
                        predictedSentIdIntArgTokens = [x.i for x in token.head.subtree]
                        predictedIntArgTokens = convertIds(predictedSentIdIntArgTokens, gtid2ct[refcon].sentenceId, conlltokens)
                        #print('refcon:', gtid2ct[refcon].token)
                        #print('fullsent:', fullsent)
                        #print('predicted sent ids:', predictedSentIdIntArgTokens)
                        #print('predicted ids:', predictedIntArgTokens)
                        #print('reconstructed from ids:', ' '.join([gtid2ct[x].token for x in predictedIntArgTokens]))

                        # adding clause final punctuation:
                        if predictedIntArgTokens[-1]+1 in gtid2ct: # will otherwise crash if intArg is at end of file
                            if gtid2ct[predictedIntArgTokens[-1]+1].token in string.punctuation:#set([',', '.', '?', '!']):
                                predictedIntArgTokens.append(predictedIntArgTokens[-1]+1)
                                # play with this, perhaps in PDTB regulations it should not be part of intarg...

                        # excluding the connective token(s)
                        predictedIntArgTokens = [x for x in predictedIntArgTokens if not x in conntokens]

                        # taking only whatever is to the right of the conn if it is a conjunction
                        if token.pos_.endswith('CONJ'): # also found an SCONJ in there
                            predictedIntArgTokens = [x for x in predictedIntArgTokens if x > gtid2ct[refcon].globalTokenId]



                        for tokenId in set(actualIntArgTokens + predictedIntArgTokens):
                            if tokenId in actualIntArgTokens and tokenId in predictedIntArgTokens:
                                tp += 1
                            elif tokenId in actualIntArgTokens:
                                fn += 1
                            else:
                                fp += 1

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    
    return precision, recall, f1                        




if __name__ == '__main__':

    conllfilelist = CONLLParser.getInputfiles('/home/peter/osloCoCop/conll16/en.train/conll_format')
    #conllTokens = []
    #for cf in conllfilelist:
        #localTokens = CONLLParser.parsePDTBFile(cf)
        #conllTokens += localTokens

    """
    for ct in conllTokens:
        print(ct.token)
        print(ct.fileId)
        print(ct.globalTokenId)
        print(ct.sentenceId)
        print(ct.sentenceTokenId)
        print()
    """

    relations = parseRelationsJson('/home/peter/osloCoCop/conll16/en.train/relations.json')

    #checkAlignment(conllfilelist, relations) # only for debugging/checking if alignment is right

    
    #p, r, f = dep_approach(conllfilelist, relations)
    #print('Results for spacy dependency based approach, finding the connective, taking its head and then all its subtree (recursively get all its children):')
    #print('\tp: %s' % str(p))
    #print('\tr: %s' % str(r))
    #print('\tf: %s' % str(f))


    p, r, f = const_approach(conllfilelist, relations)
    print('Results for constituent based approach, finding the first S, CS or VP label going up in the tree from the first connective token (with some added extras):')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))
