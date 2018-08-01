import PCCParser
import os
import re
import codecs
from collections import defaultdict
import sys
from nltk.parse import stanford
from nltk.tree import ParentedTree
import pickle

JAVAHOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64'
STANFORD_PARSER='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
STANFORD_MODELS='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
os.environ['JAVAHOME'] = JAVAHOME
os.environ['STANFORD_PARSER'] = STANFORD_PARSER
os.environ['STANFORD_MODELS'] = STANFORD_MODELS

lexParserPath = 'edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz'
lexParser = stanford.StanfordParser(model_path=lexParserPath)


# script to extract the (word) span of intArg.


def extractGoldArguments(fileversions):

    file2tokenlist = defaultdict(list)
    file2discourserelations = defaultdict(lambda : defaultdict(DiscourseRelation))
    file2tid2dt = {}
    for fno, name in enumerate(fileversions):
        sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist, discourserelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourserelations[name] = discourserelations
        file2tid2dt[name] = tid2dt
                
    return file2tokenlist, file2discourserelations, file2tid2dt


def autoparse_const_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt):

    tp = 0
    fp = 0
    fn = 0
    
    # TODO: error analysis on output, should be able to get ideas from that now

    
    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
            connSentIds = set([x.sentenceId for x in conndts])
            actualIntArgTokens = [int(x) for x in dr.intArgTokens]
            connTokenIds = [x4.tokenId for x4 in conndts]
            connSentIds = set([x.sentenceId for x in conndts])
            if not actualIntArgTokens:
                print('dr is without intarg tokens:', dr)
                print(dr.relationId)
                print(fname)
            
            # maybe skip multiple sents first (check how many there are and then decide)
            if len(connSentIds) > 1:
                print('multiple sents')
            else:
                if conndts: # not the case for rid 4 in maz-7690 (error in PCC)
                    refcon = conndts[len(conndts)-1]
                    fullsent = refcon.fullSentence
                    tree = parsermemorymap[fullsent]

                    #if int(dr.relationId) == 5 and fname == 'maz-10374':
                        #print('debug fullsent:', fullsent)
                        #print('debug tree:', tree)

                    # now locate conn in tree and check for some parent nodes (S, relS, VP, etc.) which gives the best results
                    leaveno = 0
                    if tree.leaves()[refcon.sentencePosition] == refcon.token:
                        leaveno = refcon.sentencePosition
                    else:
                        # probably due to round brackets that I removed because the nltk parser crashes on them
                        bracketno = 0
                        restored = False
                        if re.search('[\(\)]', refcon.fullSentence):
                            for char in ' '.join(refcon.fullSentence.split()[:refcon.sentencePosition]):
                                if char == '(' or char == ')':
                                    bracketno += 1
                        if bracketno:
                            if tree.leaves()[refcon.sentencePosition-bracketno] == refcon.token:
                                restored = True
                                leaveno = refcon.sentencePosition-bracketno

                        # this is not supposed to happen
                        if not restored:
                            print('positions do not match')
                            print('tree leaves:', tree.leaves())
                            print('conntoken:', refcon.token)
                            print('pos:', refcon.sentencePosition)
                            print('fullsent:', refcon.fullSentence)


                    # now have the tree position, extract first S/VP/experiment going up from that node (ultimately taking the ROOT node if the specified label is not encountered in the way up in the tree)
                    plain_tokens = get_parent_phrase_types(tree, leaveno, ['S', 'CS'], refcon)
                    # now convert the list of plain tokens (leaves) to token ids, since actualIntArgTokens are ids, and it's safer than plain tokens (could be duplicates)
                    
                    id2Token = getSentenceId2Token(fullsent, refcon)
                    predictedIntArgTokens = matchPlainTokensWithIds(plain_tokens, id2Token)

                    # TODO: figure out if there is some subclass that typically drops the conn from intArg

                    #if not list(set(predictedIntArgTokens) & set(actualIntArgTokens)):
                        #print(fname)
                        #print(dr.relationId)
                        #print('predicted:', predictedIntArgTokens)
                        #print('actual:', actualIntArgTokens)

                    plaintextpredicted = plain_tokens
                    plaintextactual = ' '.join([f2tid2dt[fname][str(x)].token for x in actualIntArgTokens])
                    if not plaintextpredicted == plaintextactual:
                        print('tree:', tree)
                        print('conn:', refcon.token)
                        print('Predicted:', plaintextpredicted)
                        print('actual:   ', plaintextactual)
                        print()
                        
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
    

                    

def matchPlainTokensWithIds(plain_tokens, id2Token):

    _sorted = sorted(id2Token.items(), key = lambda x: x[0])
    for i, pair in enumerate(_sorted):
        if plain_tokens[0] == pair[1]:
            if plain_tokens[1] == _sorted[i+1][1]:
                anchor = _sorted[0][0]
                if len(plain_tokens) == 2:
                    anchor = _sorted[i][0]
                elif plain_tokens[2] == _sorted[i+2][1]:
                    anchor = _sorted[i][0]
                else:
                    pass#print('Taking default anchor!')
                    #sys.exit(1)
                ret = []
                for i in range(anchor, anchor + len(plain_tokens)):
                    ret.append(i)
                return ret


def getSentenceId2Token(fullsent, ct):

    id2token = {}
    sp = ct.sentencePosition
    tokens = fullsent.split()
    for i, j in enumerate(tokens):
        diff = sp - i
        tokenId = int(ct.tokenId) - diff
        id2token[tokenId] = j
        
    return id2token
                
def get_parent_phrase_types(tree, pos, labels, ct):

    for i, node in enumerate(tree.pos()):
        if i == pos:
            nodePosition = tree.leaf_treeposition(i)
            # little hack to first look if conn is conjunction, in which case I probably want to take child S node
            pt = ParentedTree.convert(tree)
            children = pt[nodePosition[:1]]
            print('deb child:', children)
            # TODO: get (direct) child node here, if that's an S, take only that.
            # end of hack (if this works (improves), code it neatly)
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

                
def less_naive_baseline(f2dr, sid2dts, f2tid2dt):

    # almost same as baseline, except that I only take the words coming after the connective in the same sentence.
    tp = 0
    fp = 0
    fn = 0
    
    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            if dr: # was None in some cases, not sure why
                conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
                connSentIds = set([x.sentenceId for x in conndts])
                actualIntArgTokens = dr.intArgTokens
                lastConnectiveTokenId = max([x6.tokenId for x6 in conndts])
                baselineIntArgTokens = []
                for sid in connSentIds:
                    for x1 in sid2dts[fname][sid]:
                        
                        if x1.tokenId > lastConnectiveTokenId:
                            baselineIntArgTokens.append(x1.tokenId)

                for tokenId in set(actualIntArgTokens + baselineIntArgTokens):
                    if tokenId in actualIntArgTokens and tokenId in baselineIntArgTokens:
                        tp += 1
                    elif tokenId in actualIntArgTokens:
                        fn += 1
                    else:
                        fp += 1

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1
    

def baseline(f2dr, sid2dts, f2tid2dt):

    # doing my own precision recall based metric for now. Ask Manfred when he's back from holidays what the best metric for this task would be.
    tp = 0
    fp = 0
    fn = 0
    
    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            if dr: # was None in some cases, not sure why
                conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
                connSentIds = set([x.sentenceId for x in conndts])
                actualIntArgTokens = dr.intArgTokens
                baselineIntArgTokens = []
                for sid in connSentIds:
                    for x1 in sid2dts[fname][sid]:
                        if not x1.tokenId in set(dr.connectiveTokens):
                            baselineIntArgTokens.append(x1.tokenId)

                for tokenId in set(actualIntArgTokens + baselineIntArgTokens):
                    if tokenId in actualIntArgTokens and tokenId in baselineIntArgTokens:
                        tp += 1
                    elif tokenId in actualIntArgTokens:
                        fn += 1
                    else:
                        fp += 1

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1

                        
def getSentenceId2DiscourseTokensDict(f2tl):

    sid2dts = defaultdict(lambda : defaultdict(list))
    for fname in f2tl:
        for dt in f2tl[fname]:
            sid2dts[fname][dt.sentenceId].append(dt)
    return sid2dts

                
if __name__ == '__main__':


    #connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/connectors/')
    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/standoffConnectors/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

    f2tl, f2dr, f2tid2dt = extractGoldArguments(fileversions)

    sid2dts = getSentenceId2DiscourseTokensDict(f2tl)
    # first naive attempt, get whole sentence of connective as intArg
    p, r, f = baseline(f2dr, sid2dts, f2tid2dt)
    print('Results for baseline implementation for intArg (taking the entire sentence(s) the connective appears in):')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))

    p, r, f = less_naive_baseline(f2dr, sid2dts, f2tid2dt)
    print('Results for less naive baseline implementation for intArg (taking the words in the same sentence(s) after the connective only):')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))

    #load/parse all sentences first:
    parsermemorymap = {}
    if os.path.exists('parsermemory.pickle'):
        parsermemorymap = pickle.load(codecs.open('parsermemory.pickle', 'rb'))
        sys.stderr.write('INFO: Loaded parse trees from pickled dict.\n')
    else:
        for fname in f2tl:
            sys.stderr.write('INFO: Parsing sents of: %s\n' % fname)
            for dt in f2tl[fname]:
                fullsent = dt.fullSentence
                if not fullsent in parsermemorymap:
                    tree = lexParser.parse(fullsent.split())
                    parentedTreeIterator = ParentedTree.convert(tree)
                    for t in parentedTreeIterator:
                        parentedTree = t
                        break # always taking the first, assuming that this is the best scoring tree...
                    parsermemorymap[fullsent] = parentedTree
        with open('parsermemory.pickle', 'wb') as handle:
            pickle.dump(parsermemorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write('INFO: Pickled parse trees to parsermemory.pickle.\n')
    
    p, r, f = autoparse_const_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt)
    print('Results for constituent based approach, finding the first S label going up in the tree from the first connective token:')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))
    
    # can also compare auto parses to gold parses (both for const and deps; check gitlab) here! :)

    
    
