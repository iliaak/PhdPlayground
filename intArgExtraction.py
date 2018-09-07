import PCCParser
import os
import string
import re
import codecs
from collections import defaultdict
import sys
from nltk.parse import stanford
from nltk.tree import ParentedTree
import pickle
#from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
import dill
import spacy
import lxml



JAVAHOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64'
STANFORD_PARSER='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
STANFORD_MODELS='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
os.environ['JAVAHOME'] = JAVAHOME
os.environ['STANFORD_PARSER'] = STANFORD_PARSER
os.environ['STANFORD_MODELS'] = STANFORD_MODELS

lexParserPath = 'edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz'
lexParser = stanford.StanfordParser(model_path=lexParserPath)
depParser = StanfordDependencyParser(path_to_jar=os.path.join(STANFORD_PARSER, 'stanford-parser.jar'), path_to_models_jar=os.path.join(STANFORD_MODELS, 'stanford-german-corenlp-2017-06-09-models.jar'))#, java_options='-mx4g')


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


def goldparse_const_approach(f2dr, sid2dts, f2tid2dt, fileversions):

    tp = 0
    fp = 0
    fn = 0

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    f2ssid2goldtree = defaultdict(lambda : defaultdict())
    f2ssid2terminals = defaultdict(lambda : defaultdict())
    for fname in fileversions:
        tree = lxml.etree.parse(fileversions[fname]['syntax'], parser = xmlParser)
        for body in tree.getroot():
            for elemid, sentence in enumerate(body):
                sid = sentence.get('id')
                graph = sentence.getchildren()[0]
                nonterminalNodes = graph.find('.//nonterminals')
                terminalIds = [t.get('id') for t in graph.find('.//terminals')]
                f2ssid2goldtree[fname][sid] = nonterminalNodes
                f2ssid2terminals[fname][sid] = terminalIds
    
    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
            connSentIds = set([x.sentenceId for x in conndts])
            actualIntArgTokens = [int(x) for x in dr.intArgTokens]
            connTokenIds = [x4.tokenId for x4 in conndts]
            connSentIds = set([x.sentenceId for x in conndts])
            if not actualIntArgTokens:
                pass#print('dr is without intarg tokens;', dr)
                #print(dr.relationId)
                #print(fname)
                
            if len(connSentIds) > 1:
                pass#print('multiple sents')
            else:
                if conndts: # not the case for rid 4 in maz-7690 (error in PCC)
                    refcon = conndts[len(conndts)-1]
                    fullsent = refcon.fullSentence
                    goldtree = f2ssid2goldtree[fname][refcon.syntaxSentenceId]
                    subdict, catdict = PCCParser.getSubDict(goldtree)
                    nodeId = None
                    syntermid2tokenId = defaultdict()
                    for dt in sid2dts[fname][refcon.sentenceId]:
                        syntermid2tokenId[dt.syntaxId] = dt.tokenId
                    for ntn in goldtree:
                        for edge in ntn:
                            if edge.get('idref') == refcon.syntaxId:
                                nodeId = ntn.get('id')
                    #parentNode = PCCParser.getParentNode(nodeId, subdict)
                    leafs = PCCParser.getLeafsFromGoldTree(nodeId, subdict, f2ssid2terminals[fname][refcon.syntaxSentenceId], [])
                    predictedIntArgTokens = [int(syntermid2tokenId[x]) for x in leafs]

                    # adding clause final punctuation:
                    if str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1) in f2tid2dt[fname]: # will otherwise crash if intArg is at end of file
                        if f2tid2dt[fname][str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)].token in string.punctuation:#set([',', '.', '?', '!']):
                            predictedIntArgTokens.append(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)

                    # excluding the connective token(s)
                    predictedIntArgTokens = [x for x in predictedIntArgTokens if not str(x) in connTokenIds]

                    # taking only whatever is to the right of the conn if it is a conjunction
                    if refcon.pos.startswith('KON'):
                        predictedIntArgTokens = [x for x in predictedIntArgTokens if x > int(refcon.tokenId)]

                    # in the autoparse scenario I'm also doing this. Can try it here as well
                    #elif refcontype == 'PROAV' or refcontype.startswith('A') or refcontype.startswith('K'):
                        #plain_tokens = get_parent_phrase_plus_phrase_after_comma(tree, leaveno, ['S', 'CS', 'VP'], refcon)
                    
                    
                    
                    # debugging section:
                    plaintextpredicted = ' '.join([f2tid2dt[fname][str(x)].token for x in predictedIntArgTokens])
                    plaintextactual = ' '.join([f2tid2dt[fname][str(x)].token for x in actualIntArgTokens])
                    """
                    if not plaintextpredicted == plaintextactual:
                    print('tree:', tree)
                    print('conn:', refcon.token)
                    print('spacy pos:', token.pos_)
                    print('Predicted:', plaintextpredicted)
                    print('actual:   ', plaintextactual)
                    print()
                    """
                            
                    
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

                    

def autoparse_dep_spacy_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt):

    tp = 0
    fp = 0
    fn = 0

    #this whole pickling of spacy docs seems not to work, and parsing is quite fast anyway, so might as well do it on the fly...
    sys.stderr.write('INFO: Loading German spacy stuff...\n')
    #nlp = spacy.load('de')
    nlp = spacy.load('de_core_news_sm')
    sys.stderr.write('INFO: Done.\n')


    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
            connSentIds = set([x.sentenceId for x in conndts])
            actualIntArgTokens = [int(x) for x in dr.intArgTokens]
            connTokenIds = [x4.tokenId for x4 in conndts]
            connSentIds = set([x.sentenceId for x in conndts])
            if not actualIntArgTokens:
                print('dr is without intarg tokens;', dr)
                print(dr.relationId)
                print(fname)
                
            if len(connSentIds) > 1:
                pass#print('multiple sents')
            else:
                if conndts: # not the case for rid 4 in maz-7690 (error in PCC)
                    refcon = conndts[len(conndts)-1]
                    fullsent = refcon.fullSentence
                    #tree = parsermemorymap[fullsent]
                    tree = nlp(fullsent)
                    
                    #print('fullsent:', fullsent)
                    #print('pos of con:', refcon.sentencePosition)
                    #print('con:', refcon.token)
                    #print('len of tree:', len(tree))
                    for index, token in enumerate(tree):
                        if index == refcon.sentencePosition and refcon.token == token.text: # not always the case, due to some tokenisation differences (spacy is not very configurable...)
                            
                            predictedSentIdIntArgTokens = [x.i for x in token.head.subtree]
                            predictedIntArgTokens = convertSentIdstoDocIds(predictedSentIdIntArgTokens, refcon)
                            #print('conn pos:', token.pos_)
                            #print('conn deps:', token.dep_)
                            #print('conn head:', token.head.text)
                            #print('children:', [child for child in token.children])
                            #print('children of the head:', [child for child in token.head.children])
                            #print('all children of the head subtree:', [x.text for x in token.head.subtree])
                            #print('head indeX:', token.head.i)
                            #print('conn index:', token.i)
                            #print('\n\n')
                    

                            # adding clause final punctuation:
                            if str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1) in f2tid2dt[fname]: # will otherwise crash if intArg is at end of file
                                if f2tid2dt[fname][str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)].token in string.punctuation:#set([',', '.', '?', '!']):
                                    predictedIntArgTokens.append(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)
                                    # 86.19

                            
                            # excluding the connective token(s)
                            predictedIntArgTokens = [x for x in predictedIntArgTokens if not str(x) in connTokenIds]

                            # taking only whatever is to the right of the conn if it is a conjunction
                            if token.pos_.endswith('CONJ'): # also found an SCONJ in there
                                predictedIntArgTokens = [x for x in predictedIntArgTokens if x > int(refcon.tokenId)]


                            """
                            # cut all words from the next connective onward (if there is any, in the intArg) # EDIT: decreases score from 86.19 to 78.47. Perhaps look into special POS types of connectives...?
                            for i, x in enumerate(predictedIntArgTokens):
                                if hasattr(f2tid2dt[fname][str(x)], 'isConnective'):
                                    if f2tid2dt[fname][str(x)].isConnective:
                                        predictedIntArgTokens = predictedIntArgTokens[:i]
                            """
                            # guess this is more of a sentence segmentation error repair, but if the final char is colon (:), include also the next sentence: # EDIT: decreases score...
                            """
                            if predictedIntArgTokens:
                                if f2tid2dt[fname][str(predictedIntArgTokens[len(predictedIntArgTokens)-1])].token == ':':
                                    nextsentIds = [x.tokenId for x in sid2dts[fname][refcon.sentenceId+1]]
                                    for y in nextsentIds:
                                        predictedIntArgTokens.append(y)
                            """
                            
                            
                            # debugging section:
                            #plaintextpredicted = ' '.join([f2tid2dt[fname][str(x)].token for x in predictedIntArgTokens])
                            #plaintextactual = ' '.join([f2tid2dt[fname][str(x)].token for x in actualIntArgTokens])
                            """
                            if not plaintextpredicted == plaintextactual:
                                print('tree:', tree)
                                print('conn:', refcon.token)
                                print('spacy pos:', token.pos_)
                                print('Predicted:', plaintextpredicted)
                                print('actual:   ', plaintextactual)
                                print()
                            """
                            
                            
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

                            

def convertSentIdstoDocIds(predictedSentIdIntArgTokens, refcon):

    output = []
    for i, tsid in enumerate(predictedSentIdIntArgTokens):
        if tsid == refcon.sentencePosition:
            for j, tsid2 in enumerate(predictedSentIdIntArgTokens):
                if j < i:
                    output.append(int(refcon.tokenId) - (i - j))
                elif i == j:
                    output.append(int(refcon.tokenId))
                else:
                    output.append(int(refcon.tokenId) + (j - i))
    return output

def autoparse_dep_stanford_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt):

    tp = 0
    fp = 0
    fn = 0

    for fname in f2dr:
        discourserelations = f2dr[fname]
        for dr in discourserelations:
            conndts = [f2tid2dt[fname][x] for x in dr.connectiveTokens]
            connSentIds = set([x.sentenceId for x in conndts])
            actualIntArgTokens = [int(x) for x in dr.intArgTokens]
            connTokenIds = [x4.tokenId for x4 in conndts]
            connSentIds = set([x.sentenceId for x in conndts])
            if not actualIntArgTokens:
                print('dr is without intarg tokens;', dr)
                print(dr.relationId)
                print(fname)
                
            if len(connSentIds) > 1:
                pass#print('multiple sents')
            else:
                if conndts: # not the case for rid 4 in maz-7690 (error in PCC)
                    refcon = conndts[len(conndts)-1]
                    fullsent = refcon.fullSentence
                    tree = parsermemorymap[fullsent]
                    # will probably want to differentiate between K and A (and P) type connectives, but let's take all deps first...
                    
                    print('fullsent:', fullsent)
                    print('pos of con:', refcon.sentencePosition)
                    print('con:', refcon.token)
                    # note, if I want to re-introduce this, used get_node_by_addr or something (see traverseDeps) to get conndepnode, which is now missing...
                    print('deps:', tree)
                    
                    print('\n\n')
                    debugconndeptypes = []
                    incl = []

                    #TODO: do something about terrible recall
                    
                    for dep in conndepnode['deps']:
                        if not dep in set(['nsubj']):
                            debugconndeptypes.append(dep)
                            for pos2 in conndepnode['deps'][dep]:
                                incl = traverseDeps(tree, pos2, incl)
                                # will want to consider sth like 'if dep in set(a,b,c)' where abc are dep types
                        
                        
                    plain_tokens = []
                    for pos in sorted(incl):
                        token = tree.get_by_address(pos)['word']
                        plain_tokens.append(token)

                    id2Token = getSentenceId2Token(fullsent, refcon)
                    if plain_tokens:
                        predictedIntArgTokens = matchPlainTokensWithIds(plain_tokens, id2Token)
                        if predictedIntArgTokens:
                        
                            if str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1) in f2tid2dt[fname]: # will otherwise crash if intArg is at end of file
                                if f2tid2dt[fname][str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)].token in string.punctuation:#[',', '.']:
                                    predictedIntArgTokens.append(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)
                            
                            predictedIntArgTokens = [x for x in predictedIntArgTokens if not str(x) in connTokenIds]
                            if predictedIntArgTokens:


                                if not list(set(predictedIntArgTokens) & set(actualIntArgTokens)):
                                    print(fname)
                                    print(dr.relationId)
                                    print('predicted:', predictedIntArgTokens)
                                    print('actual:', actualIntArgTokens)
                                    print('dep types of conn:', debugconndeptypes)
                                
                                plaintextpredicted = ' '.join([f2tid2dt[fname][str(x)].token for x in predictedIntArgTokens])
                                plaintextactual = ' '.join([f2tid2dt[fname][str(x)].token for x in actualIntArgTokens])
                                
                                
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
                    
                       

def autoparse_const_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt):

    tp = 0
    fp = 0
    fn = 0
    
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
                pass#print('multiple sents')
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

                    refconPosition = tree.leaf_treeposition(leaveno)
                    pt = ParentedTree.convert(tree)
                    connnode = pt[refconPosition[:-1]]
                    refcontype = connnode.label()

                    plain_tokens = []
                    if refcontype == 'KON':
                        plain_tokens = get_right_sibling(tree, leaveno, refcon)
                    elif refcontype == 'PROAV' or refcontype.startswith('A') or refcontype.startswith('K'):
                        plain_tokens = get_parent_phrase_plus_phrase_after_comma(tree, leaveno, ['S', 'CS', 'VP'], refcon)
                    else:
                        plain_tokens = get_parent_phrase(tree, leaveno, ['S', 'CS', 'VP'], refcon)
                        

                    # now convert the list of plain tokens (leaves) to token ids, since actualIntArgTokens are ids, and it's safer than plain tokens (could be duplicates)
                    
                    id2Token = getSentenceId2Token(fullsent, refcon)
                    predictedIntArgTokens = matchPlainTokensWithIds(plain_tokens, id2Token)


                    
                    # exclude the connective  (improved f-score from 75.56 to 78.43)
                    predictedIntArgTokens = [x for x in predictedIntArgTokens if not str(x) in connTokenIds]

                    # stupid fix; re-inserting the comma after the last word if there is one (tree-based approach excludes this, PCC includes it...)
                    if str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1) in f2tid2dt[fname]: # will otherwise crash if intArg is at end of file
                        if f2tid2dt[fname][str(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)].token in string.punctuation:#[',', '.']:
                            predictedIntArgTokens.append(predictedIntArgTokens[len(predictedIntArgTokens)-1]+1)

                    
                    #if not list(set(predictedIntArgTokens) & set(actualIntArgTokens)):
                        #print(fname)
                        #print(dr.relationId)
                        #print('predicted:', predictedIntArgTokens)
                        #print('actual:', actualIntArgTokens)

                    plaintextpredicted = ' '.join([f2tid2dt[fname][str(x)].token for x in predictedIntArgTokens])
                    plaintextactual = ' '.join([f2tid2dt[fname][str(x)].token for x in actualIntArgTokens])
                    """
                    if not plaintextpredicted == plaintextactual:
                        print('tree:', tree)
                        print('conn:', refcon.token)
                        print('Predicted:', plaintextpredicted)
                        print('actual:   ', plaintextactual)
                        print()
                    """
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
    # should not happen...  (if it does, probably due to round brackets, which are removed because nltk crashes on them)      
    #print('Could not find a match for:', plain_tokens)
    #print('sorted:', _sorted)
    #print('Dying now!')
    #sys.exit(1)


def getSentenceId2Token(fullsent, ct):

    id2token = {}
    sp = ct.sentencePosition
    tokens = fullsent.split()
    for i, j in enumerate(tokens):
        diff = sp - i
        tokenId = int(ct.tokenId) - diff
        id2token[tokenId] = j
        
    return id2token

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


def traverseDeps(deps, addr, incl):

    dep = deps.get_by_address(addr)
    for _type in dep['deps']:
        for addr2 in dep['deps'][_type]:
            incl.append(addr2)
            incl = traverseDeps(deps, addr2, incl)
    return incl

"""
def traverse(deps, addr):

    dep = deps.get_by_address(addr)
    print(dep)
    for d in dep['deps']:
        for addr2 in dep['deps'][d]:
            traverse(deps, addr2)
"""

if __name__ == '__main__':

    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/connectives_standoff/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/tokenized/')
    
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
    pmname = 'parsermemory.pickle'
    if os.path.exists(pmname):
        parsermemorymap = pickle.load(codecs.open(pmname, 'rb'))
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
        with open(pmname, 'wb') as handle:
            pickle.dump(parsermemorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write('INFO: Pickled parse trees to %s.\n' % pmname)

        
    parsermemorymap2 = {} # this one is for dep parses
    pm2name = 'dependencyparsermemory.pickle'
    if os.path.exists(pm2name):
        parsermemorymap2 = dill.load(codecs.open(pm2name, 'rb'))
        sys.stderr.write('INFO: Loaded dep parse trees from pickled dict.\n')
    else:
        sys.stderr.write('INFO: Loading German spacy stuff...\n')
        #nlp = spacy.load('de')
        nlp = spacy.load('de_core_news_sm')
        sys.stderr.write('INFO: Done.\n')
        for fno, fname in enumerate(f2tl):
            sys.stderr.write('INFO: Parsing sents of: %s (%i of %i)\n' % (fname, fno+1, len(f2tl)))
            # spacy depparser code
            for dt in f2tl[fname]:
                fullsent = dt.fullSentence
                if not fullsent in parsermemorymap2:
                    doc = nlp(fullsent)
                    #if not len(doc) == len(fullsent.split()): NOTE: skipping over tokenisation differences for now. Let's hope it doesn't affect sentences which are of interest for me...
                        #sys.stderr.write("ERROR: tokenization difference for '%s'. Investigate, skipping this one for now, please re-parse manually later.\n" % fullsent)
                    #else:
                    parsermemorymap2[fullsent] = doc
            
            """ # stanford depparser code
            for dt in f2tl[fname]:
                fullsent = dt.fullSentence
                if not fullsent in parsermemorymap2:
                    deptree = depParser.parse(fullsent.split()).__next__()
                    parsermemorymap2[fullsent] = deptree
            """
        with open(pm2name, 'wb') as handle:
            dill.dump(parsermemorymap2, handle)
        sys.stderr.write('INFO: Pickled parse trees to %s.\n' % pm2name)
    
    p, r, f = autoparse_const_approach(f2dr, sid2dts, parsermemorymap, f2tid2dt)
    print('Results for constituent based approach, finding the first S, CS or VP label going up in the tree from the first connective token (with some added extras):')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))

    
    # can also compare auto parses to gold parses (both for const and deps; check gitlab) here! :)

    
    # results were dramatic for stanford (didn't look into errors manually a lot, but started at around 0.47)
    #p, r, f = autoparse_dep_stanford_approach(f2dr, sid2dts, parsermemorymap2, f2tid2dt)
    #print('Results for dependency based approach, finding the connective and taking all its (X-typed?) depdencies:')
    #print('\tp: %s' % str(p))
    #print('\tr: %s' % str(r))
    #print('\tf: %s' % str(f))

    p, r, f = autoparse_dep_spacy_approach(f2dr, sid2dts, parsermemorymap2, f2tid2dt)
    print('Results for spacy dependency based approach, finding the connective, taking its head and then all its subtree (recursively get all its children):')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))


    
    p, r, f = goldparse_const_approach(f2dr, sid2dts, f2tid2dt, fileversions)
    print('Results for const approach with gold trees:')
    print('\tp: %s' % str(p))
    print('\tr: %s' % str(r))
    print('\tf: %s' % str(f))
    
    print('TODO: Figure out if I really implemented the same set of rules for auto and gold const trees (since gold trees score worse)')
