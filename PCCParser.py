#!/usr/bin/env python3

import sys
import re
import string
from collections import defaultdict
from optparse import OptionParser
import os
import codecs
import lxml.etree
from itertools import chain

"""
Potsdam Commentary Corpus Parser for some initial playing around with the format
"""
connectorTokenId = 1
syntaxTokenId = 0 # bit ugly, but because of zero-index list...
rstTokenId = 0 # idem


class DiscourseToken:
    def __init__(self, tokenId, token):
        self.tokenId = tokenId
        self.token = token

    # should do some cleaning up at some point to remove stuff never used
    def setIntOrExt(self, val):
        self.intOrExt = val
    def setMultiWordBoolean(self, val):
        self.isMultiWord = val
    def setConnectiveBoolean(self, val):
        self.isConnective = val
    def setUnitId(self, val):
        self.unitId = val
    def setRelation(self, val):
        self.relation = val
    def setType(self, val):
        self.segmentType = val
    def setLemma(self, val):
        self.lemma = val
    def setPOS(self, val):
        self.pos = val
    def setTerminalsId(self, val):
        self.terminalsId = val
    def setMorph(self, val):
        self.morph = val
    def setRstSegmentId(self, val):
        self.rstSegmentId = val
    def setRstParent(self, val):
        self.rstParent = val
    def setRelname(self, val):
        self.relname = val
    def setIsIntArg(self, val):
        self.isIntArg = val
    def setIsExtArg(self, val):
        self.isExtArg = val
    def setPathToRoot(self, val):
        self.pathToRoot = val
    def setCompressedPath(self, val):
        self.compressedPath = val
    def setLeftsiblingCat(self, val):
        self.leftsiblingCat = val
    def setRightsiblingCat(self, val):
        self.rightsiblingCat = val
    def setRightsiblingContainsVP(self, val):
        self.rightsiblingContainsVP = val
    def setParentCategory(self, val):
        self.parentCategory = val
    def setSiblings(self, val):
        self.siblings = val
    def setPostSOS(self, val):
        self.postSOS = val
    def setPreEOS(self, val):
        self.preEOS = val
    def setSelfCategory(self, val):
        self.selfCategory = val
    def setSyntaxNodeId(self, val):
        self.syntaxId = val
    def setSentenceId(self, val):
        self.sentenceId = val
    def addFullSentence(self, val):
        self.fullSentence = val
    def addPreviousSentence(self, val):
        self.previousSentence = val
    def setConnectiveId(self, val):
        self.connectiveId = val
    def setParagraphId(self, pi):
        self.paragraphId = pi
    def setFullParagraph(self, par):
        self.fullParagraph = par
    def setPositionInParagraph(self, pos):
        self.positionInParagraph = pos
    def setParagraphInitial(self, b):
        self.paragraphInitial = b
    def setEmbeddedUnits(self, val):
        self.embeddedUnits = val
        
        
def addAnnotationLayerToDict(flist, fdict, annname):

    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict


def parseConnectorFile(connectorxml):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(connectorxml, parser = xmlParser)
    discourseTokens = extractTokens(tree.getroot(), [])
    global connectorTokenId
    connectorTokenId = 1 # reset at end of file
    return discourseTokens

def getSubDict(nonterminalnodes):

    d = {}
    cat = {}
    for nt in nonterminalnodes:
        edges = []
        for edge in nt:
            if edge.tag == 'edge': # some have secedges, which cause nesting/loops
                edges.append(edge.get('idref'))
        d[nt.get('id')] = edges
        cat[nt.get('id')] = nt.get('cat')
    return d, cat

def getPathToRoot(tid, subdict, catdict):
    
    chain = []
    traversedict(subdict, tid, chain)
    return [catdict[x] for x in chain]

def traversedict(d, key, chain):

    for k, v in d.items():
        for val in v:
            if key == val:
                chain.append(k)
                traversedict(d, k, chain)
    return chain

def compressRoute(r): # filtering out adjacent identical tags
        
    delVal = "__DELETE__"
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            r[i+1] = delVal
    return [x for x in r if x != delVal]

def getParentNode(tid, subdict):

    parentNode = None
    for k, v in subdict.items():
        for val in v:
            if tid == val:
                parentNode = k
    return parentNode

def getSiblings(tid, subdict, terminalnodeids):

    parentNode = getParentNode(tid, subdict)
    siblings = []
    if parentNode:
        for k, v in subdict.items():
            if parentNode in v:
                for val in v:
                    if not val == parentNode and not val in terminalnodeids:
                        siblings.append(val)
    return siblings
    
def getSibling(tid, subdict, direction, terminalnodeids):

    """
    parentNode = None
    for k, v in subdict.items():
        for val in v:
            if tid == val:
                parentNode = k
    """
    parentNode = getParentNode(tid, subdict)
    siblings = []
    if parentNode:
        for k, v in subdict.items():
            if parentNode in v:
                for val in v:
                    if not val == parentNode and not val in terminalnodeids:
                        siblings.append(val)
    
    # assuming here that nodes are linearly ordered. Is this always true? Think not really. Question is whether/how much it matters. Perhaps even experimenting with taking all siblings?
    tempdict = {}
    for sib in siblings:
        tempdict[int(re.sub('^[^_]+_', '', sib))] = sib
    leftSibling = None
    rightSibling = None
    if parentNode:
        pnint = int(re.sub('^[^_]+_', '', parentNode))
        siblings = sorted([int(re.sub('^[^_]+_', '', sib)) for sib in siblings] + [pnint])
        if siblings.index(pnint) > 0:
            leftSibling = tempdict[siblings[siblings.index(pnint)-1]]
        if siblings.index(pnint) < len(siblings)-1:
            rightSibling = tempdict[siblings[siblings.index(pnint)+1]]
        
    if direction == 'left':
        return leftSibling
    elif direction == 'right':
        return rightSibling
    
def getPhraseSyntaxFeatures(tl, syntaxxml):

    # features I need:
    #_selfcat, _parentcat, _leftsiblingCat, _rightsiblingCat, _rightsiblingContainsVP, _rootpath, _compressedpath
    b = False
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(syntaxxml, parser = xmlParser)
    for body in tree.getroot():
        for sentence in body:
            sid = sentence.get('id')
            graph = sentence.getchildren()[0]
            terminalNodes = graph.find('.//terminals')
            terminalnodeids = [x.get('id') for x in terminalNodes]
            nonterminalNodes = graph.find('.//nonterminals')
            subdict, catdict = getSubDict(nonterminalNodes)
            for ntn in nonterminalNodes:
                idrefs = []
                for subnode in ntn:
                    if subnode.tag == 'edge':
                        idrefs.append(subnode.get('idref'))
                tlids = [x.syntaxId for x in tl]
                c = True
                for tlid in tlids:
                    if not tlid in idrefs:
                        c = False
                #if sorted(idrefs) == sorted([x.syntaxId for x in tl]):
                # if c is still True here, all nodes are in the current nonterminal node
                if c:
                    b = True
                    # bingo: we have one single nonterminal node that contains all of our tokens. Get info from this (selfcat, parentcat, siblings etc.) and done.
                    selfcat = ntn.get('cat')
                    parentNode = getParentNode(ntn.get('id'), subdict)
                    parentCat = None
                    if parentNode:
                        parentCat = catdict[parentNode]
                    fullPathToRoot = getPathToRoot(ntn.get('id'), subdict, catdict)
                    compressedPath = compressRoute(fullPathToRoot[:])
                    leftSibling = getSibling(ntn.get('id'), subdict, 'left', terminalnodeids) # terminalnodeids is used to exclude terminals from list of siblings
                    leftCat = None
                    rightCat = None
                    rightsiblingContainsVP = False
                    if leftSibling:
                        leftCat = catdict[leftSibling]
                    rightSibling = getSibling(ntn.get('id'), subdict, 'right', terminalnodeids)
                    if rightSibling:
                        rightCat = catdict[rightSibling]
                        rightchain = getPathToRoot(rightSibling, subdict, catdict)
                        if 'VP' in rightchain:
                            rightsiblingContainsVP = True
                    return(selfcat, parentCat, leftCat, rightCat, rightsiblingContainsVP, fullPathToRoot, compressedPath)
                    
                        
    if not b:
        # this is due to weird samples from dimlex. The only ones remaining (phrases for which no features could be found) are 'als auch' (4 times; all 4 non-connective, so can be left out anyway), 'Es sei denn' (1 time, as connective, so perhaps also not very interesting) and 'und zwar' (2 times, once as connective, once not, so this one is actually interesting!)
        pass#print('NOMATCH')
        
        
def parseSyntaxFile(syntaxxml, tokenlist):

    global syntaxTokenId
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(syntaxxml, parser = xmlParser)
    sentencedict = defaultdict(str)
    for body in tree.getroot():
        for elemid, sentence in enumerate(body):
            sid = sentence.get('id') # may want to add art_id and some others too here
            graph = sentence.getchildren()[0]
            terminalsNode = graph.find('.//terminals') # could also just take first child, but playing around with lxml a bit :)
            nonterminalNodes = graph.find('.//nonterminals')
            tokenisedSentence = ' '.join([x.get('word') for x in terminalsNode])
            sentencedict[elemid] = tokenisedSentence
            subdict, catdict = getSubDict(nonterminalNodes)
            terminalnodeids = [x.get('id') for x in terminalsNode]
            maxId = max([int(re.sub('\D', '', re.sub(r'[^_]+_', '', x))) for x in terminalnodeids])
            for sentenceId, t in enumerate(terminalsNode):
                sToken = t.get('word')
                dt = tokenlist[syntaxTokenId]
                if not sToken == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (syntaxxml, sToken, str(syntaxTokenId), tokenlist[syntaxTokenId].token, str(tokenlist[syntaxTokenId].tokenId)))
                    sys.exit(1)
                    
                dt.setLemma(t.get('lemma'))
                dt.setPOS(t.get('pos'))
                dt.setTerminalsId(t.get('id'))
                dt.setMorph(t.get('morph')) # whatever this is supposed to contain (empty in examples I've seen so far)
                dt.setSyntaxNodeId(t.get('id'))
                if re.sub(r'[^_]+_', '', t.get('id')) == '1':
                    dt.setPostSOS(True)
                else:
                    dt.setPostSOS(False)
                if re.sub(r'[^_]+_', '', t.get('id')) == str(maxId):
                    dt.setPreEOS(True)
                else:
                    dt.setPreEOS(False)
                    
                # may want to go on with nonterminals/deps here
                # indeed; adding the info needed for Lin 2014 feature set here to dt:
                fullPathToRoot = getPathToRoot(t.get('id'), subdict, catdict)
                dt.setPathToRoot(fullPathToRoot)
                compressedPath = compressRoute(fullPathToRoot[:])
                # now get sibling vals (assuming an incrementing terminal node id?)
                leftSibling = getSibling(t.get('id'), subdict, 'left', terminalnodeids) # terminalnodeids is used to exclude terminals from list of siblings
                leftCat = None
                rightCat = None
                rightsiblingContainsVP = False
                if leftSibling:
                    leftCat = catdict[leftSibling]
                rightSibling = getSibling(t.get('id'), subdict, 'right', terminalnodeids)
                if rightSibling:
                    rightCat = catdict[rightSibling]
                    rightchain = getPathToRoot(rightSibling, subdict, catdict)
                    if 'VP' in rightchain:
                        rightsiblingContainsVP = True
                parentNode = getParentNode(t.get('id'), subdict)
                parentCat = None
                if parentNode:
                    parentCat = catdict[parentNode]
                dt.setParentCategory(parentCat)
                allSiblings = getSiblings(t.get('id'), subdict, terminalnodeids)
                dt.setSiblings([catdict[s] for s in allSiblings])
                dt.setSelfCategory(t.get('cat'))
                dt.setCompressedPath(compressedPath)
                dt.setLeftsiblingCat(leftCat)
                dt.setRightsiblingCat(rightCat)
                dt.setRightsiblingContainsVP(rightsiblingContainsVP)
                #if re.search('maz-00001', syntaxxml):
                    #print('debugging full sent:', tokenisedSentence)
                    #print('debugging sid:', elemid)
                dt.addFullSentence(tokenisedSentence)
                dt.setSentenceId(elemid)
                prevsent = None
                if elemid == 0:
                    prevsent = ''
                else:
                    prevsent = sentencedict[elemid-1]
                dt.addPreviousSentence(prevsent)
                
                syntaxTokenId += 1
    syntaxTokenId = 0 # reset at end of file          
    return tokenlist

def parseTokenizedFile(tokenfile, tokenlist):

    tokenizedId = 0
    paragraphId = 0
    paragraphs = codecs.open(tokenfile, 'r').read().split('\n\n')
    for pi, par in enumerate(paragraphs):
        par = re.sub('\n', ' ', par)
        for ti, token in enumerate(par.split()):
            dt = tokenlist[tokenizedId]
            if not token == dt.token:
                sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (tokenfile, token, str(tokenizedId), tokenlist[tokenizedId].token, str(tokenlist[tokenizedId].tokenId)))
            dt.setParagraphId(pi)
            dt.setFullParagraph(par)
            dt.setPositionInParagraph(ti) # if the only thing I need is paragraph initial or paragraph final, could also just fill out only this info here, but this allows me to retrieve it too...
            # EDIT: ah well, redundant coding, but think I will only be interested in paragraph-initialness, really.
            paraInit = False
            if ti == 0:
                paraInit = True
            dt.setParagraphInitial(paraInit)
            tokenizedId += 1
    return tokenlist
            

def parseRSTFile(rstxml, tokenlist):

    global rstTokenId # not sure why I made this global anymore, think it's not needed in this case
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(rstxml, parser = xmlParser)
    body = tree.getroot().find('.//body')
    for node in body:
        if node.tag == 'segment':
            sId = node.get('id')
            sParent = node.get('parent')
            sRelname = node.get('relname')
            tokens = node.text.split()
            for token in tokens:
                dt = tokenlist[rstTokenId]
                if not token == dt.token:
                    sys.stderr.write('FATAL ERROR: Tokens do not match in %s: %s(%s) vs. %s(%s).\n' % (rstxml, sToken, str(rstTokenId), tokenlist[rstTokenId].token, str(tokenlist[rstTokenId].tokenId)))
                    sys.exit(1)
                dt.setRstSegmentId(sId)
                dt.setRstParent(sParent)
                dt.setRelname(sRelname)
                rstTokenId += 1
            # may want to add non-terminal rst tree info here
            
    rstTokenId = 0
    return tokenlist

def getEmbeddedUnits(node, d):

    if node.getparent() is not None:
        parent = node.getparent()
        if parent.get('id'):
            d[parent.get('id')] = parent.get('type')
        getEmbeddedUnits(parent, d)
    return d

  
def createDiscourseToken(token, node, conn, multiWordBool):

    global connectorTokenId
    dt = DiscourseToken(connectorTokenId, token)
    segmentType = ''
    if conn:
        segmentType = 'connective'
    else:
        segmentType = 'unit'
    dt.setType(segmentType)
    dt.setUnitId(node.get('id'))
    embeddedUnits = getEmbeddedUnits(node, {})
    dt.setEmbeddedUnits(embeddedUnits)
    #dt.setEmbedded()
    if node.tag == 'connective':
        dt.setConnectiveId(node.get('id'))
    if segmentType == 'unit':
        pn = node.getparent()
        segmentType = node.get('type')
        if not segmentType: # this elaborate stuff is for tail cases, in which case lxml returns None for node.get('type')
            pn = node.getparent()
            if pn is not None and pn.tag == 'unit':
                segmentType = node.getparent().get('type')
        dt.setIntOrExt(segmentType)
        
    elif segmentType == 'connective':
        
        dt.setRelation(node.get('relation'))
    dt.setMultiWordBoolean(multiWordBool)
    dt.setConnectiveBoolean(conn) # a bit redundant, because the type also contains this info, but this is easier for if checks...
    connectorTokenId += 1
    return dt
    

def extractTokens(node, l):

    conn = False
    if node.text:
        if node.tag == 'connective':
            conn = True
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            multiWordBool = False
            if len(tokens) > 1:
                multiWordBool = True
            for token in tokens:
                dt = createDiscourseToken(token, node, conn, multiWordBool)
                l.append(dt)
        conn = False
    if len(node):
        for subnode in node.getchildren():
            extractTokens(subnode, l)

    if node.tail:
        if node.tail.strip():
            tokens = node.tail.strip().split()
            multiWordBool = False
            if len(tokens) > 1:
                multiWordBool = True
            for token in tokens:
                dt = createDiscourseToken(token, node, conn, multiWordBool)
                l.append(dt)

    return l



def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


def printPlaintext(file2edulist):

    for f in file2edulist:
        print('++++++++++ %s ++++++++++' % f)
        print(re.sub(r'\s+', ' ', ' '.join([x.text for x in file2edulist[f]])))
        print('\n')

def printConllGold(file2edulist):

    for f in file2edulist:
        print('++++++++++ %s ++++++++++' % f)
        # assuming that everything is (still) whitespace-tokenized
        for edu in file2edulist[f]:
            tokens = edu.text.split(' ')
            cb = edu.isConnective
            for i, t in enumerate(tokens):
                word, val = t, 'O'
                if cb:
                    if i == 0:
                        val = 'B-connective'
                    else:
                        val = 'I-connective'
                print('%s\t%s' % (word, val))
        print('\n')

def printStats(file2edulist):

    print('INFO: %i files processed.' % len(file2edulist))
    cn = 0
    ct = 0
    wc = 0
    educ = 0
    for f in file2edulist:
        eduList = file2edulist[f]
        educ += len(eduList)
        for edu in eduList:
            wc += len(edu.text.split(' '))
            if edu.isConnective:
                cn += 1
                ct += len(edu.text.split(' '))

    print('INFO: %i words.' % wc)
    print('INFO: %i discourse units.' % educ)
    print('INFO: %i connectors.' % cn)
    print('INFO: %i connector tokens.' % ct)
    

def getFileVersionsDict(connectorFiles, syntaxFiles, rstFiles, tokenFiles):

    d = defaultdict(lambda : defaultdict(str))
    for f in connectorFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['connectors'] = f
    for f in syntaxFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['syntax'] = f
    for f in rstFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['rst'] = f
    for f in tokenFiles:
        name = os.path.splitext(os.path.basename(f))[0]
        d[name]['tokens'] = f
        
    # sanity check:
    die = False
    for name in d:
        if not len(d[name]['connectors']):
            sys.stderr.write('ERROR: Could not find annotation layer <connectors> for %s.\n' % name)
            die = True
        if not len(d[name]['syntax']):
            sys.stderr.write('ERROR: Could not find annotation layer <syntax> for %s.\n' % name)
            die = True
        if not len(d[name]['rst']):
            sys.stderr.write('ERROR: Could not find annotation layer <rst> for %s.\n' % name)
            die = True
    if die:
        sys.stderr.write('FATAL ERROR: Please investigate annotation layers.\n')
        sys.exit(1)

    return d

def debugPrint(file2tokens):

    for f in file2tokens:
        print(f)
        for dt in file2tokens[f]:
            tokenId = dt.tokenId
            token = dt.token
            intOrExt = '___'
            if hasattr(dt, 'intOrExt'): # intOrExt is only specified for EDUs (not for connectives)
                intOrExt = dt.intOrExt
            relation = '___'    
            if (hasattr(dt, 'relation')): # relation is only specified for connectives
                relation = dt.relation
            sType = dt.segmentType
            lemma = dt.lemma
            pos = dt.pos
            rstParent = dt.rstParent
            rstRelation = dt.relname
            
            print('\t%s %s %s %s %s %s %s %s' % (str(tokenId), token, sType, pos, intOrExt, relation, lemma, rstRelation))
            

def printConnectiveStats(file2tokens):

    singleWordConnectives = defaultdict(int)
    multiWordConnectives = defaultdict(int)

    mwentrydict = defaultdict(list)
    
    for f in file2tokens:
        counted = []
        for i, dt in enumerate(file2tokens[f]):
            if dt.isConnective:
                if not i in counted:
                    if not dt.isMultiWord:
                        singleWordConnectives[dt.token] += 1
                    else:
                        tokens = [dt.token]
                        counted.append(i)
                        for j in range(i+1, len(file2tokens[f])):
                            if file2tokens[f][j].isConnective:
                                tokens.append(file2tokens[f][j].token)
                                counted.append(j)
                            else:
                                break
                        #print('multieowrd word:', tokens)
                        multiWordConnectives[' '.join(tokens)] += 1
                        # think there are only 2 word conns in there (no longer), so my entry dict works:
                        mwentrydict[tokens[0]].append(tokens[1])
                        
    connpostags = defaultdict(lambda : defaultdict(float))
    nonconnpostags = defaultdict(lambda : defaultdict(float))
    swcs = 0

    nonambiguous = set()
    ambiguous = set()
    ambigsyncats = set() # doing this for single and multi word combined
    nonambigsyncats = set()
    word2poscats = defaultdict(set)
    totalswcnc = 0
    totalswcc = 0
    for swc in singleWordConnectives:
        nonconnreading = 0
        connreading = 0
    
        for f in file2tokens:
            for dt in file2tokens[f]:
                word2poscats[dt.token].add(dt.pos)
                if dt.token == swc:
                   if dt.isConnective:
                       connreading += 1
                       totalswcc += 1
                       swcs += 1
                   else:
                       totalswcnc += 1
                       nonconnreading += 1
                # TODO: include pos tag, lemma, rst-relation etc here
                
        #print('Single Word Connective:', swc)
        #print('\tconnective ratio: %i of %i' % (connreading, connreading + nonconnreading))
        if nonconnreading == 0:
            nonambiguous.add(swc)
        else:
            ambiguous.add(swc)
        #print('\n')

    totalmwcc = 0
    totalmwcnc = 0
    mwnonambiguous = set()
    mwambiguous = set()
    for mwc in multiWordConnectives:
        ncr = 0
        cr = 0
        for f in file2tokens:
            for i, dt in enumerate(file2tokens[f]):
                if dt.token == mwc.split()[0] and file2tokens[f][i+1] == mwc.split()[1]:
                    if dt.isConnective:
                        cr += 1
                        totalmwcc += 1
                    else:
                        ncr += 1
                        totalmwcnc += 1
        if ncr == 0:
            mwnonambiguous.add(mwc)
        else:
            mwambiguous.add(mwc)
    print('%i single word connectives' % len(singleWordConnectives))
    print('of which %i are ambiguous' % len(ambiguous))
    print('%i multi word connectives:' % len(multiWordConnectives))
    print('of which %i are ambiguous:' % len(mwambiguous))

    singleWordSynCats = set()
    swambigsyncats = set()
    multiWordSynCats = set()
    for swc in singleWordConnectives:
        print(swc)
        for poscat in word2poscats[swc]:
            singleWordSynCats.add(poscat)
            if swc in ambiguous:
                swambigsyncats.add(poscat)
    for mwc in multiWordConnectives:
        for word in mwc.split():
            for poscat in word2poscats[word]:
                multiWordSynCats.add(poscat)

    print('single word syncats:', singleWordSynCats)
    print('single word ambiguous syncats:', swambigsyncats)
    print('multi word syncats:', multiWordSynCats)
    print('combined:', set(list(singleWordSynCats) + list(multiWordSynCats)))
    print('len:', len(set(list(singleWordSynCats) + list(multiWordSynCats))))
    print('len ambig ones:', len(swambigsyncats))

    print('total connective readings:', totalswcc+totalmwcc)
    print('total nonconnective readings:', totalswcnc+totalmwcnc)
    
    totaltokens = 0
    connectiveTokens = 0
    for f in file2tokens:
        for dt in file2tokens[f]:
            totaltokens += 1
            if dt.isConnective:
                connectiveTokens += 1
    print(totaltokens, connectiveTokens)

    
            
if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('-c', '--connectorsFolder', dest='connectorsFolder', help='specify PCC connectors folder')
    parser.add_option('-s', '--syntaxFolder', dest='syntaxFolder', help='specify PCC syntax folder')
    parser.add_option('-r', '--rstFolder', dest='rstFolder', help='specify PCC RST folder')
    parser.add_option('-t', '--tokensFolder', dest='tokensFolder', help='specify PCC bare tokens folder')

    
    options, args = parser.parse_args()

    if not options.connectorsFolder or not options.syntaxFolder or not options.rstFolder:
        parser.print_help(sys.stderr)
        sys.exit(1)

    connectorfiles = getInputfiles(options.connectorsFolder)
    syntaxfiles = getInputfiles(options.syntaxFolder)
    rstfiles = getInputfiles(options.rstFolder)
    tokenfiles = getInputfiles(options.tokensFolder)
    
    fileversions = getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles) # makes a dict with filename 2 connectors, syntax, etc.
    
    file2tokens = defaultdict(list)
    # this is for getting statistics on extArg position relative to the connective
    sameSentenceCases = 0
    anyOfTheFollowingSentencesCases = 0
    previousSentenceCases = 0
    anyOfThePrePreviousSentenceCases = 0
    
    for name in fileversions:
        tokenlist = parseConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = parseTokenizedFile(fileversions[name]['tokens'], tokenlist) # to add paragraph info
        file2tokens[name] = tokenlist
        #if re.search('maz-10902', name):
            #for token in tokenlist:
                #print(token.token, token.paragraphId, token.positionInParagraph, token.paragraphInitial)
        rid2connsentid = defaultdict(set) # set, because it can be spread over multiple sentences
        rid2extargsentid = defaultdict(set) # same here
        for token in tokenlist:
            if token.segmentType == 'connective':
                rid2connsentid[token.unitId].add(token.sentenceId)
            elif token.segmentType == 'unit':
                if token.intOrExt == 'ext':
                    rid2extargsentid[token.unitId].add(token.sentenceId)
            for rid in token.embeddedUnits:
                if token.embeddedUnits[rid] == 'ext':
                    rid2extargsentid[rid].add(token.sentenceId)

        for rid in rid2connsentid:
            if sorted(rid2connsentid[rid])[0] == sorted(rid2extargsentid[rid])[0]:
                sameSentenceCases += 1
            elif sorted(rid2extargsentid[rid])[0] - sorted(rid2connsentid[rid])[0] > 0:
                anyOfTheFollowingSentencesCases += 1
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] == 1:
                previousSentenceCases += 1
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] > 1:
                anyOfThePrePreviousSentenceCases += 1

    total = sameSentenceCases + anyOfTheFollowingSentencesCases + previousSentenceCases + anyOfThePrePreviousSentenceCases
    
    print('ss cases:%i / %f' % (sameSentenceCases, sameSentenceCases/total))
    print('fs cases:%i / %f' % (anyOfTheFollowingSentencesCases, anyOfTheFollowingSentencesCases/total))
    print('ps cases:%i / %f' % (previousSentenceCases, previousSentenceCases/total))
    print('pre-ps cases:%i / %f' % (anyOfThePrePreviousSentenceCases, anyOfThePrePreviousSentenceCases/total))

    

        #if re.search('maz-00001', name):
            #for token in tokenlist:
                #print(token.token, token.sentenceId)
            #print('conn positions:', rid2connsentid)
            #print('exta positions:', rid2extargsentid)
    
