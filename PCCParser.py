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

discourseRelations = {}

class DiscourseRelation(object):

    def __init__(self, relationId):
        self.relationId = relationId
        self.connectiveTokens = []
        self.intArgTokens = []
        self.extArgTokens = []
        self.sense = None
        self.intArgSynLabel = None # defaulting to None for this one case in maz-8361 (id 8) without int arg
        self.intArgSynLabelIsExactMatch = None
        self.intArgSpanningNodeText = None

    def addConnectiveToken(self, tid):
        self.connectiveTokens.append(tid)
    def addIntArgToken(self, tid):
        self.intArgTokens.append(tid)
    def addExtArgToken(self, tid):
        self.extArgTokens.append(tid)
    def setSense(self, sense):
        self.sense = sense
    def addExtArgSynfo(self, label, hasfullcoverage, spanningNodeText):
        self.extArgSynLabel = label
        self.extArgSynLabelIsExactMatch = hasfullcoverage
        self.extArgSpanningNodeText = spanningNodeText
    def addIntArgSynfo(self, label, hasfullcoverage, spanningNodeText):
        self.intArgSynLabel = label
        self.intArgSynLabelIsExactMatch = hasfullcoverage
        self.intArgSpanningNodeText = spanningNodeText

    def filterIntArgForConnectiveTokens(self):
        self.intArgTokens = [x for x in self.intArgTokens if not x in self.connectiveTokens]

        
        
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
    def setSentencePosition(self, val):
        self.sentencePosition = val
    def setSyntaxSentenceId(self, val):
        self.syntaxSentenceId = val

def flattentree(edid, tempdict, l, ttids):

    for i in tempdict[edid]:
        if i in ttids:
            l.append(i)
        else:
            flattentree(i, tempdict, l, ttids)
    return l

def getLeafsFromGoldTree(nodeId, subdict, terminalIds, output):

    for child in subdict[nodeId]:
        if child in terminalIds:
            output.append(child)
        else:
            output = getLeafsFromGoldTree(child, subdict, terminalIds, output)
    return sorted(output, key = lambda x: int(re.sub('^.*_', '', x)))
    
        
def addArgumentLabelInfo(discourseRelations, syntaxfile, tid2dt):

    sid2nonterminals = defaultdict(lambda : defaultdict(str))
    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8', remove_comments = True)
    tree = lxml.etree.parse(syntaxfile, parser = xmlParser)
    sentencedict = defaultdict(str)
    syntaxTokenId2postag = defaultdict(str)
    syntaxTokenId2word = defaultdict(str)
    for body in tree.getroot():
        for elemid, sentence in enumerate(body):
            #elemid is sentenceId
            graph = sentence.getchildren()[0]
            terminals = graph.find('.//terminals')
            nonterminals = graph.find('.//nonterminals')
            #dt.syntaxId is terminal token id
            terminaltokenIds = [t.get('id') for t in terminals]
            for t in terminals:
                syntaxTokenId2postag[t.get('id')] = t.get('pos')
                syntaxTokenId2word[t.get('id')] = t.get('word')
            nt2terminalIds = defaultdict(list)
            nt2label = defaultdict(str)
            tempdict = defaultdict(list)

            for nt in nonterminals: # result of graph.find may not be ordered
                for edge in nt:
                    if edge.tag == 'edge': # discarding sec edges
                        tempdict[nt.get('id')].append(edge.get('idref'))
            for nt in nonterminals:
                flattened = flattentree(nt.get('id'), tempdict, [], terminaltokenIds)
                sid2nonterminals[elemid][tuple(flattened)] = nt.get('cat')
    """
    if re.search('maz-00001', syntaxfile):
        for sid, nts in sid2nonterminals.items():
            print('sid and nts:', sid, nts)
    """
    for dr in discourseRelations:
        intArgSyntaxTokenIds = [tid2dt[t].syntaxId for t in dr.intArgTokens if not syntaxTokenId2postag[tid2dt[t].syntaxId].startswith('$')]
        intArgSentIds = set([tid2dt[t].sentenceId for t in dr.intArgTokens])
        if intArgSentIds:
            connectiveTokenSyntaxIds = [tid2dt[t].syntaxId for t in dr.connectiveTokens]
            intlabel, intlabelhasfullcoverage, intsynnodetokens = getArgSyntaxLabel(connectiveTokenSyntaxIds, intArgSyntaxTokenIds, intArgSentIds, sid2nonterminals, syntaxTokenId2postag, syntaxTokenId2word)
            dr.addIntArgSynfo(intlabel, intlabelhasfullcoverage, intsynnodetokens)
        else:
            sys.stderr.write('WARNING: Skipping intArg shape info for %s in %s\n' % (dr.relationId, syntaxfile))
            
        extArgSyntaxTokenIds = [tid2dt[t].syntaxId for t in dr.extArgTokens if not syntaxTokenId2postag[tid2dt[t].syntaxId].startswith('$')]
        extArgSentIds = set([tid2dt[t].sentenceId for t in dr.extArgTokens])
        extlabel, extlabelhasfullcoverage, extsynnodetokens = getArgSyntaxLabel(None, extArgSyntaxTokenIds, extArgSentIds, sid2nonterminals, syntaxTokenId2postag, syntaxTokenId2word)
        dr.addExtArgSynfo(extlabel, extlabelhasfullcoverage, extsynnodetokens)
        
        
        
def getArgSyntaxLabel(connectiveTokenSyntaxIds, argsyntaxids, argsentids, sid2nonterminals, syntaxTokenId2postag, syntaxTokenId2word):

    label = None
    hasfullcoverage = False
    nodetokens = []
    if len(argsentids) > 1:
        label = 'multi_sentence'
        nodetokens.append('multi_sentence. TODO.')
    else:
        matches = []
        for nt, ids in sid2nonterminals[list(argsentids)[0]].items():
            if set(argsyntaxids).issubset(nt):
                matches.append(nt)

        if matches:
            sm = matches[0]
            if len(matches) > 1: # if there are multiple matches, get the shortest one
                shortestmatchlength = max([len(x) for x in matches])
                for m2 in matches:
                    if len(m2) < shortestmatchlength:
                        sm = m2
            hasfullcoverage = hasFullCoverage(sm, argsyntaxids, syntaxTokenId2postag, connectiveTokenSyntaxIds)
            label = sid2nonterminals[list(argsentids)[0]][sm]
            sortedtokens = sorted(sm, key = lambda x: int(re.sub('s\d+_', '', x)))
            nodetokens = ' '.join([syntaxTokenId2word[x] for x in sortedtokens])
        else:
            pass# there was one caase (sent s1040 in maz-17953) where s1040_14 and s1040_15 ('zu tun') are not included in the root S node. Think this is an error though and not meant to be. 
            #print('NO MATCH FOUND AT ALL! INVESTIGATE!')
            #print(syntaxfile)
            #print(intArgSyntaxTokenIds)
            #print(intArgSentIds)
            #print(sid2nonterminals[list(intArgSentIds)[0]])
            
    return label, hasfullcoverage, nodetokens
                
def hasFullCoverage(m, arg, refdict, conntokensynids):

    if len(m) == len(arg):
        return True
    elif len(arg) > len(m):
        extraTokens = [x for x in arg if not x in m]
        if all(refdict[x].startswith('$') for x in extraTokens):
            return True
    elif len(arg) < len(m):
        # this is a bit implicit, but conntokensynids is None when we are dealing with extArg (which is correct)
        if not conntokensynids == None:
            extraTokens = [x for x in m if not x in arg]
            if all(x in conntokensynids for x in extraTokens):
                return True
            
    return False
                
        
def addAnnotationLayerToDict(flist, fdict, annname):

    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict

def parseStandoffConnectorFile(connectorxml):

    xmlParser = lxml.etree.XMLParser(strip_cdata = False, resolve_entities = False, encoding = 'utf-8')
    tree = lxml.etree.parse(connectorxml, parser = xmlParser)
    tokenlist = []
    discourseRelations = []
    tid2dt = {} # so I can assign info in conn, sense, args later
    #tid2segmentType = defaultdict(str)
    for node in tree.getroot():
        if node.tag == 'tokens':
            for subnode in node:
                dt = DiscourseToken(subnode.get('id'), subnode.text)
                dt.setConnectiveBoolean(False) # setting isConnective to False always, only set it to True next if it is a conn... (this is easier for other scripts using PCCParser, so at least this attr is always there (preventing an AttributeError)
                tokenlist.append(dt)
                tid2dt[subnode.get('id')] = dt
        elif node.tag == 'relations':
            for subnode in node:
                dr = DiscourseRelation(subnode.get('relation_id'))
                dr.setSense(subnode.get('sense'))
                for elem in subnode:
                    if elem.tag == 'connective_tokens':
                        for ct in elem:
                            #tid2segmentType[ct.get('id')] = 'connective'
                            dr.addConnectiveToken(ct.get('id'))
                            tid2dt[ct.get('id')].setConnectiveBoolean(True)
                    if elem.tag == 'int_arg_tokens':
                        for iat in elem:
                            #tid2segmentType[iat.get('id')] = 'unit'
                            dr.addIntArgToken(iat.get('id'))
                            tid2dt[iat.get('id')].setIntOrExt('int')
                            tid2dt[iat.get('id')].setIsIntArg(True) # these and the above are redundant (only need one), check later during refacotoring which one can go
                    if elem.tag == 'ext_arg_tokens':
                        for eat in elem:
                            #tid2segmentType[eat.get('id')] = 'unit'
                            dr.addExtArgToken(eat.get('id'))
                            tid2dt[eat.get('id')].setIntOrExt('ext')
                            tid2dt[eat.get('id')].setIsExtArg(True) # these and the above are redundant (only need one), check later during refacotoring which one can go
                dr.filterIntArgForConnectiveTokens()
                discourseRelations.append(dr)
    """
    # adding segmentTypes, as it is used in downstream stuff
    for dt in tokenlist:
        dt.setType(tid2segmentType[dt.tokenId])
    """
                
    return tokenlist, discourseRelations, tid2dt

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
            #for sentenceId, t in enumerate(terminalsNode):
            for sentencePosition, t in enumerate(terminalsNode):
                sToken = t.get('word')
                dt = tokenlist[syntaxTokenId]
                dt.setSyntaxSentenceId(sid)
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
                dt.setSentencePosition(sentencePosition)
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
    nid2 = node.get('id')
    if not nid2 and conn:
        # accomodating for modifiers over connectives
        nid2 = node.getparent().get('id')
    dt.setUnitId(nid2)
    if not nid2 in discourseRelations:
        dr = DiscourseRelation(nid2)
        discourseRelations[nid2] = dr
            
        
    embeddedUnits = getEmbeddedUnits(node, {})
    for k in embeddedUnits:
        t = embeddedUnits[k]
        if not k in discourseRelations:
            dr = DiscourseRelation(k)
            discourseRelations[k] = dr
        if t == 'int':
            discourseRelations[k].addIntArgToken(dt)
        elif t == 'ext':
            discourseRelations[k].addExtArgToken(dt)

            
    dt.setEmbeddedUnits(embeddedUnits)
    #dt.setEmbedded()
    if segmentType == 'connective':
        # accomodating for modifiers over connectives
        nid = None
        if node.get('id'):
            nid = node.get('id')
        else:
            nid = node.getparent().get('id')
        dt.setConnectiveId(nid)
        discourseRelations[nid].addConnectiveToken(dt)
    if segmentType == 'unit':
        pn = node.getparent()
        segmentType = node.get('type')
        if not segmentType: # this elaborate stuff is for tail cases, in which case lxml returns None for node.get('type')
            pn = node.getparent()
            if pn is not None and pn.tag == 'unit':
                segmentType = node.getparent().get('type')
        dt.setIntOrExt(segmentType)
        if segmentType == 'int':
            discourseRelations[node.get('id')].addIntArgToken(dt)
        elif segmentType == 'ext':
            discourseRelations[node.get('id')].addExtArgToken(dt)
        
        
    elif segmentType == 'connective':
        
        dt.setRelation(node.get('relation'))
    dt.setMultiWordBoolean(multiWordBool)
    dt.setConnectiveBoolean(conn) # a bit redundant, because the type also contains this info, but this is easier for if checks...
    connectorTokenId += 1
    return dt
    

def extractTokens(node, l):

    conn = False
    if node.text:
        if node.tag == 'connective' or node.tag == 'modifier': # flattening modifiers and connectives for now.
            conn = True
        textContent = node.text.strip()
        if textContent:
            tokens = textContent.split()
            multiWordBool = False
            if len(tokens) > 1:
                multiWordBool = True
            for token in tokens:
                dt = createDiscourseToken(token, node, conn, multiWordBool)
                if conn:
                    dt.sense = node.get('relation')
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


def getTokenId2DiscourseToken(tokenlist):

    tid2dt = {}
    for dt in tokenlist:
        tid2dt[dt.tokenId] = dt
    return tid2dt

def getArgumentPositionInfo():

    embeddedcases = 0
    extarg_intarg_nonadjacent = 0
    file2tokens = defaultdict(list)
    sensedict = defaultdict(int)
    file2discourseRelations = defaultdict(dict)
    allrelations = 0
    for name in fileversions:
        tokenlist, discourseRelations, tid2dt = parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = parseTokenizedFile(fileversions[name]['tokens'], tokenlist) # to add paragraph info
        file2tokens[name] = tokenlist
        file2discourseRelations[name] = discourseRelations
        tid2dt = getTokenId2DiscourseToken(tokenlist)

        for dr in file2discourseRelations[name]:
            allrelations += 1
            connsents = set()# plural, because can be multiple
            for ctid in dr.connectiveTokens:
                connsents.add(tid2dt[ctid].sentenceId)
            intargsents = set()
            for iatid in dr.intArgTokens:
                intargsents.add(tid2dt[iatid].sentenceId)
            extargsents = set()
            for eatid in dr.extArgTokens:
                extargsents.add(tid2dt[eatid].sentenceId)
        
            #print('debug:', dr.extArgTokens)
            if not set(dr.extArgTokens).isdisjoint(set(dr.intArgTokens)):
                # TODO: think about what to do with these (not sure if this difference between embedded or not is even relevant here)
                embeddedcases += 1
            else:
                if min(int(dr.connectiveTokens[0]), int(dr.intArgTokens[0])) - int(dr.extArgTokens[len(dr.extArgTokens)-1]) > 1:
                    # they are non-adjacent, but it could be a punctuation symbol in between (or a connective from another relation, but not sure if I want to discard these at this point)
                    inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))]).strip()
                    if re.match('\W+', inbetweenstuff):
                        pass
                    else:
                        extarg_intarg_nonadjacent += 1
                        sensedict[dr.sense] += 1
                        print('in between:', inbetweenstuff)
                        print()
                # FS non-adjacent does not happen in PCC (code below)
                """
                elif min(int(dr.connectiveTokens[0]), int(dr.extArgTokens[0])) - int(dr.intArgTokens[len(dr.intArgTokens)-1]) > 1:
                    inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.intArgTokens[len(dr.intArgTokens)-1])+1, int(dr.extArgTokens[0]))]).strip()
                    if re.match('\W+', inbetweenstuff):
                        pass
                    else:
                        extarg_intarg_nonadjacent += 1
                    
                        print('in between:', inbetweenstuff)
                        print()
                """
        
            
    print('extarg_intarg_nonadjacent:', extarg_intarg_nonadjacent, extarg_intarg_nonadjacent / allrelations)
    print('all relations:', allrelations)
    print('embedded cases:', embeddedcases)
    for pair in sorted(sensedict.items(), key = lambda x: x[1], reverse=True):
        print(pair)
                
def getArgumentPositionInfoSentenceIdBased():

    file2tokens = defaultdict(list)
    file2discourseRelations = defaultdict(dict)
    # this is for getting statistics on extArg position relative to the connective
    sameSentenceCases = 0
    anyOfTheFollowingSentencesCases = 0
    previousSentenceCases = 0
    anyOfThePrePreviousSentenceCases = 0
    intargembeddedinextarg = 0
    ext_int_overlap = 0
    #int_ext_overlap = 0
    ext_circumfixes_int = 0
    sensedict = defaultdict(int)
    allrelations = 0
    
    for name in fileversions:
        #tokenlist = parseConnectorFile(fileversions[name]['connectors'])
        tokenlist, discourseRelations, tid2dt = parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = parseTokenizedFile(fileversions[name]['tokens'], tokenlist) # to add paragraph info
        file2tokens[name] = tokenlist
        addArgumentLabelInfo(discourseRelations, fileversions[name]['syntax'])
        file2discourseRelations[name] = discourseRelations
        # create dict so that tid accesses DiscourseToken. Have already done this in parseStandoffConnectorFile, so could pass it on from there. Let's see how this code evolves and which one needs to go later on.
        tid2dt = getTokenId2DiscourseToken(tokenlist)
        #if re.search('maz-10902', name):
            #for token in tokenlist:
                #print(token.token, token.sentencePosition, token.paragraphId, token.positionInParagraph, token.paragraphInitial)

        # can delete the stuff below if my new standoff parsing works
        anaphoricOnes = []
        for dr in file2discourseRelations[name]:
            allrelations += 1
            connsents = set()# plural, because can be multiple
            for ctid in dr.connectiveTokens:
                connsents.add(tid2dt[ctid].sentenceId)
            intargsents = set()
            for iatid in dr.intArgTokens:
                intargsents.add(tid2dt[iatid].sentenceId)
            extargsents = set()
            for eatid in dr.extArgTokens:
                extargsents.add(tid2dt[eatid].sentenceId)
            if sorted(intargsents) == sorted(extargsents):
               sameSentenceCases += 1
               #print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
               #print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
               #print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
               #inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))])
               #print('in between:', inbetweenstuff)
               #print()
            elif sorted(extargsents)[0] - sorted(intargsents)[len(intargsents)-1] == 1:
                anyOfTheFollowingSentencesCases += 1
                #print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
                #print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
                #print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
                #inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.intArgTokens[len(dr.intArgTokens)-1])+1, int(dr.extArgTokens[0]))])
                #print('in between:', inbetweenstuff)
                #print()
            elif sorted(intargsents)[0] - sorted(extargsents)[len(extargsents)-1] == 1:
                previousSentenceCases += 1
                #print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
                #print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
                #print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
                #inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))])
                #print('in between:', inbetweenstuff)
                #print()
            elif sorted(intargsents)[0] - sorted(extargsents)[len(extargsents)-1] >1 :
                anyOfThePrePreviousSentenceCases += 1
                print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
                print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
                print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
                inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))])
                print('in between:', inbetweenstuff)
                print()
                sensedict[dr.sense] += 1
            elif intargsents.issubset(extargsents):
                intargembeddedinextarg += 1
                #print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
                #print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
                #print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
                #inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))])
                #print('in between:', inbetweenstuff)
                #print()
            elif sorted(intargsents)[0] == sorted(extargsents)[0] and len(intargsents) > len(extargsents):
                ext_int_overlap += 1
            elif sorted(extargsents)[0] < sorted(intargsents)[0] and sorted(extargsents)[len(extargsents)-1] > sorted(intargsents)[len(intargsents)-1]:
                ext_circumfixes_int += 1
            elif sorted(intargsents)[0] == sorted(extargsents)[len(extargsents)-1] and sum(intargsents) > sum(extargsents):
                ext_int_overlap += 1
                
                
            else:
                print('LOST CASE, DEBUG:')
                print('int sent ids:', sorted(intargsents))
                print('ext sent ids:', sorted(extargsents))
                print('conn:', ' '.join([tid2dt[x].token for x in dr.connectiveTokens]))
                print('int:', ' '.join([tid2dt[x].token for x in dr.intArgTokens]))
                print('ext:', ' '.join([tid2dt[x].token for x in dr.extArgTokens]))
                inbetweenstuff = ' '.join([tid2dt[str(x)].token for x in range(int(dr.extArgTokens[len(dr.extArgTokens)-1])+1, int(dr.intArgTokens[0]))])
                print('in between:', inbetweenstuff)
                print()
    print('total number of relations:', allrelations)
    print('ss cases: %i / %f' % (sameSentenceCases, sameSentenceCases/allrelations))
    print('fs cases: %i / %f' % (anyOfTheFollowingSentencesCases, anyOfTheFollowingSentencesCases/allrelations))
    print('ps cases: %i / %f' % (previousSentenceCases, previousSentenceCases/allrelations))
    print('pre-ps cases: %i / %f' % (anyOfThePrePreviousSentenceCases, anyOfThePrePreviousSentenceCases/allrelations))
    print('int arg embedded in ext arg: %i / %f' % (intargembeddedinextarg, intargembeddedinextarg/allrelations))
    print('ext, then int, with overlapping sent ids: %i / %f' % (ext_int_overlap, ext_int_overlap/allrelations))
    print('ext circumfixes int, but int not included in ext (no embedding for sent ids): %i / %f' % (ext_circumfixes_int, ext_circumfixes_int/allrelations))
    print('mystery cases remaining:', allrelations - sameSentenceCases - anyOfTheFollowingSentencesCases - previousSentenceCases - anyOfThePrePreviousSentenceCases - intargembeddedinextarg - ext_int_overlap - ext_circumfixes_int)
    print('senses of pre-previous sent cases:')
    for pair in sorted(sensedict.items(), key = lambda x: x[1], reverse=True):
        print(pair)


def getArgumentShapeInfo(f2drs, f2tid2dt):

    intArgShapes = defaultdict(int)
    intArgLabel2Coverage = defaultdict(lambda : defaultdict(int))
    extArgShapes = defaultdict(int)
    extArgLabel2Coverage = defaultdict(lambda : defaultdict(int))
    intArgSyntacticTypesInfo = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    extArgSyntacticTypesInfo = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    import json
    connectiveSyncatJsonDict = json.load(codecs.open('pccConnectiveSyncats.json'))

    for f, drs in f2drs.items():
        for dr in drs:
            conntext = ' '.join([f2tid2dt[f][cid].token for cid in dr.connectiveTokens])
            connsyncats = connectiveSyncatJsonDict[conntext]
            if type(connsyncats) == str:
                connsyncats = [connsyncats]
            intArgShapes[dr.intArgSynLabel] += 1
            if dr.intArgSynLabelIsExactMatch:
                intArgLabel2Coverage[dr.intArgSynLabel]['exact match'] += 1
                #intArgSyntacticTypesInfo[connpostags][dr.intArgSynLabel]['exact match'] += 1
                for cs in connsyncats:
                    intArgSyntacticTypesInfo[cs][dr.intArgSynLabel]['exact match'] += 1
            else:
                intArgLabel2Coverage[dr.intArgSynLabel]['syntax label is superset'] += 1
                for cs in connsyncats:
                    intArgSyntacticTypesInfo[cs][dr.intArgSynLabel]['superset match'] += 1
                #intArgSyntacticTypesInfo[connpostags][dr.intArgSynLabel]['superset match'] += 1
                print('True shape/tokens:', ' '.join([f2tid2dt[f][x].token for x in dr.intArgTokens]))
                print('label:', dr.intArgSynLabel)
                print('connective syncat(s):', connsyncats)
                print('connective tokens:', ' '.join([f2tid2dt[f][x].token for x in dr.connectiveTokens]))
                print('Superset shape/tokens:', dr.intArgSpanningNodeText)
                print()
                # investigate here. Then see if the conn syn type (adv, prep, etc) exhibits some special properties (perhaps csu/cco are always sentential, or something similar?)
            extArgShapes[dr.extArgSynLabel] += 1
            if dr.extArgSynLabelIsExactMatch:
                extArgLabel2Coverage[dr.extArgSynLabel]['exact match'] += 1
                for cs in connsyncats:
                    extArgSyntacticTypesInfo[cs][dr.extArgSynLabel]['exact match'] += 1
                #extArgSyntacticTypesInfo[connpostags][dr.extArgSynLabel]['exact match'] += 1
            else:
                extArgLabel2Coverage[dr.extArgSynLabel]['syntax label is superset'] += 1
                #extArgSyntacticTypesInfo[connpostags][dr.extArgSynLabel]['superset match'] += 1
                for cs in connsyncats:
                    extArgSyntacticTypesInfo[cs][dr.extArgSynLabel]['superset match'] += 1

    print('int arg labels:')
    for pair in sorted(intArgShapes.items(), key = lambda x: x[1], reverse=True):
        print("%s\t%s\t%s\t%s" % (pair[0], pair[1], intArgLabel2Coverage[pair[0]]['exact match'], intArgLabel2Coverage[pair[0]]['syntax label is superset']))
        
    for syntype in intArgSyntacticTypesInfo:
        #print(syntype)
        for arglabel in intArgSyntacticTypesInfo[syntype]:
            print('%s\t%s\t%s\t%s' % (syntype, arglabel, intArgSyntacticTypesInfo[syntype][arglabel]['exact match'], intArgSyntacticTypesInfo[syntype][arglabel]['superset match']))

    print('\next arg labels:')
    for pair in sorted(extArgShapes.items(), key = lambda x: x[1], reverse=True):
        print("%s\t%s\t%s\t%s" % (pair[0], pair[1], extArgLabel2Coverage[pair[0]]['exact match'], extArgLabel2Coverage[pair[0]]['syntax label is superset']))
        
    for syntype in extArgSyntacticTypesInfo:
        #print(syntype)
        for arglabel in extArgSyntacticTypesInfo[syntype]:
            print('%s\t%s\t%s\t%s' % (syntype, arglabel, extArgSyntacticTypesInfo[syntype][arglabel]['exact match'], extArgSyntacticTypesInfo[syntype][arglabel]['superset match']))

            
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

    file2tokens = defaultdict(list)
    file2discourseRelations = defaultdict(dict)
    fileversions = getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles) # makes a dict with filename 2 connectors, syntax, etc.
    file2tid2dt = defaultdict(lambda : defaultdict(DiscourseToken)) # only used for debugging/priting I think
    for name in fileversions:
        #tokenlist = parseConnectorFile(fileversions[name]['connectors'])
        tokenlist, discourseRelations, tid2dt = parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokens[name] = tokenlist
        tid2dt = getTokenId2DiscourseToken(tokenlist)
        file2tid2dt[name] = tid2dt
        addArgumentLabelInfo(discourseRelations, fileversions[name]['syntax'], tid2dt)
        file2discourseRelations[name] = discourseRelations

    
        
    #getArgumentPositionInfoSentenceIdBased()
    #getArgumentPositionInfo()
    getArgumentShapeInfo(file2discourseRelations, file2tid2dt)
            
    ############### debugging section #################
    """
    if re.search('maz-2669', name):
    for token in tokenlist:
    int_ext = None
    if token.segmentType == 'unit':
    int_ext = token.intOrExt
    print(token.token, token.tokenId, token.unitId, token.sentenceId, token.segmentType, int_ext)
    """
    
    ###################################################


    
        
