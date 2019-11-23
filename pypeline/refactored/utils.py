import os, sys, re


def getInputfiles(infolder):
    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist


# the NLTK parser crashed on round brackets.
def filterTokens(tokens):
    skipSet = ["(", ")"]
    return [t for t in tokens if not t in skipSet]


def get_parent(pt, i):

    nodePosition = pt.leaf_treeposition(i)
    parent = pt[nodePosition[:-1]].parent()
    return parent


def find_lowest_embracing_node(node, reftoken):

    if not contains_sublist(node.leaves(), list(reftoken)):
        while node.parent():  # recurse till rootnode
            return find_lowest_embracing_node(node.parent(), reftoken)
    return node


def find_lowest_embracing_node_discont(node, reftoken):

    if not contains_discont_sublist(node.leaves(), list(reftoken)):
        while node.parent():  # recurse till rootnode
            return find_lowest_embracing_node(node.parent(), reftoken)
    return node


def addAnnotationLayerToDict(flist, fdict, annname):
    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict


def getDataSplits(numIterations, dataSize):
    p = int(dataSize / 10)
    pl = [int(x) for x in range(0, dataSize, p)]
    pl.append(int(dataSize))
    return pl


def getPathToRoot(ptree, route):
    if ptree.parent() == None:
        route.append(ptree.label())
        return route
    else:
        route.append(ptree.label())
        getPathToRoot(ptree.parent(), route)
    return route


def compressRoute(r):  # filtering out adjacent identical tags
    delVal = "__DELETE__"
    for i in range(len(r) - 1):
        if r[i] == r[i + 1]:
            r[i + 1] = delVal
    return [x for x in r if x != delVal]


def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i : i + n]) for i in range(len(lst) - n + 1))


def contains_discont_sublist(lst, sublst):

    stripped = [x for x in lst if x in sublst]
    if compressRoute(stripped) == sublst:
        return True


def get_match_positions(lst, sublst):
    return [i for i, j in enumerate(lst) if lst[i : i + len(sublst)] == sublst]


def get_discont_match_positions(lst, sublst):
    # just taking first index. Might be wrong.
    positions = []
    for item in sublst:
        positions.append(lst.index(item))
    return positions


def replaceBrackets(s):
    s = re.sub("\)", "]", re.sub("\(", "[", s))
    s = re.sub("`", "'", s)
    return s


def narrowMatches(matches, pcct, pccTokens, index):
    rightmatches = [
        x for x in matches if x[5] == pcct.token + "_" + pccTokens[index + 1].token
    ]
    if len(rightmatches) > 1:
        leftrightmatches = [
            x
            for x in rightmatches
            if x[2] == pccTokens[index - 1].token + "_" + pcct.token
        ]
        if len(leftrightmatches) > 1:

            print("debugging leftrightmatches:", leftrightmatches)
            print("matches:", matches)
            print("token:", pcct.token)
            print("idex:", index)
            print("sent:", pcct.fullSentence)
            # sys.stderr.write('FATAL ERROR: Dying due to non-unique matches...\n')
            # sys.exit(1)
            return False
        elif len(leftrightmatches) == 1:
            return leftrightmatches
        else:
            # sys.stderr.write('FATAL ERROR: Could not find tree match at all...\n')
            # sys.exit(1)
            return False
    else:
        return rightmatches


def mergePhrasalConnectives(l):
    l2 = []
    for t in l:
        nt = (t[0], [t[1]])
        l2.append(nt)
    l = l2
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i][0] == l[j][0]:
                if l[i][1][-1] == l[j][1][0] - 1:
                    l[i] = (l[i][0], sorted(l[i][1] + l[j][1]))
    ri = set()
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i][0] == l[j][0]:
                if set(l[j][1]).issubset(set(l[i][1])):
                    ri.add(j)
    l2 = []
    for i in range(len(l)):
        if not i in ri:
            l2.append(l[i])
    l = l2
    return l


def getPostagFromTree(ptree, tokenIndex):
    # had I known it was this simple, wouldn't have to dedicate a function to it...
    return ptree.pos()[tokenIndex][1]


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]
