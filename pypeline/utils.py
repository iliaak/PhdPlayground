import os

def getInputfiles(infolder):
    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

# the NLTK parser crashed on round brackets.
def filterTokens(tokens):
    skipSet = ['(', ')']
    return [t for t in tokens if not t in skipSet]

def addAnnotationLayerToDict(flist, fdict, annname):
    for f in flist:
        basename = os.path.basename(f)
        fdict[basename][annname] = f
    return fdict

def getPathToRoot(ptree, route):
    if ptree.parent() == None:
        route.append(ptree.label())
        return route
    else:
        route.append(ptree.label())
        getPathToRoot(ptree.parent(), route)
    return route

def compressRoute(r): # filtering out adjacent identical tags
    delVal = "__DELETE__"
    for i in range(len(r)-1):
        if r[i] == r[i+1]:
            r[i+1] = delVal
    return [x for x in r if x != delVal]

def narrowMatches(matches, pcct, pccTokens, index):
    rightmatches = [x for x in matches if x[5] == pcct.token + '_' + pccTokens[index+1].token]
    if len(rightmatches) > 1:
        leftrightmatches = [x for x in rightmatches if x[2] == pccTokens[index-1].token + '_' + pcct.token]
        if len(leftrightmatches) > 1:
            sys.stderr.write('FATAL ERROR: Dying due to non-unique matches...\n')
            sys.exit(1)
        elif len(leftrightmatches) == 1:
            return leftrightmatches
        else:
            sys.stderr.write('FATAL ERROR: Could not find tree match at all...\n')
            sys.exit(1)
    else:
        return rightmatches

def mergePhrasalConnectives(l):
    l2 = []
    for t in l:
        nt = (t[0], [t[1]])
        l2.append(nt)
    l = l2
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i][0] == l[j][0]:
                if l[i][1][-1] == l[j][1][0]-1:
                    l[i] = (l[i][0], sorted(l[i][1] + l[j][1]))
    ri = set()
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i][0] == l[j][0]:
                if set(l[j][1]).issubset(set(l[i][1])):
                    ri.add(j)
    l2 = []
    for i in range(len(l)):
        if not i in ri:
            l2.append(l[i])
    l = l2
    return l
