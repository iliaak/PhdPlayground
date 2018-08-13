import numpy as np
import keras
from keras.models import Sequential

from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.utils import to_categorical

from collections import defaultdict
import sys
import csv
import pickle
import PCCParser
import os
import codecs
import time

from nltk.parse import stanford
from nltk.tree import ParentedTree


JAVAHOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64'
STANFORD_PARSER='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
STANFORD_MODELS='/home/peter/phd/PhdPlayground/stanfordStuff/stanford-parser-full-2017-06-09'
os.environ['JAVAHOME'] = JAVAHOME
os.environ['STANFORD_PARSER'] = STANFORD_PARSER
os.environ['STANFORD_MODELS'] = STANFORD_MODELS

lexParserPath = 'edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz'
lexParser = stanford.StanfordParser(model_path=lexParserPath)


pos2column = {
    0:'token',
    1:'pos',
    2:'leftbigram',
    3:'leftpos',
    4:'leftposbigram',
    5:'rightbigram',
    6:'rightpos',
    7:'rightposbigram',
    8:'selfCategory',
    9:'parentCategory',
    10:'leftsiblingCategory',
    11:'rightsiblingCategory',
    12:'rightsiblingContainsVP',
    13:'pathToRoot',
    14:'compressedPath',
    15:'class_label'
}


def getDataSplits(numIterations, dataSize):

    p = int(dataSize / 10)
    pl = [int(x) for x in range(0, dataSize, p)]
    pl.append(int(dataSize))    
    return pl


def loadExternalEmbeddings(embfile):

    starttime = time.time()
    sys.stderr.write('INFO: Loading external embeddings from %s.\n' % embfile)
    ed = defaultdict()
    #"""
    with open(embfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            values = line.split()
            ed[values[0]] = np.array([float(x) for x in values[1:]])
    #"""
    endtime = time.time()
    sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))
    return ed

def filterTokens(tokens):
    skipSet = ['(', ')']
    return [t for t in tokens if not t in skipSet]

def getSingleWordFeaturesFromTree(index, tokens, token, parentedTree):

    treeVector = getVectorsForTree(parentedTree)
    matches = [x for x in treeVector if x[0] == token.token]
    if len(matches) > 1:
        matches = narrowMatches(matches, token, tokens, index)
        return matches[0]
    elif len(matches) == 1:
        return matches[0]

    
def narrowMatches(matches, token, tokens, index):

    rightmatches = [x for x in matches if x[5] == token.token + '_' + tokens[index+1].token]
    if len(rightmatches) > 1:
        leftrightmatches = [x for x in rightmatches if x[2] == tokens[index-1].token + '_' + token.token]
        if len(leftrightmatches) > 1:
            
            print('ERROR: Dying due to non-unique matches')
            sys.exit(1)
        elif len(leftrightmatches) == 1:
            return leftrightmatches
        else:
            print('Could not find match!')
            sys.exit(1)
    else:
        return rightmatches


def getVectorsForTree(tree):

    treeVectors = []
    parentedTree = ParentedTree.convert(tree)
    for i, node in enumerate(parentedTree.pos()):
        features = []
        currWord = node[0]
        currPos = node[1]
        features.append(currWord)
        features.append(currPos)
        ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
        rn = "EOS" if i == len(parentedTree.pos())-1 else parentedTree.pos()[i+1]
        lpos = "_" if ln == "SOS" else ln[1]
        rpos = "_" if rn == "EOS" else rn[1]
        lstr = ln if ln == "SOS" else ln[0]
        rstr = rn if rn == "EOS" else rn[0]
        lbigram = lstr + '_' + currWord
        rbigram = currWord + '_' + rstr
        lposbigram = lpos + '_' + currPos
        rposbigram = currPos + '_' + rpos
        features.append(lbigram)
        features.append(lpos)
        features.append(lposbigram)
        features.append(rbigram)
        features.append(rpos)
        features.append(rposbigram)

        selfcat = currPos # always POS for single words
        features.append(selfcat)
        nodePosition = parentedTree.leaf_treeposition(i)
        parent = parentedTree[nodePosition[:-1]].parent()
        parentCategory = parent.label()
        features.append(parentCategory)

        ls = parent.left_sibling()
        lsCat = False if not ls else ls.label()
        rs = parent.right_sibling()
        rsCat = False if not rs else rs.label()
        features.append(lsCat)
        features.append(rsCat)
        rsContainsVP = False
        if rs:
            if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                rsContainsVP = True
        # TODO: Figure out how to check if rs contains a trace (given the tree/grammar)
        features.append(rsContainsVP)
        #featureList.append(rsContainsTrace) # TODO
        rootRoute = getPathToRoot(parent, [])
        features.append('_'.join(rootRoute))
        cRoute = compressRoute([x for x in rootRoute])
        features.append('_'.join(cRoute))

        treeVectors.append(features)

    return treeVectors

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

def constructEmbeddingsMatrix(matrix, ed, dim):

    # construct labeldict
    cid = 1
    ld = {}
    dl = {} # reversed, may want this for debugging later...
    for row in matrix:
        for elem in row[1:-1]:
            if not elem in ld:
                ld[elem] = cid
                dl[cid] = elem
                cid += 1

    # based on the following features: 'token', 'pos', 'leftbigram', 'leftpos', 'leftposbigram', 'rightbigram', 'rightpos', 'rightposbigram', 'selfCategory', 'parentCategory', 'leftsiblingCategory', 'rightsiblingCategory', 'rightsiblingContainsVP', 'pathToRoot', 'compressedPath', 'class_label'

                
    nm = []
    labels = []
    for row in matrix:
        vector = np.zeros((1, dim), dtype=np.float)
        conn, pos, leftbigram, leftpos, leftposbigram, rightbigram, rightpos, rightposbigram, selfcategory, parentcategory, leftsiblingcategory, rightsiblingcategory, rightsiblingcontainsvp, pathtoroot, compressedpath, class_label = row[1:]
        if not conn in ed: # unlikely to happen
            vector = np.append(vector, np.random.random((1, dim)), axis=0)
        else:
            vector = np.append(vector, np.array([ed[conn]]), axis=0)
            
        vector = np.append(vector, [np.repeat(ld[pos], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[leftbigram], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[leftpos], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[leftposbigram], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[rightbigram], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[rightpos], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[rightposbigram], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[selfcategory], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[parentcategory], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[leftsiblingcategory], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[rightsiblingcategory], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[rightsiblingcontainsvp], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[pathtoroot], dim)],axis=0)
        vector = np.append(vector, [np.repeat(ld[compressedpath], dim)],axis=0)
        
        nm.append(vector)
        labels.append([class_label])

    return np.array(nm), labels
    

def trainAndEval(inputdim, outputdim, X_train, Y_train, X_test, Y_test, numbers, epochs, modelname):

    sys.stderr.write('INFO: Starting with training of network.\n')

    # set parameters:
    batch_size = 32
    hidden_dims = 250
    hidden_dims2 = 64

    model = Sequential()

    model.add(Conv1D(hidden_dims, 3, activation='relu', input_shape=(None,inputdim)))
    model.add(Conv1D(hidden_dims, 3, activation='relu'))
    #model.add(MaxPooling1D(3))
    #model.add(Conv1D(hidden_dims2, 3, activation='relu'))
    #model.add(Conv1D(hidden_dims2, 3, activation='relu'))
    #model.add(Conv1D(64, 3, activation='relu'))
    #model.add(Conv1D(64, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(hidden_dims2, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(hidden_dims2, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(hidden_dims2, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(outputdim, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  #model.compile(loss='categorical_crossentropy',
                  #optimizer='adam',
                  optimizer='rmsprop',
                  #metrics=['categorical_accuracy'])
                  metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              #epochs=epochs,
              #validation_data=(x_test, y_test))
              epochs=epochs)

    

    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
    print('DEBUG Result: ', loss_and_metrics)
    """
    numbers = np.array([[0,0,0,0],[0,0,0,0]], dtype=np.int)
    for x in range(len(matrix_test)):
        test_data = np.array([matrix_test[x]])
        cla = model.predict(test_data,batch_size=128)
        label = np.argmax(Yvec_test[x])
        label_predicted = np.argmax(cla)
        numbers[label][label_predicted] = numbers[label][label_predicted] + 1
    """
    # TODO:
    # guess I want to manually get f-score here (or from metrics?)
    # then compare with Vladimir's stuff to see if I can sensibly use an LSTM here instead


def preprocess():

    fmatrix = []
    mid = 0
    #fmatrix.append(['id'] + [pos2column[key] for key in pos2column])


    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/standoffConnectors/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)


    pfname = 'sent2parsed.pickle'
    sent2parsed = {}
    if os.path.exists(pfname):
        with codecs.open(pfname, 'rb') as handle:
            sent2parsed = pickle.load(handle)
    
    file2tokenlist = defaultdict(list)

    connectivecandidates = set()
    for fno, name in enumerate(fileversions):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        for token in tokenlist:
            if token.isConnective:
                connectivecandidates.add(token.token)
        file2tokenlist[name] = tokenlist

    for fno, fname in enumerate(file2tokenlist):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (fname, fno+1, len(fileversions)))
        tokenlist = file2tokenlist[fname]
        for index, token in enumerate(tokenlist):
            if token.token in connectivecandidates:
                fullsent = token.fullSentence
                filteredtokens = filterTokens(fullsent.split())
                parentedTree = None
                if fullsent in sent2parsed:
                    parentedTree = sent2parsed[fullsent]
                else:
                    tree = lexParser.parse(filteredtokens)
                    parentedTreeIterator = ParentedTree.convert(tree)
                    for t in parentedTreeIterator:
                        parentedTree = t
                        break # always taking the first, assuming that this is the best scoring tree...
                sent2parsed[fullsent] = parentedTree
                features = getSingleWordFeaturesFromTree(index, tokenlist, token, parentedTree)
                row = [mid] + features + [token.isConnective]
                mid += 1
                fmatrix.append(row)


    with open(pfname, 'wb') as handle:
        pickle.dump(sent2parsed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write('INFO: Parsed map pickled to %s.\n' % pfname)


    dim = 300
    embd = loadExternalEmbeddings('cc.de.300.vec')
    matrix, labels = constructEmbeddingsMatrix(fmatrix, embd, dim)

    featurematrix = 'connectiveBinaryClassFeatures.matrix'
    with open(featurematrix, 'w') as fout:
        fout.write(str(np.shape(matrix))+'\n')
        for samples in matrix:
            line = ''
            for row in samples:
                for val in row:
                    line = line + ' ' + str(val)
                line = line.strip() + '\t'
            fout.write(line + '\n')
    fout.close()

    labelmatrix = 'connectiveBinaryClassLabels.matrix'
    with open(labelmatrix, 'w') as csvout:
        w = csv.writer(csvout, delimiter=',')
        for row in labels:
            w.writerow(row)


def execute(dataf, labelsf, inputdim, epochs, modelname):

    numIterations = 1

    data = []
    with open(dataf, 'r') as f:
        embeddingslines = [x.strip() for x in f.readlines()]
        for embLine in embeddingslines[1:]:
            vector = []
            parts = embLine.split('\t')
            for p in parts:
                values = p.strip().split(' ')
                vectorLine = np.array([float(x) for x in values])
                vector.append(vectorLine)
            data.append(vector)
        data = np.array(data)

    """
    sys.stderr.write('INFO: Loading data from preprocessed matrix.\n')
    data = []
    labels = []
    with open(dataf, 'r') as d:
        for index, line in enumerate(d.readlines()):
            if index == 0: # first line holds dimensions
                pass
            else:
                line = line.replace('\t', ' ').strip()
                data.append(line.split())
    """
    labels = []
    with open(labelsf, 'r') as l:
        for line in l.readlines():
            labels.append(line.strip())
    #"""
    splits = getDataSplits(numIterations, len(data))

    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration %i of %i...\n' % (i+1, numIterations))
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        testrows = []
        trainrows = []
        for index, tupl in enumerate(zip(data, labels)):
            dat, lab = tupl
            if index >= splits[i] and index <= splits[i+1]:
                X_test.append(dat)
                Y_test.append(lab)
            else:
                X_train.append(dat)
                Y_train.append(lab)

        
        labeldict = {'False':0, 'True':1}
        numbers = np.array([[0,0,0,0],[0,0,0,0]], dtype=np.int)

        Y_train_cat = [labeldict[x] for x in Y_train]
        Y_train_cat = to_categorical(Y_train_cat)
        outputdim = len(Y_train_cat[0])
        Y_test_cat = [labeldict[x] for x in Y_test]
        Y_test_cat = to_categorical(Y_test_cat)

        trainAndEval(inputdim, outputdim, np.array(X_train), np.array(Y_train_cat), np.array(X_test), np.array(Y_test_cat), numbers, epochs, modelname)
        


            

if __name__ == '__main__':

    #preprocess()
    
    execute('connectiveBinaryClassFeatures.matrix', 'connectiveBinaryClassLabels.matrix', 300, 20, 'dummy')
    
    
