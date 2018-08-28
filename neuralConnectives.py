import numpy as np
import keras
from keras.models import Sequential
import pandas

import keras_metrics
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Input, Embedding, GlobalMaxPooling1D, TimeDistributed, Conv1D, GlobalAveragePooling1D, MaxPooling1D, concatenate
from keras.utils import to_categorical
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_score, recall_score, f1_score
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

ROWDIM = 0
VOCAB_SIZE = 0
INSTANCES = 0

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


def loadSentenceEmbeddings(embfile):

    dim = 300
    ed = defaultdict()
    with open(embfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            sent = ' '.join(values[:len(values)-dim])
            numbers = values[len(values)-dim:]
            ed[sent.strip()] = np.array([float(x) for x in numbers])
    return ed
            
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



def trainAndEvalCNN(inputdim, outputdim, X_train, Y_train, X_test, Y_test, numbers, epochs, modelname):

    sys.stderr.write('INFO: Starting with training of CNN network.\n')

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
    model.add(Dense(outputdim, activation='sigmoid'))
    
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
    predictions = model.predict(X_test)


    precision = precision_score(Y_test, predictions, average='micro')
    recall = recall_score(Y_test, predictions, average='micro')
    f1 = f1_score(Y_test, predictions, average='micro')
    print('p:', precision)
    print('r:', recall)
    print('f:', f1)
    
    # TODO:
    # compare with Vladimir's stuff to see if I can sensibly use an LSTM here instead

    
def create_baseline_model():
    # copied from https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    #dim = 67 + 2 # max sent length plus number of additional features
    dim = ROWDIM #300 + 300 + 300 + 2 # adding left neighbour and right neighbour
    #"""
    model = Sequential()
    model.add(Dense(60, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    #"""
    # LSTM model (from https://keras.io/getting-started/sequential-model-guide/ at paragraph Sequence classification with LSTM:)
    """
    model = Sequential()
    model.add(Embedding(dim, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    """
    """
    hidden_size = 128 # very much randomly picked
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, hidden_size, input_length=INSTANCES))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    #if use_dropout:
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    """
    
    return model
    
def preprocessLSTM():


    pfname = 'sent2parsed.pickle'
    sent2parsed = {}
    if os.path.exists(pfname):
        with codecs.open(pfname, 'rb') as handle:
            sent2parsed = pickle.load(handle)
    

    
    fmatrix = []

    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/standoffConnectors/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

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

    ld = {'_':0}
    dl = {0:'_'}
    vid = 1
    maxsentlength = 0
    pccsentences = set()
    for fno, fname in enumerate(file2tokenlist):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (fname, fno+1, len(fileversions)))
        tokenlist = file2tokenlist[fname]
        for index, token in enumerate(tokenlist):
            if token.token in connectivecandidates:
                fullsent = token.fullSentence
                pccsentences.add(fullsent)
                if len(fullsent.split()) > maxsentlength:
                    maxsentlength = len(fullsent.split())
                labelval = 0
                if token.isConnective:
                    labelval = 1
                #row = [token.sentencePosition, token.pos, fullsent, labelval]#token.isConnective]
                features = getSingleWordFeaturesFromTree(index, tokenlist, token, sent2parsed[fullsent])
                row = [token.sentencePosition]
                row += features
                row.append(labelval)

                ###row = [token.sentencePosition, token.pos, token.token, fullsent, fullsent.split()[max(token.sentencePosition-1, 0)], fullsent.split()[min(token.sentencePosition+1, len(fullsent.split()))], labelval]#token.isConnective]
                
                for t2 in fullsent.split():
                    if not t2 in ld:
                        ld[t2] = vid
                        dl[vid] = t2
                        vid += 1
                if not token.pos in ld:
                    ld[token.pos] = vid
                    dl[vid] = token.pos
                    vid += 1

                    
                #row = [fullsent, token.isConnective]
                fmatrix.append(row)
    """
    intmatrix = []
    labels = []
    for frow in fmatrix:
        text, label = frow
        introw = [0]*maxsentlength
        for i2, t in enumerate(text.split()):
            introw[i2] = ld[t]
        intmatrix.append(introw)
        labels.append(label)
    """
    # PLAN/TODO: get access to TIGER or TÜBA-DZ and generate pos2vec embeddings (https://machinelearningmastery.com/develop-word-embeddings-python-gensim/), then use pos embedding as feature to see if I can get it up from 83. 
    # 
    #sys.exit()
    #sentembd = loadSentenceEmbeddings('pcc_sentence_embeddings/sent_embeddings.txt')

    embd = {}
    embd = loadExternalEmbeddings('cc.de.300.vec')
    
    dim = 300
    nmatrix = []
    feature2ohvposition, feature2ohvlength = getf2ohvpos(fmatrix) # nested dict, first key is pos, second key is feature value, val is position in one hot vector

    vocabset = set()
    
    for row in fmatrix:
        # getting a bit messy. What is appended to fmatrix and what is added to nmatrix differs. Keep that in mind :)
        ###position, postag, tok, sent, leftn, rightn, label = row
        for xx in row[:-1]:
            vocabset.add(xx)
        position, tok, postag, leftn, leftpos, leftposbigram, rightn, rightpos, rightposbigram, selfcat, parentcat, lscat, rscat, rscontainsvp, rootroute, compressedroute, label = row
        nrow = []
        position_ohv = [0] * feature2ohvlength[0]
        position_ohv[feature2ohvposition[0][position]] = 1
        postag_ohv = [0] * feature2ohvlength[2]
        postag_ohv[feature2ohvposition[2][postag]] = 1
        leftpos_ohv = [0] * feature2ohvlength[4]
        leftpos_ohv[feature2ohvposition[4][leftpos]] = 1
        rightpos_ohv = [0] * feature2ohvlength[7]
        rightpos_ohv[feature2ohvposition[7][rightpos]] = 1
        selfcat_ohv = [0] * feature2ohvlength[9]
        selfcat_ohv[feature2ohvposition[9][selfcat]] = 1
        parentcat_ohv = [0] * feature2ohvlength[9]
        parentcat_ohv[feature2ohvposition[9][parentcat]] = 1
        lscat_ohv = [0] * feature2ohvlength[10]
        lscat_ohv[feature2ohvposition[10][lscat]] = 1
        rscat_ohv = [0] * feature2ohvlength[11]
        rscat_ohv[feature2ohvposition[11][rscat]] = 1
        rscontainsvp_ohv = [0] * feature2ohvlength[12]
        rscontainsvp_ohv[feature2ohvposition[12][rscontainsvp]] = 1
        rootroute_ohv = [0] * feature2ohvlength[13]
        rootroute_ohv[feature2ohvposition[13][rootroute]] = 1
        compressedroute_ohv = [0] * feature2ohvlength[14]
        compressedroute_ohv[feature2ohvposition[14][compressedroute]] = 1

        nrow += position_ohv
        nrow += postag_ohv
        nrow += leftpos_ohv
        nrow += rightpos_ohv
        nrow += selfcat_ohv
        nrow += parentcat_ohv
        nrow += lscat_ohv
        nrow += rscat_ohv
        nrow += rscontainsvp_ohv
        nrow += rootroute_ohv
        nrow += compressedroute_ohv
        
        if tok in embd:
            for item in embd[tok]:
                nrow.append(item)
        else:
            for item in np.ndarray.flatten(np.random.random((1, dim))):
                nrow.append(item)
        
        if leftn in embd:
            for item in embd[leftn]:
                nrow.append(item)
        else:
            for item in np.ndarray.flatten(np.random.random((1, dim))):
                nrow.append(item)
        if rightn in embd:
            for item in embd[rightn]:
                nrow.append(item)
        else:
            for item in np.ndarray.flatten(np.random.random((1, dim))):
                nrow.append(item)


        # 10 epochs with all syntactic features:
        #a: 0.837048473348
        #p: 0.806631417301
        #r: 0.834144574823
        #f: 0.783044938925
        # 100 epochs:
        #a: 0.853193893393
        #p: 0.699021800703
        #r: 0.812059973924
        #f: 0.782833851965



                
        #for item2 in sentembd[sent]:
            #nrow.append(item2)
        #nrow += ['_']*maxsentlength
        """
        nrow += [ld['_']]*maxsentlength
        localtokens = sent.split()
        for i3 in range(len(localtokens)):
            nrow[i3+nr_features] = ld[localtokens[i3]]
        """
        global ROWDIM
        ROWDIM = len(nrow)
        nrow.append(label)
        nmatrix.append(nrow)

    global VOCAB_SIZE
    VOCAB_SIZE = len(vocabset)
    global INSTANCES
    INSTANCES = len(nmatrix)

    
    df = pandas.DataFrame(np.array(nmatrix), columns=None)
    #print(df)
    ds = df.values
    #print('shape:', np.shape(df)[1])
    X = ds[:,0:np.shape(df)[1]-1].astype(float)
    Y = ds[:,np.shape(df)[1]-1]
    # note that I don't really need to do the following, since my labels are already integers (or floats, not sure). Can try with leaving it out, if it all works fine still, all the better :)
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    #maxslength: 67

    # abusing the preprocess function and just experimenting here (with the actual Keras stuff directly)
    seed = 6
    batch_size = 5
    epochs = 10
    classifier = KerasClassifier(build_fn=create_baseline_model, epochs=epochs, batch_size=batch_size)
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    #pccs = open('pcc_sentences.txt', 'w')
    #for sent in pccsentences:
        #pccs.write(sent + '\n')
    #pccs.close()
    #sys.exit(1)
    
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    print('X shape:', np.shape(X))
    print('Y shape:', np.shape(Y))
    acc = cross_val_score(classifier, X, Y, cv=kfold, scoring='accuracy')
    f1 = cross_val_score(classifier, X, Y, cv=kfold, scoring='f1')
    precision = cross_val_score(classifier, X, Y, cv=kfold, scoring='precision')
    recall = cross_val_score(classifier, X, Y, cv=kfold, scoring='recall')
    print()
    print('a:', acc.mean())
    print('p:', precision.mean())
    print('r:', recall.mean())
    print('f:', f1.mean())



def getf2ohvpos(fmatrix):

    f = defaultdict(lambda : defaultdict(int))
    f2 = defaultdict(set)
    a = defaultdict()
    rowsize = 0
    for row in fmatrix:
        rowsize = len(row)
        for pos, val in enumerate(row):
            f2[pos].add(val)
    for i in f2:
        a[i] = len(f2[i])
        for c, i2 in enumerate(f2[i]):
            f[i][i2] = c
    return f, a
            
def preprocessCNN():

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

def executeLSTM(sentenceMatrix, positionMatrix, labelMatrix, epochs):

    numIterations = 1

    embeddings = []
    positions = []
    labels = []

    inputdim = 0
    with open(sentenceMatrix) as sm:
        for line in sm.readlines()[1:]:
            line = line.strip()
            inputdim = len(line.split())
            embeddings.append(line.split())
    embeddings = np.array(embeddings)
    with open(positionMatrix) as pm:
        for line in pm.readlines()[1:]:
            line = line.strip()
            positions.append(line.split())
    positions = np.array(positions)
    with open(labelMatrix) as lm:
        for line in lm.readlines():
            line = line.strip()
            labels.append(line)

    splits = getDataSplits(numIterations, len(embeddings))

    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration %i of %i...\n' % (i+1, numIterations))
        X1_train = []
        X2_train = []
        Y_train = []
        X1_test = []
        X2_test = []
        Y_test = []
        for index, tupl in enumerate(zip(embeddings, positions, labels)):
            sent, pos, lab = tupl
            if index >= splits[i] and index <= splits[i+1]:
                X1_test.append(sent)
                X2_test.append(pos)
                Y_test.append(lab)
            else:
                X1_train.append(sent)
                X2_train.append(pos)
                Y_train.append(lab)

        
        #numbers = np.array([[0,0,0,0],[0,0,0,0]], dtype=np.int)

        Y_train_cat = to_categorical(Y_train)
        outputdim = len(Y_train_cat[0])
        Y_test_cat = to_categorical(Y_test)

        trainAndEvalLSTM(inputdim, outputdim, np.array(X1_train), np.array(X2_train), np.array(Y_train_cat), np.array(X1_test), np.array(X2_test), np.array(Y_test_cat), epochs)
    
            
def executeCNN(dataf, labelsf, inputdim, epochs, modelname):

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

        trainAndEvalCNN(inputdim, outputdim, np.array(X_train), np.array(Y_train_cat), np.array(X_test), np.array(Y_test_cat), numbers, epochs, modelname)


            

if __name__ == '__main__':

    #preprocessCNN()
    
    #executeCNN('connectiveBinaryClassFeatures.matrix', 'connectiveBinaryClassLabels.matrix', 300, 20, 'dummy')
    

    preprocessLSTM()
    #executeLSTM('LSTM_sentenceEncoding.matrix', 'LSTM_position.matrix', 'LSTM_label.matrix', 20)
