import PCCParser
import os
import re
import codecs
from collections import defaultdict
import pandas
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

import numpy
from sklearn import svm
import sys
from nltk.tag import stanford
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
import keras_metrics
import time

embd = {}
posembd = {}
pos2column = {}
ROWDIM = 0

def getFeatureMatrix(fileversions, sent2tagged):

    someId2extArgPosition = defaultdict(int)
    someId2connective = defaultdict(str)

    matrix = []
    headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId', 'class_label']
    for i2, j2 in enumerate(headers):
        if i2 > 0:
            pos2column[i2-1] = j2
    matrix.append(headers)
    mid = 1
    connectiveSingleTokens = 0


    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    for fno, name in enumerate(fileversions):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist = PCCParser.parseConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        rid2connsentid = defaultdict(set)
        rid2conn = defaultdict(list)
        rid2extargsentid = defaultdict(set)
        rid2conndt = defaultdict(list)
        rid2extargtokens = defaultdict(list)
        rid2intargtokens = defaultdict(list)
        
        for token in tokenlist:
            if token.segmentType == 'connective':
                connectiveSingleTokens += 1
                rid2connsentid[token.unitId].add(token.sentenceId)
                rid2conn[token.unitId].append(token.token)
                rid2conndt[token.unitId].append(token)
            elif token.segmentType == 'unit':
                if token.intOrExt == 'ext':
                    rid2extargsentid[token.unitId].add(token.sentenceId)
                    rid2extargtokens[token.unitId].append(token)
                elif token.intOrExt == 'int':
                    rid2intargtokens[token.unitId].append(token)
            for rid in token.embeddedUnits:
                if token.embeddedUnits[rid] == 'ext':
                    rid2extargsentid[rid].add(token.sentenceId)

        for rid in rid2connsentid:
            c = ' '.join([dt.token for dt in rid2conndt[rid]])
            p = 0
            someId2connective[name + '_' + str(rid)] = ' '.join(rid2conn[rid])
            if sorted(rid2connsentid[rid])[0] == sorted(rid2extargsentid[rid])[0]:
                someId2extArgPosition[name + '_' + str(rid)] = 1
                p = 1
            elif sorted(rid2extargsentid[rid])[0] - sorted(rid2connsentid[rid])[0] > 0:
                someId2extArgPosition[name + '_' + str(rid)] = 2
                p = 2
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] == 1:
                someId2extArgPosition[name + '_' + str(rid)] = 0
                p = 0
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] > 1:
                someId2extArgPosition[name + '_' + str(rid)] = 0
                p = 0
            fullsent = rid2conndt[rid][0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[rid2conndt[rid][0].sentencePosition][1]
            nextpos = tagged[rid2conndt[rid][len(c.split())-1].sentencePosition + 1][1]
            
            row = [str(mid), c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, rid2conndt[rid][0].sentenceId, p]
            mid += 1
            matrix.append(row)

    return matrix, sent2tagged

def getDataSplits(numIterations, dataSize):

    p = int(dataSize / 10)
    pl = [int(x) for x in range(0, dataSize, p)]
    pl.append(int(dataSize))    
    return pl



def getNumbers(test_features, test_labels, results, d, debugMode):

    if debugMode:
        #fix recovering of connective here for error analysis
        for inp, pred, label in zip(test_features, results, test_labels):
            reconstr = []
            for fi, fv in enumerate(inp):
                reconstr.append(d[pos2column[fi]].inverse_transform(fv))
            #if not pred == label:
            #print('WRONG CLASSIFICATION:')
            #print('features:', reconstr)
            #print('predicted:', d['class_label'].inverse_transform(pred))
            #print('actual:', d['class_label'].inverse_transform(label))
            #print()
            
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    accuracy = accuracy_score(test_labels, results)
    precision = precision_score(test_labels, results, average='weighted')
    recall = recall_score(test_labels, results, average='weighted')
    f1 = f1_score(test_labels, results, average='weighted')

    return accuracy, precision, recall, f1

def intifyMatrix(m):

    d = {}
    i = 1
    im = []
    for row in m:
        introw = []
        row = [str(x) for x in row]
        for s in row[:-1]:
            if s in d:
                introw.append(d[s])
            else:
                d[s] = i
                introw.append(i)
                i += 1
        introw.append(row[len(row)-1]) # don't want to intify the label (which is already an int anyway)
        im.append(introw)

    return im, d

def trainKerasPositionClassifier(csvmatrix):

    headers, csvmatrix = csvmatrix[0], csvmatrix[1:]
    dim = 300
    nmatrix = []
    smatrix = []
    for row in csvmatrix:
        srow = [str(x) for x in row]
        smatrix.append(srow)
    feature2ohvposition, feature2ohvlength = getf2ohvpos(smatrix)
    for row in smatrix:
        # headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId', 'class_label']
        _id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
        nrow = []
        position_ohv = [0] * feature2ohvlength[3]
        position_ohv[feature2ohvposition[3][sentencePosition]] = 1
        #postag_ohv = [0] * feature2ohvlength[2]
        #postag_ohv[feature2ohvposition[2][postag]] = 1
        rootroute_ohv = [0] * feature2ohvlength[4]
        rootroute_ohv[feature2ohvposition[4][rootroute]] = 1
        
        nrow += position_ohv
        #nrow += postag_ohv
        nrow += rootroute_ohv
        
        if tok in embd:
            for item in embd[tok]:
                nrow.append(item)
        else:
            for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                nrow.append(item)
        # NOTE that this code assumes that postag embeddings and regular embeddings have the same dimension
        #"""
        if postag in posembd:
            for item in posembd[postag]:
                nrow.append(item)
        else:
            print('pos not in emb:', postag)
            for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                nrow.append(item)
        #"""
                
        global ROWDIM
        ROWDIM = len(nrow)
        nrow.append(label)
        nmatrix.append(nrow)

    
    df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
    ds = df.values
    X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
    Y = ds[:,numpy.shape(df)[1]-1]

    seed = 6
    batch_size = 5
    epochs = 10
    classifier = KerasClassifier(build_fn=create_baseline_model, epochs=epochs, batch_size=batch_size)
    classifier.fit(X, Y, verbose=0)


    return classifier, feature2ohvposition, feature2ohvlength

def trainPositionClassifier(csvmatrix): # hardcoded to use randomforest, may want to use keras instead at some point
    # converting to dataframe and then to ints
    headers, csvmatrix = csvmatrix[0], csvmatrix[1:]
    intmatrix, d = intifyMatrix(csvmatrix)
    df = pandas.DataFrame(intmatrix, columns=headers)
    #d = defaultdict(LabelEncoder)
    #fit = df.apply(lambda x: d[x.name].fit_transform(x))
    #df = df.apply(lambda x: d[x.name].transform(x))
    
    train_labels = df.class_label
    labels = list(set(train_labels))
    train_labels = numpy.array([labels.index(x) for x in train_labels])
    train_features = df.iloc[:,1:len(headers)-1]
    train_features = numpy.array(train_features)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train_features, train_labels)

    return classifier, d


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


def evaluatePositionOnlyNeural(fmatrix):

    #embd = {}
    embd = loadExternalEmbeddings('cc.de.300.vec')
    posembd = loadExternalEmbeddings('tiger_pos_model.vec')
    
    dim = 300
    nmatrix = []
    smatrix = []
    for row in fmatrix:
        srow = [str(x) for x in row]
        smatrix.append(srow)
    feature2ohvposition, feature2ohvlength = getf2ohvpos(smatrix)
    for row in smatrix:
        # headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId', 'class_label']
        _id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
        nrow = []
        position_ohv = [0] * feature2ohvlength[3]
        position_ohv[feature2ohvposition[3][sentencePosition]] = 1
        #postag_ohv = [0] * feature2ohvlength[2]
        #postag_ohv[feature2ohvposition[2][postag]] = 1
        rootroute_ohv = [0] * feature2ohvlength[4]
        rootroute_ohv[feature2ohvposition[4][rootroute]] = 1
        
        nrow += position_ohv
        #nrow += postag_ohv
        nrow += rootroute_ohv
        
        if tok in embd:
            for item in embd[tok]:
                nrow.append(item)
        else:
            for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                nrow.append(item)
        #"""
        if postag in posembd:
            for item in posembd[postag]:
                nrow.append(item)
        else:
            for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                nrow.append(item)
        #"""
                
        global ROWDIM
        ROWDIM = len(nrow)
        nrow.append(label)
        nmatrix.append(nrow)

    
    df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
    ds = df.values
    X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
    Y = ds[:,numpy.shape(df)[1]-1]

    seed = 6
    batch_size = 5
    epochs = 10
    classifier = KerasClassifier(build_fn=create_baseline_model, epochs=epochs, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    acc = cross_val_score(classifier, X, Y, cv=kfold, scoring='accuracy')
    f1 = cross_val_score(classifier, X, Y, cv=kfold, scoring='f1_micro')
    precision = cross_val_score(classifier, X, Y, cv=kfold, scoring='precision_micro')
    recall = cross_val_score(classifier, X, Y, cv=kfold, scoring='recall_micro')

    return acc.mean(), precision.mean(), recall.mean(), f1.mean()


def create_baseline_model():

    
    # this works (baseline)!
    """
    dim = ROWDIM
    model = Sequential()
    model.add(Dense(60, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()]) # perhaps this should not be binary (but categorical instead)
    """

    hidden_dims = 250
    hidden_dims2 = 64
    outputDimension = 1

    model = Sequential()

    model.add(Dense(hidden_dims, input_dim=ROWDIM, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims2, activation='relu'))#2
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims2, activation='relu'))
    model.add(Dropout(0.5))


    #model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(outputDimension, activation='sigmoid')) # softmax
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()]) # perhaps this should not be binary (but categorical instead)
    """
    model.compile(loss='binary_crossentropy',
                  #model.compile(loss='categorical_crossentropy',
                  #optimizer='adam',
                  optimizer='rmsprop',
                  #metrics=['categorical_accuracy'])
                  metrics=['accuracy'])
    """
    
    return model

def evaluatePositionOnly(csvmatrix, numIterations): # hardcoded to use randomforest

    # converting to dataframe and then to ints
    headers, csvmatrix = csvmatrix[0], csvmatrix[1:]
    df = pandas.DataFrame(csvmatrix, columns=headers)
    d = defaultdict(LabelEncoder)
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    df = df.apply(lambda x: d[x.name].transform(x))

    # getting data slices for x-fold cv
    pl = getDataSplits(numIterations, len(csvmatrix))

    _ps = []
    _fs = []
    _rs = []
    _as = []
    
    for i in range(numIterations):
        #sys.stderr.write('INFO: Starting iteration %i of %i for %s.\n' % (i+1, numIterations, algorithm))
        testrows = []
        trainrows = []
        row2origfeaturevals = defaultdict(list)
        for index, row in df.iterrows():
            if index >= pl[i] and index <= pl[i+1]:
                testrows.append(row)
            else:
                for pos, item in enumerate(csvmatrix[index]):
                    row2origfeaturevals[pos].append(item)
                trainrows.append(row)
                
        testdf = pandas.DataFrame(testrows)
        traindf = pandas.DataFrame(trainrows)

        train_labels = traindf.class_label
        labels = list(set(train_labels))
        train_labels = numpy.array([labels.index(x) for x in train_labels])
        train_features = traindf.iloc[:,1:len(headers)-1]
        train_features = numpy.array(train_features)

        test_labels = testdf.class_label
        labels = list(set(test_labels))
        test_labels = numpy.array([labels.index(x) for x in test_labels])
        test_features = testdf.iloc[:,1:len(headers)-1]
        test_features = numpy.array(test_features)

        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(train_features, train_labels)

        results = classifier.predict(test_features)
        
        a, p, r, f = getNumbers(test_features, test_labels, results, d, False)
        _as.append(a)
        _ps.append(p)
        _rs.append(r)
        _fs.append(f)
        
    macroPrecision = sum(_ps)/len(_ps)
    macroRecall = sum(_rs)/len(_rs)
    macroF1 = sum(_fs)/len(_fs)
    acc = sum(_as)/len(_as)
    return acc, macroPrecision, macroRecall, macroF1


def getTestAndTrainFiles(filedict, i, j):

    train = defaultdict()
    test = defaultdict()
    splits = getDataSplits(i, len(filedict)) # bit wasteful to do this all the time, in refactoring can do this outside of iterations loop
    for h, k in enumerate(filedict):
        if h >= splits[j] and h <= splits[j+1]:
            test[k] = fileversions[k]
        else:
            train[k] = fileversions[k]
    return test, train

def loadExternalEmbeddings(embfile):

    starttime = time.time()
    sys.stderr.write('INFO: Loading external embeddings from %s.\n' % embfile)
    ed = defaultdict()
    #"""
    with open(embfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            values = line.split()
            ed[values[0]] = numpy.array([float(x) for x in values[1:]])
    #"""
    endtime = time.time()
    sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))
    return ed


def classifyExtArgs(testfiles, positionClassifier, sent2tagged, feat2ohvpos, feat2ohvlen):

    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    for fno, name in enumerate(testfiles):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist = PCCParser.parseConnectorFile(testfiles[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(testfiles[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(testfiles[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(testfiles[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        rid2connsentid = defaultdict(set)
        rid2conn = defaultdict(list)
        rid2extargsentid = defaultdict(set)
        rid2conndt = defaultdict(list)
        rid2extargtokens = defaultdict(list)
        rid2intargtokens = defaultdict(list)


        sid2tokens = defaultdict(list)
        
        for token in tokenlist:
            sid2tokens[token.sentenceId].append(token)
            if token.segmentType == 'connective':
                rid2connsentid[token.unitId].add(token.sentenceId)
                rid2conn[token.unitId].append(token.token)
                rid2conndt[token.unitId].append(token)
            elif token.segmentType == 'unit':
                if token.intOrExt == 'ext':
                    rid2extargsentid[token.unitId].add(token.sentenceId)
                    rid2extargtokens[token.unitId].append(token)
                elif token.intOrExt == 'int':
                    rid2intargtokens[token.unitId].append(token)
            for rid in token.embeddedUnits:
                if token.embeddedUnits[rid] == 'ext':
                    rid2extargsentid[rid].add(token.sentenceId)

        for rid in rid2extargtokens:
            sid = rid2conndt[rid][0].sentenceId
            actualtokens = rid2extargtokens[rid]
            c = ' '.join([dt.token for dt in rid2conndt[rid]])
            fullsent = rid2conndt[rid][0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[rid2conndt[rid][0].sentencePosition][1]
            nextpos = tagged[rid2conndt[rid][len(c.split())-1].sentencePosition + 1][1]
            ### only need this p for debugging...
            p = 0
            if sorted(rid2connsentid[rid])[0] == sorted(rid2extargsentid[rid])[0]:
                p = 1
            elif sorted(rid2extargsentid[rid])[0] - sorted(rid2connsentid[rid])[0] > 0:
                p = 2
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] == 1:
                p = 0
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] > 1:
                p = 0
            
            #features = [c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, sid]
            
            #localheaders = ['connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId']
            #intfeatures, d = intifyFeatures([str(x) for x in features], d)
            #df = pandas.DataFrame([intfeatures], columns=localheaders)
            #test_features = df.iloc[:,:]
            #test_features = numpy.array(test_features)
            #relativeSentPos = positionClassifier.predict(test_features)

            # this below fails silently if key not in dict (feat2ohvlen and feat2ohvpos are default dicts), in which case it sets the first elem to 1 (because val is 0 I think). Guess random would be better, but assuming that this happen that often, it's not so bad I guess.
            ###_id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
            nrow = []
            position_ohv = [0] * feat2ohvlen[3]
            position_ohv[feat2ohvpos[3][str(rid2conndt[rid][0].sentencePosition)]] = 1
            #postag_ohv = [0] * feat2ohvlen[2]
            #postag_ohv[feat2ohvpos[2][currpos]] = 1
            rootroute_ohv = [0] * feat2ohvlen[4]
            rootroute_ohv[feat2ohvpos[4][str(rid2conndt[rid][0].pathToRoot)]] = 1
            #print('debugging position ohv len:', len(position_ohv))
            #print('hot index:', feat2ohvpos[3][str(rid2conndt[rid][0].sentencePosition)])
            #print('debugging postag ohv len:', len(postag_ohv))
            #print('hot index:', feat2ohvpos[2][currpos])
            #print('debugging postag ohv len:', len(rootroute_ohv))
            #print('hot index:', feat2ohvpos[4][str(rid2conndt[rid][0].pathToRoot)])
            #print()
            nrow += position_ohv
            #nrow += postag_ohv
            nrow += rootroute_ohv

            dim = 300
            if c in embd:
                for item in embd[c]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)
            #"""        
            if currpos in posembd:
                for item in posembd[currpos]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)
            #"""
                    
            relativeSentPos = positionClassifier.predict(numpy.array([nrow,]))

            """
            print('rid:', rid)
            print('actualtokens:', [(t.token, t.tokenId) for t in actualtokens])
            print('conn sent:', [(ttt.token, ttt.tokenId) for ttt in sid2tokens[sorted(rid2connsentid[rid])[0]]])
            print('c:', c)
            print('rel pos:', relativeSentPos)
            print('actual pos:', p)
            # reminder: 0 is (any of the) previous sentence(s), 1 is same sentence and 2 is following sentence
            """
            # taking all tokens of the sentence:
            targetSentId = None
            if int(relativeSentPos[0][0]) == 0:
                targetSentId = max(sid - 1, 0)
            elif int(relativeSentPos[0][0]) == 1:
                targetSentId = sid
            elif int(relativeSentPos[0][0]) == 2:
                targetSentId = sid + 1 # hoping I'm not hitting the end here. Will return IndexError if this is the case.
            #else:
                #print('DEBG: Weird value:', relativeSentPos, c, rid, name)
            predictedTokens = [(tt.token, tt.tokenId) for tt in sid2tokens[targetSentId]]
            #print('predicted tokens:', predictedTokens)
            #print()
            #print()
            total += 1
            if targetSentId in rid2extargsentid[rid]:
                correct += 1
            predictedIds = [t[1] for t in predictedTokens]
            actualIds = [t.tokenId for t in actualtokens]
            
            for tokenId in set(predictedIds + actualIds):
                if tokenId in actualIds and tokenId in predictedIds:
                    tp += 1
                elif tokenId in actualIds:
                    fn += 1
                else:
                    fp += 1

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

                    
    return correct / total, precision, recall, f1 #  TODO: acc here is for position classification, not for token ext args
                    

def intifyFeatures(f, d):

    n = []
    maxId = 0
    for j in d:
        if d[j] > maxId:
            maxId = d[j]
    maxId += 1
    for x in f:
        if x in d:
            n.append(d[x])
        else:
            d[x] = maxId
            n.append(maxId)
            maxId += 1
            
    return n, d

def loadEmbeddingsOnce():

    global embd
    embd = loadExternalEmbeddings('cc.de.300.vec')
    global posembd
    posembd = loadExternalEmbeddings('tiger_pos_model.vec')
    
    #embd = {}
    #print('WARNING:NOT LOADING ANYTHING!!!')


if __name__ == '__main__':

    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/connectives/')
    #connectorfiles = PCCParser.getInputfiles('/share/corrected_connectives/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

    import pickle
    pfname = 'sent2tagged.pickle'
    
    sent2tagged = {}
    if os.path.exists(pfname):
        with codecs.open(pfname, 'rb') as handle:
            sent2tagged = pickle.load(handle)

    ### Section below is to train and eval only the position classifier
    
    
    matrix, sent2tagged = getFeatureMatrix(fileversions, sent2tagged)

    with open(pfname, 'wb') as handle:
        pickle.dump(sent2tagged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write('INFO: Tagged map pickled to "%s".\n' % pfname)


    
    # lines below are for only evaluating sentence position classifier. Will want to first do this, then get actual words and get p, r, f for that, but have to wrap all of that in x-fold cv
    #alg = 'randomforest' # hardcoded in subfunctions
    #numIterations = 10
    #a, p, r, f = evaluatePositionOnly(matrix, numIterations)
    #print('%s accuracy: %f' % (alg, a))
    #print('%s precision: %f' % (alg, p))
    #print('%s recall: %f' % (alg, r))
    #print('%s f1: %f' % (alg, f))
    
    #randomforest accuracy: 0.941964
    #randomforest precision: 0.940528
    #randomforest recall: 0.941964
    #randomforest f1: 0.940630
    

    #a, p, r, f = evaluatePositionOnlyNeural(matrix)
    #print('%s accuracy: %f' % ('keras', a))
    #print('%s precision: %f' % ('keras', p))
    #print('%s recall: %f' % ('keras', r))
    #print('%s f1: %f' % ('keras', f))

    # REDO this with my definitive, final keras model architecture before citing it for some paper
    #keras accuracy: 0.945316
    #keras precision: 0.936295
    #keras recall: 0.942589
    #keras f1: 0.946714
    
    # NOTE: difference between randomforest and keras classifier do not really seem to be significant (every time I run it again, they swap 1st and 2nd place)
    
    #### Here comes the section to evaluate the full thing (first detect position, then token span)

    #"""
    _a = []
    _p = []
    _r = []
    _f = []
    numIterations = 10
    loadEmbeddingsOnce() # needed during training and testing, loading only once here (only necessary if keras classifier is used)
    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration %i of %i...\n' % (i+1, numIterations))
        test, train = getTestAndTrainFiles(fileversions, numIterations, i)
        trainmatrix, sent2tagged = getFeatureMatrix(train, sent2tagged)
        #positionClassifier, d = trainPositionClassifier(trainmatrix)
        positionClassifier, feat2ohvpos, feat2ohvlen = trainKerasPositionClassifier(trainmatrix)
        #tp, fp, fn = classifyExtArgs(test, positionClassifier, sent2tagged, d, tp, fp, fn)
        a, p, r, f = classifyExtArgs(test, positionClassifier, sent2tagged, feat2ohvpos, feat2ohvlen)
        _a.append(a)
        _p.append(p)
        _r.append(r)
        _f.append(f)
    print('avg a:', sum(_a) / len(_a))
    print('avg p:', sum(_p) / len(_p))
    print('avg r:', sum(_r) / len(_r))
    print('avg f:', sum(_f) / len(_f))

    # TODO: baseline stands. Now improve upon this with some syntax rules.
    
    #precision = tp / float(tp + fp)
    #recall = tp / float(tp + fn)
    #f1 = 2 * ((precision * recall) / (precision + recall))
    #print('p:', precision)
    #print('r:', recall)
    #print('f:', f1)
    
    #avg a: 0.950243050811965
    #avg p: 0.44815664520804877
    #avg r: 0.7308088956887122
    #avg f: 0.5539544728215057


    
    #with pos embeddings:
    #keras accuracy: 0.952299
    #keras precision: 0.945092
    #keras recall: 0.947787
    #keras f1: 0.954101

    #avg a: 0.9605333048952014
    #avg p: 0.46250563420120755
    #avg r: 0.734805830191133
    #avg f: 0.5639276113926787

    #without pos embeddings (one hot vector instead):
    #keras accuracy: 0.952363
    #keras precision: 0.954173
    #keras recall: 0.950586
    #keras f1: 0.946958

    #avg a: 0.9551760572284728
    #avg p: 0.45652153693700254
    #avg r: 0.7289523854631429
    #avg f: 0.559801976512649

