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
from keras.utils import to_categorical

import numpy
from sklearn import svm
import sys
from nltk.tag import stanford
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
import keras_metrics
import time
from nltk.tree import ParentedTree
from nltk.translate.ribes_score import position_of_ngram
import spacy
from nltk.tag.stanford import CoreNLPPOSTagger

embd = {}
posembd = {}
pos2column = {}
ROWDIM = 0
KERAS_OUTPUT_DIM = 0

DEBUG_MODE = False

jar = '/home/peter/Downloads/stanford-postagger-full-2018-10-16/stanford-postagger.jar'
model = '/home/peter/Downloads/stanford-postagger-full-2018-10-16/models/german-fast.tagger'

pos_tagger = stanford.StanfordPOSTagger(model, jar, encoding='utf8')
#pos_tagger = CoreNLPPOSTagger(model, jar, encoding='utf8')



def getFeatureMatrix(fileversions, sent2tagged, goldpostags):

    someId2extArgPosition = defaultdict(int)
    someId2connective = defaultdict(str)

    samesentbaselinedict = defaultdict(lambda : defaultdict(int))
    matrix = []
    samesentmatrix = []
    headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId', 'class_label']
    for i2, j2 in enumerate(headers):
        if i2 > 0:
            pos2column[i2-1] = j2
    matrix.append(headers)
    samesentmatrix.append(headers)
    mid = 1
    connectiveSingleTokens = 0

    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    for fno, name in enumerate(fileversions):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        #tokenlist = PCCParser.parseConnectorFile(fileversions[name]['connectors'])
        tokenlist, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        gtid2ct, sid2tokens = flipdict(tokenlist)

        for dr in discourseRelations:
            # discourseRelations contains tokenIds, instead of DiscourseToken objects. Change this at some point.
            conntokens = [gtid2ct[int(x)] for x in dr.connectiveTokens]
            intargtokens = [gtid2ct[int(x)] for x in dr.intArgTokens]
            extargtokens = [gtid2ct[int(x)] for x in dr.extArgTokens]
            c = ' '.join([dt.token for dt in conntokens])
            p = 0
            if set.intersection(set([dt.sentenceId for dt in conntokens]), set([dt.sentenceId for dt in extargtokens])): # if at least one token of conn and ext arg are in same sentence, assume same sentence scenario
                pass
            else:
                p = conntokens[0].sentenceId - extargtokens[0].sentenceId
            fullsent = conntokens[0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[conntokens[0].sentencePosition][1]
            if goldpostags:
                currpos = conntokens[0].pos
            nextpos = tagged[conntokens[len(conntokens)-1].sentencePosition + 1][1]
            row = [str(mid), c, currpos, conntokens[0].sentencePosition, conntokens[0].pathToRoot, conntokens[0].sentenceId, p]
            mid += 1
            matrix.append(row)
            if p == 0:
                sspos = 0 # default/extarg is before conn
                if int(conntokens[-1].tokenId) < int(extargtokens[0].tokenId):
                    sspos = 1
                samesentbaselinedict[conntokens[0].token][sspos] += 1
                ssrow = [str(mid), c, currpos, conntokens[0].sentencePosition, conntokens[0].pathToRoot, conntokens[0].sentenceId, sspos] # mid will not be consecutive, but think it is not really used anyway. And now at least it corresponds to the other matrix, should that come in handy some time.
                samesentmatrix.append(ssrow)

    return matrix, samesentmatrix, samesentbaselinedict, sent2tagged


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
    labels = []
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
        labels.append(label)

        nmatrix.append(nrow)

    
    df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
    ds = df.values
    X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
    #Y = ds[:,numpy.shape(df)[1]-1]
    Y = to_categorical(numpy.array(labels))

    global KERAS_OUTPUT_DIM
    KERAS_OUTPUT_DIM = len(Y[0])
    

    seed = 6
    batch_size = 5
    epochs = None
    if DEBUG_MODE:
        epochs = 1
    else:
        epochs = 100
    classifier = KerasClassifier(build_fn=create_baseline_model, epochs=epochs, batch_size=batch_size)
    classifier.fit(X, Y, verbose=0)


    return classifier, feature2ohvposition, feature2ohvlength



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


def evaluatePositionOnlyNeural(fmatrix, numIterations):

    embd = {}
    posembd = {}
    #print('WAAAARNING! NOT LOADING EMBS!')
    embd = loadExternalEmbeddings('/home/peter/github/PhdPlayground/customEmbeddings/cc.de.300.PCC.vec')
    posembd = loadExternalEmbeddings('/home/peter/github/PhdPlayground/customEmbeddings/tiger_pos_model.vec.vec')
    
    dim = 300
    nmatrix = []
    smatrix = []
    for row in fmatrix:
        srow = [str(x) for x in row]
        smatrix.append(srow)
    feature2ohvposition, feature2ohvlength = getf2ohvpos(smatrix)
    labels = []
    for row in smatrix[1:]: # skipping header (assuming that first row is header)
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
        labels.append(label)
        nrow.append(label)
        #print('label:', label)
        nmatrix.append(nrow)


    seed = 6
    batch_size = 5
    epochs = 100
    classifier = KerasClassifier(build_fn=create_baseline_model, epochs=epochs, batch_size=batch_size)


    df = pandas.DataFrame(numpy.array(nmatrix), columns=None)
    ds = df.values
    X = ds[:,0:numpy.shape(df)[1]-1].astype(float)
    #Y = ds[:,numpy.shape(df)[1]-1]
    Y = to_categorical(numpy.array(labels))

    global KERAS_OUTPUT_DIM
    KERAS_OUTPUT_DIM = len(Y[0])
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(classifier, X, Y, cv=kfold)
    return results.mean()


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
    #"""
    hidden_dims = 250
    hidden_dims2 = 64
    outputDimension = KERAS_OUTPUT_DIM

    model = Sequential()

    model.add(Dense(hidden_dims, input_dim=ROWDIM, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims2, activation='relu'))#2
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims2, activation='relu'))
    model.add(Dropout(0.5))


    #model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(outputDimension, activation='sigmoid')) # softmax/sigmoid
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def evaluatePositionOnlyMVBaseline(csvmatrix, numIterations):

    headers, csvmatrix = csvmatrix[0], csvmatrix[1:]
    pl = getDataSplits(numIterations, len(csvmatrix))
    df = pandas.DataFrame(csvmatrix, columns=headers)
    d = defaultdict(LabelEncoder)
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    df = df.apply(lambda x: d[x.name].transform(x))

    _as = []
    _ps = []
    _rs = []
    _fs = []
    
    for i in range(numIterations):
        testrows = []
        trainrows = []
        for index, row in df.iterrows():
            if index >= pl[i] and index <= pl[i+1]:
                testrows.append(row)
            else:
                trainrows.append(row)
        pd = defaultdict(lambda : defaultdict(int))
        for row in trainrows:
            c, l = row[1], row[-1]
            pd[c][l] += 1
        c2mvl = defaultdict(int)
        for c in pd:
            c2mvl[c] = sorted(pd[c].items(), key = lambda x: x[1], reverse=True)[0][0]
        
        testdf = pandas.DataFrame(testrows)
        test_labels = testdf.class_label
        labels = list(set(test_labels))
        test_labels = numpy.array([labels.index(x) for x in test_labels])
        test_features = testdf.iloc[:,1:len(headers)-1]
        test_features = numpy.array(test_features)

        predicted_labels = []
        for i, row in testdf.iterrows():
            tc, tl = row.connective, row.class_label
            predicted_labels.append(c2mvl[tc])
        
        a, p, r, f = getNumbers(test_features, test_labels, numpy.array(predicted_labels), d, False)
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

def loadParserMemoryMap():

    parsermemorymap = {}
    pmname = '/home/peter/github/PhdPlayground/picklejars/parsermemory.pickle'
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
    return parsermemorymap

def classifyExtArgs_baseline_approach(testfiles, positionClassifier, sent2tagged, feat2ohvpos, feat2ohvlen):

    
    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    for fno, name in enumerate(testfiles):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        gtid2ct, sid2tokens = flipdict(tokenlist)

        for dr in discourseRelations:
            # discourseRelations contains tokenIds, instead of DiscourseToken objects. Change this at some point.
            conntokens = [gtid2ct[int(x)] for x in dr.connectiveTokens]
            intargtokens = [gtid2ct[int(x)] for x in dr.intArgTokens]
            extargtokens = [gtid2ct[int(x)] for x in dr.extArgTokens]
            c = ' '.join([dt.token for dt in conntokens])
            p = 0
            if set.intersection(set([dt.sentenceId for dt in conntokens]), set([dt.sentenceId for dt in extargtokens])): # if at least one token of conn and ext arg are in same sentence, assume same sentence scenario
                pass
            else:
                p = conntokens[0].sentenceId - extargtokens[0].sentenceId

            sid = conntokens[0].sentenceId
            actualtokens = extargtokens
            fullsent = conntokens[0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[conntokens[0].sentencePosition][1]
            nextpos = tagged[conntokens[len(conntokens)-1].sentencePosition + 1][1]
            #features = [c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, sid]
            #localheaders = ['connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId']
            ###_id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
            nrow = []
            position_ohv = [0] * feat2ohvlen[3]
            position_ohv[feat2ohvpos[3][str(conntokens[0].sentencePosition)]] = 1
            rootroute_ohv = [0] * feat2ohvlen[4]
            rootroute_ohv[feat2ohvpos[4][str(conntokens[0].pathToRoot)]] = 1
            nrow += position_ohv
            nrow += rootroute_ohv

            dim = 300
            if c in embd:
                for item in embd[c]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)

            if currpos in posembd:
                for item in posembd[currpos]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)

                    
            relativeSentPos = positionClassifier.predict(numpy.array([nrow,]))
            targetSentId = sid - relativeSentPos[0]
            predictedTokens = [(tt.token, tt.tokenId) for tt in sid2tokens[targetSentId]]

            total += 1
            if targetSentId in set([x.sentenceId for x in extargtokens]):
                correct += 1
            predictedIds = [t[1] for t in predictedTokens]
            actualIds = [t.tokenId for t in actualtokens]
            predictedIds = [int(x) for x in predictedIds]
            actualIds = [int(x) for x in actualIds]
            
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

                    
    return correct / total, precision, recall, f1, set(), total #  TODO: acc here is for position classification, not for token ext args


def flipdict(tokenlist): # not actually a dict as input.  # and also adding ct.fullsent in this function

    d = defaultdict()
    sid2tokens = defaultdict(list)
    for ct in tokenlist:
        sid2tokens[ct.sentenceId].append(ct)
    
    for ct in tokenlist:
        d[int(ct.tokenId)] = ct

    return d, sid2tokens


def classifyExtArgs_deps_approach(testfiles, positionClassifier, sent2tagged, feat2ohvpos, feat2ohvlen, sameSentClassifier, sfeat2ohvpos, sfeat2ohvlen, nlp, outputset, samesentbaselinepred):

    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    total = 0
    correct = 0
    s_total = 0
    s_correct = 0
    tp = 0
    fp = 0
    fn = 0
    for fno, name in enumerate(testfiles):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))

        tokenlist, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        gtid2ct, sid2tokens = flipdict(tokenlist)

        for dr in discourseRelations:
            # discourseRelations contains tokenIds, instead of DiscourseToken objects. Change this at some point.
            conntokens = [gtid2ct[int(x)] for x in dr.connectiveTokens]
            intargtokens = [gtid2ct[int(x)] for x in dr.intArgTokens]
            extargtokens = [gtid2ct[int(x)] for x in dr.extArgTokens]
            c = ' '.join([dt.token for dt in conntokens])
            p = 0
            if set.intersection(set([dt.sentenceId for dt in conntokens]), set([dt.sentenceId for dt in extargtokens])): # if at least one token of conn and ext arg are in same sentence, assume same sentence scenario
                pass
            else:
                p = conntokens[0].sentenceId - extargtokens[0].sentenceId

            sid = conntokens[0].sentenceId
            actualtokens = extargtokens
            fullsent = conntokens[0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[conntokens[0].sentencePosition][1]
            nextpos = tagged[conntokens[len(conntokens)-1].sentencePosition + 1][1]

        ###_id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
            p_nrow = []
            s_nrow = []
            p_position_ohv = [0] * pfeat2ohvlen[3]
            p_position_ohv[pfeat2ohvpos[3][str(conntokens[0].sentencePosition)]] = 1
            s_position_ohv = [0] * sfeat2ohvlen[3]
            s_position_ohv[sfeat2ohvpos[3][str(conntokens[0].sentencePosition)]] = 1
            p_rootroute_ohv = [0] * pfeat2ohvlen[4]
            p_rootroute_ohv[pfeat2ohvpos[4][str(conntokens[0].pathToRoot)]] = 1
            s_rootroute_ohv = [0] * sfeat2ohvlen[4]
            s_rootroute_ohv[sfeat2ohvpos[4][str(conntokens[0].pathToRoot)]] = 1
            p_nrow += p_position_ohv
            s_nrow += s_position_ohv
            #nrow += postag_ohv
            p_nrow += p_rootroute_ohv
            s_nrow += s_rootroute_ohv

            dim = 300
            if c in embd:
                for item in embd[c]:
                    p_nrow.append(item)
                    s_nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    p_nrow.append(item)
                    s_nrow.append(item)
            #"""        
            if currpos in posembd:
                for item in posembd[currpos]:
                    p_nrow.append(item)
                    s_nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    p_nrow.append(item)
                    s_nrow.append(item)                    
            #"""
                    
            relativeSentPos = positionClassifier.predict(numpy.array([p_nrow,]))
            
            targetSentId = sid - relativeSentPos[0]
            refcon = conntokens[0]
            fullsent = refcon.fullSentence

            # do MV baseline, because this is going nowhere, either based on string or pos tag/syn cat:
            sameSentPosPrediction = 0
            if refcon.sentencePosition == 0: # this is also coded implicitly below
                sameSentPosPrediction = 0
            elif samesentbaselinepred[refcon.token] == 1:
                sameSentPosPrediction = 1
            s_total += 1
            sameSentPosActual = 0
            if extargtokens[0].tokenId > conntokens[0].tokenId:
                sameSentPosActual = 1
            if sameSentPosPrediction == sameSentPosActual:
                s_correct += 1

            

            predictedTokens = [(tt.token, tt.tokenId) for tt in sid2tokens[targetSentId]]

            if relativeSentPos[0] == 0 and p == 0: # if same sentence, take only tokens up to the connective
                refcon = conntokens[0]
                fullsent = refcon.fullSentence
                tree = nlp(fullsent)
                for index, token in enumerate(tree):
                    if index == refcon.sentencePosition and refcon.token == token.text: # not always the case, due to some tokenisation differences (spacy is not very configurable...)
                        if index == 0:
                            # get all parent of conn dep, and take the inverse of that
                            negativeOfPredictedSentIdExtArgTokens = [x.i for x in token.head.subtree]
                            predictedSentIdExtArgTokens = [y for y, x in enumerate(tree) if not y in negativeOfPredictedSentIdExtArgTokens]
                            predictedTokens = convertSentIdstoDocIds(tree, predictedSentIdExtArgTokens, refcon)
                        else:
                            # get all parent of conn dep, and take either left or right, depending on classifier prediction
                            
                            if sameSentPosPrediction == 0:
                                sortOfNegativeOfPredictedSentIdExtArgTokens = [x.i for x in token.head.subtree]
                                predictedSentIdExtArgTokens = [y for y, x in enumerate(tree) if y < refcon.sentencePosition]
                                predictedTokens = convertSentIdstoDocIds(tree, predictedSentIdExtArgTokens, refcon)
                            elif sameSentPosPrediction == 1:
                                sortOfNegativeOfPredictedSentIdExtArgTokens = [x.i for x in token.head.subtree]
                                predictedSentIdExtArgTokens = [y for y, x in enumerate(tree) if y > refcon.sentencePosition]
                                predictedTokens = convertSentIdstoDocIds(tree, predictedSentIdExtArgTokens, refcon)

            # section to print out non-adjacent ones. Basing this on actual tokens (not predicted ones) for now. Interesting would be to score only non-adjacent ones (get f score for this subset only).

            if intargtokens:
                if int(actualtokens[-1].tokenId) < int(conntokens[0].tokenId) - 1 and int(actualtokens[-1].tokenId) < int(intargtokens[0].tokenId) - 1:
                    outputset.add('connective:%s\nconn postag:%s\nintarg:%s\nextarg:%s\next-int\nin between tokens:%s' % (c, currpos, str([(t.token, t.tokenId) for t in intargtokens]), str([(t.token, t.tokenId) for t in actualtokens]), ' '.join([gtid2ct[x].token for x in range(int(actualtokens[-1].tokenId), int(intargtokens[0].tokenId))])))

                elif int(actualtokens[0].tokenId) > int(conntokens[-1].tokenId) + 1 and int(actualtokens[0].tokenId) > int(intargtokens[-1].tokenId) + 1:
                    outputset.add('connective:%s\nconn postag:%s\nintarg:%s\nextarg:%s\nint-ext\nin between tokens:%s' % (c, currpos, str([(t.token, t.tokenId) for t in intargtokens]), str([(t.token, t.tokenId) for t in actualtokens]), ' '.join([gtid2ct[x].token for x in range(int(intargtokens[-1].tokenId), int(actualtokens[0].tokenId))])))

                else:
                    pass
            #outputinstance = 'connective:%s\nconnective postag:%s\n' % (c, currpos, 
            
            total += 1
            if targetSentId in set([x.sentenceId for x in extargtokens]):
                correct += 1
            predictedIds = [t[1] for t in predictedTokens]
            actualIds = [t.tokenId for t in actualtokens]
            predictedIds = [int(x) for x in predictedIds]
            actualIds = [int(x) for x in actualIds]
            
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
    sys.stderr.write('p, r, f: %f, %f, %f\n' %  (precision, recall, f1))
                    
    return correct / total, precision, recall, f1, outputset, total #  TODO: acc here is for position classification, not for token ext args
            
                                

def convertSentIdstoDocIds(tree, predictedSentIdTokens, refcon): # NOTE: this one is not the same as in intArgExtraction. Since the conn is not syntactically integrated with the ext arg, have to loop through whole tree, as the conn will not be part of predictedSentIdTokens

    tempmap = {}
    for i, tsid in enumerate(tree):
        #print('i, tsid:', i, tsid)
        if i == refcon.sentencePosition:
            for j, tsid2 in enumerate(tree):
                if j < i:
                    tempmap[j] = int(refcon.tokenId) - (i - j)
                elif i == j:
                    tempmap[j] = int(refcon.tokenId)
                else:
                    tempmap[j] = int(refcon.tokenId) + (j - i)

    # since this dep structure can be discontinuous, and arguments in PCC never are, just include everything in the middle as well
    output = []
    #if predictedSentIdTokens:
        #for k in range(predictedSentIdTokens[0],predictedSentIdTokens[-1]):
            #output.append(tuple((tree[k].text, tempmap[k])))
    output = [tuple((tree[k].text, tempmap[k])) for k in predictedSentIdTokens]
    #print('output;', output)
    return output
            
    

def classifyExtArgs_const_approach(testfiles, positionClassifier, sent2tagged, feat2ohvpos, feat2ohvlen, sameSentClassifier, sfeat2ohvpos, sfeat2ohvlen, samesentbaselinepred):

    parsermemorymap = loadParserMemoryMap()
    
    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    p_total = 0
    p_correct = 0
    s_total = 0
    s_correct = 0
    tp = 0
    fp = 0
    fn = 0
    total = 0
    for fno, name in enumerate(testfiles):
        #sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
        tokenlist, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fileversions[name]['connectors'])
        tokenlist = PCCParser.parseSyntaxFile(fileversions[name]['syntax'], tokenlist)
        tokenlist = PCCParser.parseRSTFile(fileversions[name]['rst'], tokenlist)
        tokenlist = PCCParser.parseTokenizedFile(fileversions[name]['tokens'], tokenlist)
        file2tokenlist[name] = tokenlist
        file2discourseRelations[name] = PCCParser.discourseRelations
        gtid2ct, sid2tokens = flipdict(tokenlist)

        for dr in discourseRelations:
            total += 1
            # discourseRelations contains tokenIds, instead of DiscourseToken objects. Change this at some point.
            conntokens = [gtid2ct[int(x)] for x in dr.connectiveTokens]
            intargtokens = [gtid2ct[int(x)] for x in dr.intArgTokens]
            extargtokens = [gtid2ct[int(x)] for x in dr.extArgTokens]
            c = ' '.join([dt.token for dt in conntokens])
            p = 0
            if set.intersection(set([dt.sentenceId for dt in conntokens]), set([dt.sentenceId for dt in extargtokens])): # if at least one token of conn and ext arg are in same sentence, assume same sentence scenario
                pass
            else:
                p = conntokens[0].sentenceId - extargtokens[0].sentenceId

            sid = conntokens[0].sentenceId
            actualtokens = extargtokens
            fullsent = conntokens[0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[conntokens[0].sentencePosition][1]
            nextpos = tagged[conntokens[len(conntokens)-1].sentencePosition + 1][1]

            # this below fails silently if key not in dict (feat2ohvlen and feat2ohvpos are default dicts), in which case it sets the first elem to 1 (because val is 0 I think). Guess random would be better, but assuming that this happen that often, it's not so bad I guess.
            ###_id, tok, postag, sentencePosition, rootroute, sentenceId, label = row
            p_nrow = []
            s_nrow = []
            p_position_ohv = [0] * pfeat2ohvlen[3]
            p_position_ohv[pfeat2ohvpos[3][str(conntokens[0].sentencePosition)]] = 1
            s_position_ohv = [0] * sfeat2ohvlen[3]
            s_position_ohv[sfeat2ohvpos[3][str(conntokens[0].sentencePosition)]] = 1
            p_rootroute_ohv = [0] * pfeat2ohvlen[4]
            p_rootroute_ohv[pfeat2ohvpos[4][str(conntokens[0].pathToRoot)]] = 1
            s_rootroute_ohv = [0] * sfeat2ohvlen[4]
            s_rootroute_ohv[sfeat2ohvpos[4][str(conntokens[0].pathToRoot)]] = 1
            p_nrow += p_position_ohv
            s_nrow += s_position_ohv
            p_nrow += p_rootroute_ohv
            s_nrow += s_rootroute_ohv

            dim = 300
            if c in embd:
                for item in embd[c]:
                    p_nrow.append(item)
                    s_nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    p_nrow.append(item)
                    s_nrow.append(item)
            if currpos in posembd:
                for item in posembd[currpos]:
                    p_nrow.append(item)
                    s_nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    p_nrow.append(item)
                    s_nrow.append(item)                    
            relativeSentPos = positionClassifier.predict(numpy.array([p_nrow,]))
            targetSentId = sid - relativeSentPos[0]
            fullsent = conntokens[0].fullSentence
            refcon = conntokens[0]
            tree = parsermemorymap[fullsent]

            predictedTokens = [(tt.token, tt.tokenId) for tt in sid2tokens[targetSentId]]
            """
            print('conn:', c)
            print('intarg:', [(tt.token, tt.tokenId) for tt in intargtokens])
            print('actual tokens:', [(tt.token, tt.tokenId) for tt in extargtokens])
            print('predicted/actual sid:', targetSentId, sid)
            print('baseline tokens:', predictedTokens)
            """
            
            
            #sameSentPosPrediction = sameSentClassifier.predict(numpy.array([s_nrow,]))
            # do MV baseline, because this is going nowhere, either based on string or pos tag/syn cat:
            sameSentPosPrediction = 0
            if refcon.sentencePosition == 0: # this is also coded implicitly below
                sameSentPosPrediction = 0
            elif samesentbaselinepred[refcon.token] == 1:
                sameSentPosPrediction = 1
            s_total += 1
            sameSentPosActual = 0
            if extargtokens[0].tokenId > conntokens[0].tokenId:
                sameSentPosActual = 1
            if sameSentPosPrediction == sameSentPosActual:
                s_correct += 1
            
            if relativeSentPos[0] == 0 and p == 0: # if same sentence, take only tokens up to the connective
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

                if leaveno == 0: # take the following S
                    labels = ['S', 'CS', 'VP']
                    for i, node in enumerate(tree.pos()):
                        if i == leaveno:
                            nodePosition = tree.leaf_treeposition(i)
                            pt = ParentedTree.convert(tree)
                            pn = climb_tree(pt, nodePosition, labels)
                            rs = pn.right_sibling()
                            if rs:
                                # just take the rest of the sentence. May want to do something more fancy here (not using labels currently)
                                for st in pt.subtrees():
                                    if st == rs:
                                        right_remaining = right_siblings(st, [st])
                                        left_remaining = left_siblings(st, []) # leaving out node itself here on purpose
                                        right_leaves = []
                                        left_leaves = []
                                        for r_r in right_remaining:
                                            right_leaves += r_r.leaves()
                                        for l_r in left_remaining:
                                            left_leaves += l_r.leaves()
                                        predictedTokens = [tuple((x, int(sid2tokens[sid][0].tokenId) + y + len(left_leaves))) for y, x in enumerate(right_leaves)]
                                        # deleting first comma (annotation design decision)
                                        if predictedTokens[0][0] == ',':
                                            predictedTokens = predictedTokens[1:]
               

                else: # take next or previous S, depending on sameSentClassifier prediction
                    labels = ['S', 'CS', 'VP']
                    for i, node in enumerate(tree.pos()):
                        if i == leaveno:
                            nodePosition = tree.leaf_treeposition(max(0, i-1))
                            pt = ParentedTree.convert(tree)
                            children = pt[nodePosition[:1]]
                            labelnode = climb_tree(tree, nodePosition, labels)
                            left_remaining = left_siblings(labelnode, []) # leaving out node itself here on purpose
                            
                            firstwordpos = position_of_ngram(tuple(labelnode.leaves()), fullsent.split()) # not perfect, let's hope there aren't multiple matches
                            if not firstwordpos: # as a fallback, just take first occ in sent str. Probably labelnode is empty here too, should check...
                                firstwordpos = fullsent.split().index(c.split()[0])
                                if sameSentPosPrediction == 0:
                                    predictedTokens = [tuple((x, int(sid2tokens[sid][y].tokenId))) for y, x in enumerate(tree.leaves()) if y < firstwordpos]
                                elif sameSentPosPrediction == 1:
                                    predictedTokens = [tuple((x, int(sid2tokens[sid][y].tokenId))) for y, x in enumerate(tree.leaves()) if y > firstwordpos]
                            else:

                                if sameSentPosPrediction == 0:
                                    predictedTokens = [tuple((x, int(sid2tokens[sid][y].tokenId) + int(firstwordpos))) for y, x in enumerate(labelnode.leaves()) if y + firstwordpos < leaveno]
               
                                elif sameSentPosPrediction == 1:
                                    predictedTokens = [tuple((x, int(sid2tokens[sid][y].tokenId) + int(firstwordpos))) for y, x in enumerate(labelnode.leaves()) if y + firstwordpos > leaveno]
               
                                    
                                    

            if len(c.split()) > 1:

                first, last = conntokens[0], conntokens[-1]
                conntokenids = [ct.tokenId for ct in conntokens]
                if int(last.tokenId) - int(first.tokenId) > 1: # if it is potentially discontinuous. The if not ... condition in next list comprehension excludes thre-or-more-token connectives that are continuous (in combination with the if alt in the following line)
                    #alt2 = [tuple((sid2tokens[first.sentenceId][y].token, sid2tokens[last.sentenceId][y].tokenId)) for y in range(first.sentencePosition+1, last.sentencePosition)]
                    #print('deb alt2:', alt2)

                    #alt = [tuple((sid2tokens[sid][y].token, sid2tokens[sid][y].tokenId)) for y in range(first.sentencePosition+1, last.sentencePosition) if not sid2tokens[sid][y].tokenId in conntokenids]
                    alt = []
                    if last.sentenceId == first.sentenceId or int(last.sentenceId) - int(first.sentenceId) == 1:
                        for j in range(int(first.tokenId), int(last.tokenId)):
                            alt.append(tuple(('dummy', j))) # appending dummy because it is difficult to get actual token here, but comparison later on is done based on ids only anyway
                            predictedTokens = alt

                    else: # just take the entire sentence of the first token. TODO: check if first and last (i.e. rid2conndt) is something I have available at runtime, or if it is taken directly from the gold annotations...
                        predictedTokens = [(tt.token, tt.tokenId) for tt in sid2tokens[first.sentenceId]]
                    predictedTokens = [x for x in predictedTokens if not x[1] == first.tokenId and not x[1] == last.tokenId]
               
                    
            
            p_total += 1
            if targetSentId in set([x.sentenceId for x in extargtokens]):
                p_correct += 1
            predictedIds = [t[1] for t in predictedTokens]
            actualIds = [t.tokenId for t in actualtokens]
            # fucking python, without its type checking. took me way too much time to locate this inconsistency (tokenIds inside predicted were sometimes str, sometimes int)
            predictedIds = [int(x) for x in predictedIds]
            actualIds = [int(x) for x in actualIds]
            
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
    sys.stderr.write('p, r, f: %f, %f, %f\n' %  (precision, recall, f1))
    sys.stderr.write('pos, samesent accs: %f, %f\n' % (p_correct / p_total, s_correct / s_total))


    return p_correct / p_total, s_correct / s_total, precision, recall, f1, set(), total #  TODO: accs here are for position and samesent classification, not for token ext args

def left_siblings(st, l):

    if st.left_sibling():
        l.append(st.left_sibling())
        l = left_siblings(st.left_sibling(), l)
    return l

def right_siblings(st, l):

    if st.right_sibling():
        l.append(st.right_sibling())
        l = right_siblings(st.right_sibling(), l)
    return l


def climb_tree(tree, nodePosition, labels):
    
    pTree = ParentedTree.convert(tree)
    parent = pTree[nodePosition[:-1]].parent()
    if parent.label() in labels or parent.label() == 'ROOT': # second condition in case the label I'm looking for is not there
        return parent
    else:
        return climb_tree(tree, nodePosition[:-1], labels)

                    

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
    global posembd
    if DEBUG_MODE:
        embd = {}
        posembd = {}
        print('WARNING:NOT LOADING ANYTHING!!!')
    else:
        posembd = loadExternalEmbeddings('/home/peter/github/PhdPlayground/customEmbeddings/tiger_pos_model.vec.vec')
        embd = loadExternalEmbeddings('/home/peter/github/PhdPlayground/customEmbeddings/cc.de.300.PCC.vec')
    


if __name__ == '__main__':

    ##global DEBUG_MODE
    #DEBUG_MODE = True

    
    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/connectives-standoff/')
    #connectorfiles = PCCParser.getInputfiles('/share/corrected_connectives/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.1/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

    import pickle
    pfname = '/home/peter/github/PhdPlayground/picklejars/sent2tagged.pickle'
    
    sent2tagged = {}
    if os.path.exists(pfname):
        with codecs.open(pfname, 'rb') as handle:
            sent2tagged = pickle.load(handle)

    ### Section below is to train and eval only the position classifier
    

    goldpostags = False
    matrix, samesentmatrix, samesentbaselinedict, sent2tagged = getFeatureMatrix(fileversions, sent2tagged, goldpostags)
    # think the samesentbaselinedict always favors pos 0, but let's keep it dynamic anyway:
    samesentbaselinepred = defaultdict(int)
    for k in samesentbaselinedict:
        if samesentbaselinedict[k][1] > samesentbaselinedict[k][1]:
            samesentbaselinepred[k] = 1
        else:
            samesentbaselinepred[k] = 0

    
    classd = defaultdict(int)
    for row in matrix[1:]:
        classd[row[-1]] += 1
    #print('sent dist:', classd)
    #sent dist: defaultdict(<class 'int'>, {0: 66, 1: 435, 2: 603, 3: 6})
    #sent dist: defaultdict(<class 'int'>, {0: 603, 1: 435, 2: 47, 3: 8, 4: 6, 6: 4, 9: 1, -1: 6})

    #sys.exit()

    with open(pfname, 'wb') as handle:
        pickle.dump(sent2tagged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write('INFO: Tagged map pickled to "%s".\n' % pfname)

    
    
    # lines below are for only evaluating sentence position classifier. Will want to first do this, then get actual words and get p, r, f for that, but have to wrap all of that in x-fold cv
    numIterations = 10
    #a, p, r, f = evaluatePositionOnlyMVBaseline(matrix, numIterations)
    #print('%s accuracy: %f' % ('majority vote baseline', a))
    #print('%s precision: %f' % ('majority vote baseline', p))
    #print('%s recall: %f' % ('majority vote baseline', r))
    #print('%s f1: %f' % ('majority vote baseline', f))
    #sys.exit(1)


    #alg = 'randomforest' # hardcoded in subfunctions
    #a, p, r, f = evaluatePositionOnly(matrix, numIterations)
    #print('%s accuracy: %f' % (alg, a))
    #print('%s precision: %f' % (alg, p))
    #print('%s recall: %f' % (alg, r))
    #print('%s f1: %f' % (alg, f))

    #randomforest accuracy: 0.941964
    #randomforest precision: 0.940528
    #randomforest recall: 0.941964
    #randomforest f1: 0.940630
    

    #a = evaluatePositionOnlyNeural(matrix, numIterations)
    #print('%s accuracy: %f' % ('keras', a))
    #sys.exit(1)
    # REDO this with my definitive, final keras model architecture before citing it for some paper
    #keras accuracy: 0.945316
    #keras precision: 0.936295
    #keras recall: 0.942589
    #keras f1: 0.946714

    # evaluating the same sent before or after classifier, abusing the function name (since matrix shape is the same)
    #a = evaluatePositionOnlyNeural(samesentmatrix, numIterations)
    #print('%s accuracy: %f' % ('keras', a))
    #keras accuracy: 0.931967

    
    #### Here comes the section to evaluate the full thing (first detect position, then token span)

    #"""
    _a1 = []
    _a2 = []
    _p = []
    _r = []
    _f = []
    numIterations = 10
    loadEmbeddingsOnce()
    sys.stderr.write('INFO: Loading German spacy stuff...\n') # only needed for dep approach...
    #nlp = spacy.load('de')
    nlp = spacy.load('de_core_news_sm')
    sys.stderr.write('INFO: Done.\n')

    outputset = set()
    all_rids = 0
    for i in range(numIterations):
        sys.stderr.write('INFO: Starting iteration %i of %i...\n' % (i+1, numIterations))
        test, train = getTestAndTrainFiles(fileversions, numIterations, i)
        trainmatrix, trainsamesentmatrix, samesentbaselinedict, sent2tagged  = getFeatureMatrix(train, sent2tagged, goldpostags)
        positionClassifier, pfeat2ohvpos, pfeat2ohvlen = trainKerasPositionClassifier(trainmatrix)
        sameSentClassifier, sfeat2ohvpos, sfeat2ohvlen = trainKerasPositionClassifier(trainsamesentmatrix)
        
        #tp, fp, fn = classifyExtArgs(test, positionClassifier, sent2tagged, d, tp, fp, fn)
        #a, p, r, f, outputset, total_rids = classifyExtArgs_baseline_approach(test, positionClassifier, sent2tagged, pfeat2ohvpos, pfeat2ohvlen)
        #a, a2, p, r, f, outputset, total_rids = classifyExtArgs_const_approach(test, positionClassifier, sent2tagged, pfeat2ohvpos, pfeat2ohvlen, sameSentClassifier, sfeat2ohvpos, sfeat2ohvlen, samesentbaselinepred)
        a, p, r, f, outputset, total_rids = classifyExtArgs_deps_approach(test, positionClassifier, sent2tagged, pfeat2ohvpos, pfeat2ohvlen, sameSentClassifier, sfeat2ohvpos, sfeat2ohvlen, nlp, outputset, samesentbaselinepred)
        _a1.append(a)
        #_a2.append(a2)
        _p.append(p)
        _r.append(r)
        _f.append(f)
        all_rids += total_rids
    print('avg a1:', sum(_a1) / len(_a1))
    #print('avg a2:', sum(_a2) / len(_a2))
    print('avg p:', sum(_p) / len(_p))
    print('avg r:', sum(_r) / len(_r))
    print('avg f:', sum(_f) / len(_f))

    sys.stderr.write('(stderr duplicate) avg a1: %s\n' % str(sum(_a1) / len(_a1)))
    #sys.stderr.write('(stderr duplicate) avg a2: %s\n' % str(sum(_a2) / len(_a2)))
    sys.stderr.write('(stderr duplicate) avg p: %s\n' % str(sum(_p) / len(_p)))
    sys.stderr.write('(stderr duplicate) avg r: %s\n' % str(sum(_r) / len(_r)))
    sys.stderr.write('(stderr duplicate) avg f: %s\n' % str(sum(_f) / len(_f)))
                
    # highest score for const_app so far:
    #(stderr duplicate) avg a: 0.9559687337827686
    #(stderr duplicate) avg p: 0.6627479637935921
    #(stderr duplicate) avg r: 0.6236756276184285
    #(stderr duplicate) avg f: 0.6419034475838139


    print('nr of non adjacents:', len(outputset))
    print('among nr of total instances:', all_rids)
    for item in outputset:
        print(item)
        print()
    


    
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

