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
import numpy
from sklearn import svm
import sys
from nltk.tag import stanford
#from nltk.tag.corenlp import CoreNLPPOSTagger

# this is all done a bit haphazardly, seeing if using postagbigrams improves performance. If it does, do this properly, probably by taking the parse trees that I already generated for connective identification. I'm hoping/assuming here that the parser will assign the same/very similar postags as the pos tagger does.

# Add the jar and model via their path (instead of setting environment variables):
jar = '/home/peter/phd/PhdPlayground/stanfordStuff/stanford-postagger-full-2018-02-27/stanford-postagger.jar'
model = '/home/peter/phd/PhdPlayground/stanfordStuff/stanford-postagger-full-2018-02-27/models/german-fast.tagger'

pos_tagger = stanford.StanfordPOSTagger(model, jar, encoding='utf8')

#headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'class_label'
"""
pos2column = {
    0:'connective',
    1:'pos',
    2:'sentencePosition',
    3:'pathToRoot',
    4:'class_label'
}

"""
pos2column = {}

def getNumbers(test_features, test_labels, results, d):

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


def getDataSplits(numIterations, dataSize):

    p = int(dataSize / 10)
    pl = [int(x) for x in range(0, dataSize, p)]
    pl.append(int(dataSize))    
    return pl

def positionMapping(pos):

    if pos == 0:
        return pos
    elif pos > 0 and pos < 5:
        return 1
    elif pos >= 5 and pos < 10:
        return 2
    else:
        return 3


def evaluate(csvmatrix, numIterations, algorithm):

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
                
        testdf = pandas.DataFrame(testrows) # a bit redundant, going back and forth here...
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

        from sklearn.utils import class_weight
        weights = class_weight.compute_class_weight('balanced', numpy.unique(train_labels), train_labels)
        #cwd = {-2: weights[0], -1:weights[1], 0:weights[2], 1:weights[3]}

        classifier = None
        if algorithm == 'decisiontree':
            classifier = DecisionTreeClassifier(random_state=0)
        elif algorithm == 'logisticregression':
            classifier = LogisticRegression()
        elif algorithm == 'randomforest':
            #classifier = RandomForestClassifier(class_weight=cwd, n_estimators=1000)
            classifier = RandomForestClassifier(n_estimators=100)
        elif algorithm == 'svm':
            classifier = svm.SVC(C=1, kernel='poly') # try with linear for pcc, takes a loooong time... # poly (for kernel)
        elif algorithm == 'mlp':
            classifier = MLPClassifier()
        elif algorithm == 'knn':
            classifier = KNeighborsClassifier()
        elif algorithm == 'gaussiannb':
            classifier = GaussianNB()
        elif algorithm == 'extratree':
            classifier = ExtraTreeClassifier()
        elif algorithm == 'sgd':
            classifier = SGDClassifier()
        else:
            sys.stderr.write('ERROR: Algorithm "%s" not supported. Dying now.\n' % algorithm)
    
            
        classifier.fit(train_features, train_labels)

        results = classifier.predict(test_features)
        
        a, p, r, f = getNumbers(test_features, test_labels, results, d)
        #print('a:', a)
        #print('p:', p)
        #print('r:', r)
        #print('f:', f)
        _as.append(a)
        _ps.append(p)
        _rs.append(r)
        _fs.append(f)
        
    macroPrecision = sum(_ps)/len(_ps)
    macroRecall = sum(_rs)/len(_rs)
    macroF1 = sum(_fs)/len(_fs)
    acc = sum(_as)/len(_as)
    return acc, macroPrecision, macroRecall, macroF1




if __name__ == '__main__':

    # WARNING: This code starts to become a real mess. Once the idea is clear, some serious refactoring has to be done.

    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/connectors/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

    connectorfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/connectors/')
    syntaxfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/syntax/')
    rstfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/rst/')
    tokenfiles = PCCParser.getInputfiles('/share/potsdam-commentary-corpus-2.0.0/potsdam-commentary-corpus-2.0.0/tokenized/')
    
    fileversions = PCCParser.getFileVersionsDict(connectorfiles, syntaxfiles, rstfiles, tokenfiles)

    someId2extArgPosition = defaultdict(int)
    someId2connective = defaultdict(str)

    sameSentenceCases = 0
    anyOfTheFollowingSentencesCases = 0
    previousSentenceCases = 0
    anyOfThePrePreviousSentenceCases = 0

    matrix = []
    headers = ['id', 'connective', 'pos', 'sentencePosition', 'pathToRoot', 'sentenceId', 'class_label']
    for i2, j2 in enumerate(headers):
        if i2 > 0:
            pos2column[i2-1] = j2
    matrix.append(headers)
    mid = 1
    connectiveSingleTokens = 0

    import pickle
    pfname = 'sent2tagged.pickle'
    
    sent2tagged = {}
    if os.path.exists(pfname):
        with codecs.open(pfname, 'rb') as handle:
            sent2tagged = pickle.load(handle)
    
    file2tokenlist = defaultdict(list)
    file2discourseRelations = defaultdict(list)
    for fno, name in enumerate(fileversions):
        sys.stderr.write('Processing file %s (%i of %i).\n' % (name, fno+1, len(fileversions)))
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
                sameSentenceCases += 1
                someId2extArgPosition[name + '_' + str(rid)] = 0
                p = 0
            elif sorted(rid2extargsentid[rid])[0] - sorted(rid2connsentid[rid])[0] > 0:
                anyOfTheFollowingSentencesCases += 1
                someId2extArgPosition[name + '_' + str(rid)] = 1
                p = 1
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] == 1:
                previousSentenceCases += 1
                someId2extArgPosition[name + '_' + str(rid)] = -1
                p = -1
            elif sorted(rid2connsentid[rid])[0] - sorted(rid2extargsentid[rid])[0] > 1:
                anyOfThePrePreviousSentenceCases += 1
                # note: chaning -2 to -1 here to throw together PS and prePS cases!
                someId2extArgPosition[name + '_' + str(rid)] = -1
                p = -1
            # features: conn (surface form), part-of-speech of first one, sentence position of first one, path to root of first one
            #row = [str(mid), c, rid2conndt[rid][0].pos, positionMapping(rid2conndt[rid][0].sentencePosition), rid2conndt[rid][0].pathToRoot, p]
            # getting postag of right neighbour here. If this improves performance, do this properly (pre-tagging entire corpus only once, not doing it on the fly here)
            fullsent = rid2conndt[rid][0].fullSentence
            tagged = None
            if fullsent in sent2tagged:
                tagged = sent2tagged[fullsent]
            else:
                tagged = pos_tagger.tag(fullsent.split()) # assuming tokenization
                sent2tagged[fullsent] = tagged
            currpos = tagged[rid2conndt[rid][0].sentencePosition][1]
            nextpos = tagged[rid2conndt[rid][len(c.split())-1].sentencePosition + 1][1]
            
            #if p == 0 or p == -1: # discarding FS cases
            #row = [str(mid), c, rid2conndt[rid][0].pos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, rid2conndt[rid][0].sentenceId, p]
            row = [str(mid), c, currpos, rid2conndt[rid][0].sentencePosition, rid2conndt[rid][0].pathToRoot, rid2conndt[rid][0].sentenceId, p]
            mid += 1
            matrix.append(row)

            #if not len(rid2extargsentid[rid]):
                #print('DEBUGGING no ext arg case:', name, rid, rid2conn[rid])

    #print('connective tokens (single words):', connectiveSingleTokens)
    #print('DYING HERE DUE TO SYS EXIT')
    #sys.exit(0)
    with open(pfname, 'wb') as handle:
        pickle.dump(sent2tagged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('tagged map pickled to %s' % pfname)

    
    total = sameSentenceCases + anyOfTheFollowingSentencesCases + previousSentenceCases + anyOfThePrePreviousSentenceCases
    
    print('ss cases: %i / %f' % (sameSentenceCases, sameSentenceCases/total))
    print('fs cases: %i / %f' % (anyOfTheFollowingSentencesCases, anyOfTheFollowingSentencesCases/total))
    print('ps cases: %i / %f' % (previousSentenceCases, previousSentenceCases/total))
    print('pre-ps cases: %i / %f' % (anyOfThePrePreviousSentenceCases, anyOfThePrePreviousSentenceCases/total))

    # could also have done this counting above already...
    conn2fs = defaultdict(int)
    conn2ss = defaultdict(int)
    conn2ps = defaultdict(int)
    conn2pps = defaultdict(int)
    allconns = set()
    
    for sid in someId2extArgPosition:
        conn = someId2connective[sid]
        allconns.add(conn)
        pos = someId2extArgPosition[sid]
        if pos == 1:
            conn2fs[conn] += 1
        elif pos == 0:
            conn2ss[conn] += 1
        elif pos == -1:
            conn2ps[conn] += 1
        elif pos == -2:
            conn2pps[conn] += 1
    conn2majorityVote = defaultdict(int)
    for c in allconns:
        m = max(conn2fs[c], conn2ss[c], conn2ps[c], conn2pps[c])
        if m == conn2fs[c]: # the ordering of if elses here means that if there is a tie between ss and ps, ss gets precedence... Completely arbitrary decision I guess
            conn2majorityVote[c] = 1
        elif m == conn2ss[c]:
            conn2majorityVote[c] = 0
        elif m == conn2ps[c]:
            conn2majorityVote[c] = -1
        else:
            conn2majorityVote[c] = -2
    #"""
    total = 0
    correct = 0

    for sid in someId2extArgPosition:
        total += 1
        c = someId2connective[sid]
        realposition = someId2extArgPosition[sid]
        majorityvoteposition = conn2majorityVote[c]
        if realposition == majorityvoteposition:
            correct += 1
    print('majority vote total/correct/accuracy:', total, correct, correct/total)
    #"""
    
    # now let's try to build a classifier with some features; first one coming to mind is position in sentence (if initial, extArg is likely to be prevsent), although this i sreflected by casing difference already, so may not really add anything.

    #algs = ['decisiontree', 'logisticregression', 'randomforest', 'mlp', 'knn', 'gaussiannb', 'extratree', 'sgd']
    algs = ['randomforest']
    numIterations = 10
    for alg in algs:
        a, p, r, f = evaluate(matrix, numIterations, alg)
        print('%s accuracy: %f' % (alg, a))
        print('%s precision: %f' % (alg, p))
        print('%s recall: %f' % (alg, r))
        print('%s f1: %f' % (alg, f))

    # conclusion; with a simple feature set (connective, pos, sentence position, path to root), only the randomforest classifier performs slightly better than mv baseline (ca. 95 for randomforest compared to 92 for mv baseline). All other algrithms perform from slightly to considerably worse.


    #########################################################################################################
    sys.exit(0)
    # attempt at getting score for intArg simple baseline
    for fname in file2tokenlist:
        tokenlist = file2tokenlist[fname]
        drs = file2discourseRelations[fname]
        
        for drId in drs:
            dr = drs[drId]
            print('drId:', drId)
            print('conn:', ' '.join([x.token for x in dr.connectiveTokens]))
            print('int:', ' '.join([x.token for x in dr.intArgTokens]))
            print('ext:', ' '.join([x.token for x in dr.extArgTokens]))
            #continue here. The above does not work, the discourseRelation dict is not properly filled
