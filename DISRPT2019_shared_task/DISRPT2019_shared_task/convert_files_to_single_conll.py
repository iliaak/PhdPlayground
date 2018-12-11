import os
import sys
import codecs
import re
import shutil

def getInputfiles(infolder):

    filelist = []
    for f in os.listdir(infolder):
        abspathFile = os.path.abspath(os.path.join(infolder, f))
        filelist.append(abspathFile)
    return filelist

def convert(i, o):

    for j, f in enumerate(i):
        #print('Processing file:', f)
        o.write('# newdoc id = %s\n' % re.sub('\.segmented\.conll', '', os.path.basename(f)))
        lastId = 0
        for line in codecs.open(f).readlines():
            #print('debugging l:', line)
            if re.search('\t', line):
                if int(line.split('\t')[0]) == lastId:
                    o.write('\n')
                lastId = int(line.split('\t')[0])
            if re.search('B-Segment$', line):
                p = line.split('\t')
                if p[-2] == '_':
                    p2 = p[:-2]
                    p2.append('BeginSeg=Yes\n')
                    o.write('\t'.join(p2))
                    #print('\t'.join(p2))
                else:
                    p2 = p[:-1]
                    p2[-1] = p2[-1]+'|'+'BeginSeg=Yes\n'
                    o.write('\t'.join(p2))                            
            else:
                o.write(line)
        if j < len(i)-1:
            o.write('\n')

def validate(f):

    for line in codecs.open(f).readlines():
        if re.search('\t', line):
            if not len(line.split('\t')) == 10:
                print('ERROR AT LINE:', line)
                print(line.split('\t'))
                print('WRONG NUMBER OF COLUMNS! (%i)' % len(line.split('\t')))
                print('File:', f)
            #else:
                #print(line.split('\t'))

def removeDoubleNewlines(f):

    df = codecs.open('new.conll', 'w')
    changed = False
    lines = codecs.open(f).readlines()
    newlines = []
    for i, line in enumerate(lines):
        if i > 0:
            if re.match('^$', line):
                if re.match('^$', lines[i-1]):
                    pass
                else:
                    newlines.append(line)
            else:
                newlines.append(line)
        else:
            newlines.append(line)
    for nl in newlines:
        df.write(nl)
    df.close()
    shutil.move('new.conll', f)
                
if __name__ == '__main__':

    #"""
    devfiles = getInputfiles('dev/out')
    devout = codecs.open('dev.conll', 'w')
    convert(devfiles, devout)
    devout.close()
    removeDoubleNewlines('dev.conll')
    validate('dev.conll')
    #"""
    #"""
    testfiles = getInputfiles('test/out')
    testout = codecs.open('test.conll', 'w')
    convert(testfiles, testout)
    testout.close()
    removeDoubleNewlines('test.conll')
    validate('test.conll')
    #"""

    #"""
    trainfiles = getInputfiles('train/out')
    trainout = codecs.open('train.conll', 'w')
    convert(trainfiles, trainout)
    trainout.close()
    removeDoubleNewlines('train.conll')
    validate('train.conll')
    #"""
