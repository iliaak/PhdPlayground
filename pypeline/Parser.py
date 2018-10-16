#!/usr/bin/python3
import ConnectiveClassifier
import ArgumentExtractor
import configparser
import sys
import time
import codecs
import os
from nltk.parse import stanford
from nltk.tree import ParentedTree
import numpy
import tensorflow as tf

import ConnectiveClassifier
import ArgumentExtractor


class Parser:

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['stanfordParser']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['stanfordModels']
        os.environ['CLASSPATH'] = self.config['lexparser']['path']
        self.lexParser = stanford.StanfordParser(model_path=self.config['lexparser']['germanModel'])
        

    def loadEmbeddings(self, debugflag=False):

        self.embd = {}
        self.posembd = {}

        if not debugflag:
            starttime = time.time()
            wordembfile = self.config['embeddings']['wordembeddings']
            sys.stdout.write('INFO: Loading external embeddings from %s.\n' % wordembfile)
            with codecs.open(wordembfile, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    values = line.split()
                    self.embd[values[0]] = numpy.array([float(x) for x in values[1:]])
            endtime = time.time()
            sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))

        if not debugflag:
            starttime = time.time()
            posembfile = self.config['embeddings']['posembeddings']
            sys.stdout.write('INFO: Loading external embeddings from %s.\n' % posembfile)
            with codecs.open(posembfile, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    values = line.split()
                    self.posembd[values[0]] = numpy.array([float(x) for x in values[1:]])
            endtime = time.time()
            sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))

        if debugflag:
            sys.stderr.write('WARNING: Embedding loading skipped (debug mode)\n')


    def preParse(self, sentences):
        runtimeparsermemory = {}
        for sentence in sentences:
            tokens = sentence.split()
            ptree = None
            tree = self.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break # always taking the first, assuming that this is the best scoring tree.
            runtimeparsermemory[sentence] = ptree
            
        return runtimeparsermemory


if __name__ == '__main__':


    ### training part
    graph = tf.get_default_graph() 
    parser = Parser()
    parser.loadEmbeddings(True)
    cc = ConnectiveClassifier.ConnectiveClassifier()
    #cc.setGraph()
    cc.train(parser, True)
    arg = ArgumentExtractor.ArgumentExtractor()
    arg.setGraph()
    arg.train(parser, True)
    
    
    sentences = ['Auf Grund der dramatischen Kassenlage in Brandenburg hat sie jetzt eine seit mehr als einem Jahr erarbeitete Kabinettsvorlage überraschend auf Eis gelegt und vorgeschlagen , erst 2003 darüber zu entscheiden .',
                 'Überraschend , weil das Finanz- und das Bildungsressort das Lehrerpersonalkonzept gemeinsam entwickelt hatten .',
                 'Der Rückzieher der Finanzministerin ist aber verständlich .',
                 'Es dürfte derzeit schwer zu vermitteln sein , weshalb ein Ressort pauschal von künftigen Einsparungen ausgenommen werden soll auf Kosten der anderen .',
                 'Reiches Ministerkollegen werden mit Argusaugen darüber wachen , dass das Konzept wasserdicht ist .', 'Tatsächlich gibt es noch etliche offene Fragen .',
                 'So ist etwa unklar , wer Abfindungen erhalten soll , oder was passiert , wenn zu wenig Lehrer die Angebote des vorzeitigen Ausstiegs nutzen .',
                 'Dennoch gibt es zu Reiches Personalpapier eigentlich keine Alternative .',
                 'Das Land hat künftig zu wenig Arbeit für zu viele Pädagogen .',
                 'Und die Zeit drängt .',
                 'Der große Einbruch der Schülerzahlen an den weiterführenden Schulen beginnt bereits im Herbst 2003 .',
                 'Die Regierung muss sich entscheiden , und zwar schnell .',
                 'Entweder sparen um jeden Preis oder Priorität für die Bildung .',
                 'Es regnet Hunde und Katze .',
                 'Und was soll ich machen ?',
                 'Weil es regnet , bleiben wir zu Hause .',
                 'Es regnet , aber wir bleiben zu Hause .'
    ]

    runtimeparsermemory = parser.preParse(sentences)
    connectivepositions = cc.run(parser, sentences, runtimeparsermemory)
    #connectivepositions = [(14, [0]), (15, [0]), (16, [3])]
    relations = arg.run(parser, sentences, runtimeparsermemory, connectivepositions)
    for rid in relations:
        print('rid:', rid)
        conn = relations[rid]['connective']
        intarg = relations[rid]['intarg']
        extarg = relations[rid]['extarg']
        print('\tconn:', conn, sentences[conn[0]].split()[conn[1][0]:conn[1][-1]+1])
        print('\tint:', intarg, sentences[intarg[0]].split()[intarg[1][0]:intarg[1][-1]+1])
        if extarg:
            if extarg[1]:
                print('\text:', extarg, sentences[extarg[0]].split()[extarg[1][0]:extarg[1][-1]+1])

