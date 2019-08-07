#!/usr/bin/python3
from flask import Flask
from flask import request
from flask_cors import CORS
import re
import time
from collections import defaultdict

import Parser
import ConnectiveClassifier
import ArgumentExtractor

from nltk import word_tokenize

app = Flask(__name__)
CORS(app)

parser = None
cc = None
arg = None

# export FLASK_APP=flaskController.py 
# nohup python3 -m flask run &

def getTokenised(sentences): # perhaps slightly out of place here (better to put in utils or sth, but here I can ensure tokenisation is the same during processing and painting)
    tokenised = []
    for sent in sentences:
        tokenised.append(' '.join(word_tokenize(sent)))
    return tokenised

@app.route('/train', methods=['GET'])
def train():

    starttime = time.time()
    global parser, cc, arg

    parser = Parser.Parser()
    #parser.loadEmbeddings(True)
    parser.loadEmbeddings()
    cc = ConnectiveClassifier.ConnectiveClassifier()
    #cc.setGraph() # not needed anymore with old-fashioned RandomForest Classifier
    #cc.train(parser, True)
    cc.train(parser)
    arg = ArgumentExtractor.ArgumentExtractor()
    arg.setGraph()
    #arg.train(parser, True)
    arg.train(parser)

    endtime = time.time()
   
    return 'Succesfully initialized and trained models, took %s seconds.\n' % (str(endtime - starttime))

@app.route('/parse', methods=['GET'])
def parse():

    if not parser or not cc or not arg:
        return 'Please use the train endpoint first to initialize models.\n'

    if request.args.get('input') == None:
        return 'Please provide an input text.\n'

    #sentences = ['Auf Grund der dramatischen Kassenlage in Brandenburg hat sie jetzt eine seit mehr als einem Jahr erarbeitete Kabinettsvorlage überraschend auf Eis gelegt und vorgeschlagen , erst 2003 darüber zu entscheiden .','Überraschend , weil das Finanz- und das Bildungsressort das Lehrerpersonalkonzept gemeinsam entwickelt hatten .','Der Rückzieher der Finanzministerin ist aber verständlich .','Es dürfte derzeit schwer zu vermitteln sein , weshalb ein Ressort pauschal von künftigen Einsparungen ausgenommen werden soll auf Kosten der anderen .','Reiches Ministerkollegen werden mit Argusaugen darüber wachen , dass das Konzept wasserdicht ist .', 'Tatsächlich gibt es noch etliche offene Fragen .','So ist etwa unklar , wer Abfindungen erhalten soll , oder was passiert , wenn zu wenig Lehrer die Angebote des vorzeitigen Ausstiegs nutzen .','Dennoch gibt es zu Reiches Personalpapier eigentlich keine Alternative .','Das Land hat künftig zu wenig Arbeit für zu viele Pädagogen .','Und die Zeit drängt .','Der große Einbruch der Schülerzahlen an den weiterführenden Schulen beginnt bereits im Herbst 2003 .','Die Regierung muss sich entscheiden , und zwar schnell .','Entweder sparen um jeden Preis oder Priorität für die Bildung .','Es regnet Hunde und Katze .','Und was soll ich machen ?','Weil es regnet , bleiben wir zu Hause .','Es regnet , aber wir bleiben zu Hause .', 'Und wie geht es weiter ?']

    sentences = [x.strip() for x in request.args.get('input').split('___NEWLINE___')]
    sentences = [x for x in sentences if x] # take out empty sents
    # pre-tokenize here. Important that this is in line with tokenization when the painting happens below
    print('debugging sentences:', sentences)
    sentences = getTokenised(sentences)
    print('debugging tokenised sentences:', sentences)
    runtimeparsermemory = parser.preParse(sentences)
    #print('runtimeparsermemory:', runtimeparsermemory)

    connectivepositions = cc.run(parser, sentences, runtimeparsermemory)
    print('debugging connectivepositions:', connectivepositions)
    #connectivepositions = [(14, [0]), (15, [0]), (16, [3]), (17, [0])]
    relations = arg.run(parser, sentences, runtimeparsermemory, connectivepositions)
    
    # the following results in heavy painting (every token its own markup), but otherwise getting the overlapping spans right is a lot of code, and this is for demo purposes only anyway
    paintedOutput = [x.split() for x in sentences]
    for rid in relations:
        conn = relations[rid]['connective']
        intarg = relations[rid]['intarg']
        extarg = relations[rid]['extarg']
        print('rid:', rid)
        print('\tconn:', conn)
        print('\tintarg:', intarg)
        print('\textarg:', extarg)
        
        for i in conn[1]:
            paintedOutput[conn[0]][i] = '<connRef id=%s class="badge badge-warning">' % str(rid) + paintedOutput[conn[0]][i] + '</connRef id=%s>' % str(rid)
        for i in intarg[1]:
            paintedOutput[intarg[0]][i] = '<intargRef id=%s class="badge badge-success">' % str(rid) + paintedOutput[intarg[0]][i] + '</intargRef id=%s>' % str(rid) # bold face?
        if extarg:
            if extarg[1]:
                for i in extarg[1]:
                    paintedOutput[extarg[0]][i] = '<extargRef id=%s class="badge badge-primary">' % str(rid) + paintedOutput[extarg[0]][i] + '</extargRef id=%s>' % str(rid) # italics?
        
    painted = []
    for sent in paintedOutput:
        painted.append(' '.join(sent))
    ret = '<br/>'.join(painted)
    print('debug raw response:', ret)

    return ret

if __name__ == '__main__':

    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port,debug=True)
