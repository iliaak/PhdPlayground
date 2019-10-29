#!/usr/bin/python3

import configparser
import sys
import os
import json
from nltk.parse import stanford
import LexConnClassifier

class Pypeline:

    def __init__(self):
        self.config = configparser.ConfigParser()
        if not os.path.exists(os.path.join(os.getcwd(), 'config.ini')):
            sys.stderr.write('ERROR: Config.ini not found.\n')
            sys.exit()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['stanfordParser']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['stanfordModels']
        os.environ['CLASSPATH'] = self.config['lexparser']['path']
        self.lexParser = stanford.StanfordParser(model_path=self.config['lexparser']['germanModel'])


if __name__ == '__main__':

    p = Pypeline()
    lcf = LexConnClassifier.LexConnClassifier(p)
    lcf.train(p) # optionally takes PCC file_ids list in case of x-fold cv training in pipeline setup. By default, it the entire PCC.

    _input = """
    Pass mal auf.
    Also wenn du schon von Schocken redest.
    Du bist hier der Typ der grade mit 'ner Socke redet.
    Die auf deiner Hand sitzt.
    Und mit deiner Stimme spricht.
    Und du behauptest die Behauptung dieser Socke stimmt nicht?
    """
    out = lcf.classify(p, _input)
    jsout = json.dumps(out, ensure_ascii=False, indent=2) # apparently json (not sure if in general, or just the python implementation) does not allow for int keys, which is why the token key ids are strings...
    print(jsout)
