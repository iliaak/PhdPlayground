#!/usr/bin/python3
import ConnectiveClassifier
import ArgumentExtractor
import sys

#cc = ConnectiveClassifier.ConnectiveClassifier()
#cc.train()

#sys.exit()
#cc.loadClassifier()


sentences = ['Auf Grund der dramatischen Kassenlage in Brandenburg hat sie jetzt eine seit mehr als einem Jahr erarbeitete Kabinettsvorlage überraschend auf Eis gelegt und vorgeschlagen , erst 2003 darüber zu entscheiden .', 'Überraschend , weil das Finanz- und das Bildungsressort das Lehrerpersonalkonzept gemeinsam entwickelt hatten .', 'Der Rückzieher der Finanzministerin ist aber verständlich .', 'Es dürfte derzeit schwer zu vermitteln sein , weshalb ein Ressort pauschal von künftigen Einsparungen ausgenommen werden soll auf Kosten der anderen .', 'Reiches Ministerkollegen werden mit Argusaugen darüber wachen , dass das Konzept wasserdicht ist .', 'Tatsächlich gibt es noch etliche offene Fragen .', 'So ist etwa unklar , wer Abfindungen erhalten soll , oder was passiert , wenn zu wenig Lehrer die Angebote des vorzeitigen Ausstiegs nutzen .', 'Dennoch gibt es zu Reiches Personalpapier eigentlich keine Alternative .', 'Das Land hat künftig zu wenig Arbeit für zu viele Pädagogen .', 'Und die Zeit drängt .', 'Der große Einbruch der Schülerzahlen an den weiterführenden Schulen beginnt bereits im Herbst 2003 .', 'Die Regierung muss sich entscheiden , und zwar schnell .', 'Entweder sparen um jeden Preis oder Priorität für die Bildung .', 'Es regnet Hunde und Katze .']

#connectivepositions = cc.run(sentences)

#print('debugging conn positions:', connectivepositions)

# move on to arguments
arg = ArgumentExtractor.ArgumentExtractor()
#arg.loadEmbeddings()
#arg.trainPositionClassifiers()

arg.loadClassifiers()


print('Done.')
