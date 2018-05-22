from lxml import etree
import sys
import codecs
from optparse import OptionParser
import os
import re

if __name__ == '__main__':
   
    parser = OptionParser("usage: %prog corpus")
    parser.add_option("-l", "--list", dest="connectivelist", help="specify path to a file containing a list of connectives (one per line)...")
    parser.add_option('-o', '--output', dest='outputxml', help='specify the name of the output xml file to write')

    options, args = parser.parse_args()
    
    if not options.connectivelist or not options.outputxml:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if not os.path.isfile(options.connectivelist):
        sys.stderr.write('ERROR: File %s could not be found.\n' % options.connectivelist)
        sys.exit(1)
        
    connectives = [x.strip() for x in codecs.open(options.connectivelist, 'r').readlines()]

    root = etree.Element('dimlex')
    doc = etree.ElementTree(root)

    _id = 1
    for conn in connectives:
        connnode = etree.Element('entry')
        nodeId = 'c' + str(_id)
        connnode.attrib['id'] = nodeId
        _id += 1
        connnode.attrib['word'] = conn

        # orths
        orths = etree.Element('orths')
        orth = etree.Element('orth')
        orth.attrib['type'] = 'cont' if not re.search('\s', conn) else 'TODO:cont_discont'
        orth.attrib['canonical'] = '1'
        orth.attrib['onr'] = nodeId + 'o1'
        part = etree.Element('part')
        part.attrib['type'] = 'single' if not re.search('\s', conn) else 'phrasal'
        part.text = conn
        orth.append(part)
        orths.append(orth)
        orth2 = etree.Element('orth')
        orth2.attrib['type'] = 'cont' if not re.search('\s', conn) else 'TODO:cont_discont'
        orth2.attrib['canonical'] = '0'
        orth2.attrib['onr'] = nodeId + 'o2'
        part2 = etree.Element('part')
        part2.attrib['type'] = 'single' if not re.search('\s', conn) else 'phrasal'
        part2.text = conn.capitalize()
        orth2.append(part2)
        orths.append(orth2)
        connnode.append(orths)

        # ambiguity
        ambiguity = etree.Element('ambiguity')
        sem_ambig = etree.Element('sem_ambiguity')
        sem_ambig.text = '0_1'
        ambiguity.append(sem_ambig)
        # don't have numbers on conn and non_conn frequency...
        connnode.append(ambiguity)

        # focuspart
        focuspart = etree.Element('focuspart')
        focuspart.text = 'TODO'
        connnode.append(focuspart)

        # non_conn_reading
        non_conn_reading = etree.Element('non_conn_reading')
        example1 = etree.Element('example')
        example1.attrib['type'] = ''
        example1.attrib['tfreq'] = ''
        example1.text = 'TODO:example_here_if_applicable'
        #non_conn_reading.text = '0_1'
        non_conn_reading.append(example1)
        connnode.append(non_conn_reading)

        # stts
        stts = etree.Element('stts')
        example2 = etree.Element('example')
        example2.attrib['type'] = ''
        example2.attrib['tfreq'] = ''
        example2.text = 'TODO'
        stts.append(example2)
        connnode.append(stts)

        # syn
        syn = etree.Element('syn')
        cat = etree.Element('cat')
        cat.text = 'syncat_here'
        syn.append(cat)
        integr = etree.Element('integr')
        syn.append(integr)
        ordering = etree.Element('ordering')
        ante = etree.Element('ante')
        ante.text = 'TODO'
        ordering.append(ante)
        post = etree.Element('post')
        post.text = 'TODO'
        ordering.append(post)
        insert = etree.Element('insert')
        insert.text = 'TODO'
        ordering.append(insert)
        syn.append(ordering)
        sem = etree.Element('sem')
        pdtb3_relation = etree.Element('pdtb3_relation')
        pdtb3_relation.attrib['sense'] = 'TODO'
        # no frequency info for the senses
        sem.append(pdtb3_relation)
        syn.append(sem)
        connnode.append(syn)

        root.append(connnode)


    outFile = options.outputxml
    doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True) 
    
