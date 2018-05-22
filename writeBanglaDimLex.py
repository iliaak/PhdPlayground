from lxml import etree
import sys
import codecs
from optparse import OptionParser
import os
import re
import csv

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

    cd = csv.DictReader(codecs.open(options.connectivelist, 'r', 'utf-8'), delimiter='\t')
    root = etree.Element('dimlex')
    doc = etree.ElementTree(root)

    nodeid = 1
    for c in cd:
        conn = c['Connective Name'].strip()
        node = etree.Element('entry')
        node.set('id', 'c' + str(nodeid))
        nodeid += 1
        node.set('word', c['Connective Name'].strip())
        orths = etree.Element('orths')
        orth = etree.Element('orth')
        orth.attrib['type'] = 'cont' if not re.search('\s', conn) else 'TODO:cont_discont'
        orth.attrib['canonical'] = '1'
        orth.attrib['onr'] = str(nodeid) + 'o1'
        part = etree.Element('part')
        part.attrib['type'] = 'single' if not re.search('\s', conn) else 'phrasal'
        part.text = c['Connective Name'].strip()
        orth.append(part)
        orths.append(orth)
        node.append(orths)
        sttsnode = etree.Element('stts')
        ex = etree.Element('example')
        ex.text = c['Connective use Example'].strip()
        sttsnode.append(ex)
        node.append(sttsnode)
        synnode = etree.Element('syn')
        catnode = etree.Element('cat')
        catnode.text = c['Parts of Speech'].strip()
        synnode.append(catnode)
        semnode = etree.Element('sem')
        sensenode = etree.Element('pdtb3_relation')
        sensenode.set('sense', re.sub(' -> ', ':', c['Sense (PDTB-3)']).strip())
        semnode.append(sensenode)
        synnode.append(semnode)
        node.append(synnode)
        
        root.append(node)
        
    doc.write(options.outputxml, xml_declaration=True, encoding='utf-8', pretty_print=True)
    
    # since this is for using in conano, have to transform using the xslt sheet:
    conanooutput = 'conano_'+options.outputxml
    xslt = etree.parse('/home/peter/dimlex/dimlex2conano.xsl')
    transform = etree.XSLT(xslt)
    newdoc = transform(doc)
    for entry in newdoc.getroot():
        for node in entry:
            if node.tag == 'orth':
                node.attrib.pop('canonical', None)
                node.attrib.pop('onr', None)
            
            if node.tag == 'stts':
                entry.remove(node)

    newdoc.write(conanooutput, xml_declaration=True, encoding='utf-8', pretty_print=True)
