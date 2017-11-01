import sys, getopt
from sys import argv
import os

#Need to handle input parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"",["help","ipath=","itagfile=","otagfile=","encoding="])
except getopt.GetoptError:
    print('Error specifying parameters. Try python Extract.py -h for help')
    sys.exit(2)

inpath=''
itagfile=""
otagfile=''
iencoding="utf-8"
#
#Refer to https://docs.python.org/3/library/codecs.html#standard-encodings
#
for opt, arg in opts:
    if opt == '-h':
        print('Options: Extract.py --help --ipath <input directory> --itagfile <tag file> --otagfile <output file>')
    elif opt in ("--ipath"):
        inpath=arg
    elif opt in ("--itagfile"):
        itagfile=arg
    elif opt in ("--otagfile"):
        otagfile=arg
    elif opt in ("--encoding"):
        iencoding=arg
if inpath == '' :
    print('Input path  cannot be empty')
    sys.exit(2)

#Read tag file tagID,tagname1, tagname2. Sore a s a list [[tagID,[tagname1, tagname2...]] ...]
import csv
tags=[]
with open(itagfile,encoding=iencoding) as tagfile:
       row=csv.reader(tagfile,dialect='excel') 
       tags.append([row[0],row[1:]])

import minetext_utilities
from operator import itemgetter       
for f in os.listdir(inpath):
    with open(os.path.join(inpath,f),"r",encoding=iencoding) as infile, open(otagfile,"w",encoding=iencoding) as outfile:
        s=infile.read()
        result=[]
        for tag in tags:
            score=max([(t,minetext_utilities.tag_text(s,t)) for t in tag[1]],key=itemgetter(1))
            result.append([f, score[0], score[1]])
        out=csv.writer(outfile,dialect='excel')
        for r in result:
            out.writerow(result)
