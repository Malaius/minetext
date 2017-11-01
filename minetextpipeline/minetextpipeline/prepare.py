import sys, getopt
from sys import argv
import os

#Need to handle input parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"",["help","ipath=","oclean=","ochunks=","encoding="])
except getopt.GetoptError:
    print('Error specifying parameters. Try python Extract.py --help for help')
    sys.exit(2)

inpath=''
outpath=''
iencoding="utf-8"
#
#Refer to https://docs.python.org/3/library/codecs.html#standard-encodings
#
for opt, arg in opts:
    if opt == '--help':
        print('Options: Extract.py --help --ipath <input directory> --oclean <output directory')
    elif opt in ("--ipath"):
        inpath=arg
    elif opt in ("--oclean"):
        outpath=arg
    elif opt in ("--encoding"):
        iencoding=arg
if inpath == '' :
    print('Input path  cannot be empty')
    sys.exit(2)

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
#
# Remove stopwords, tokenize (punctuation symbols are different tokens), move to lowercase
def clean_doc(s):
    l=wordpunct_tokenize(s)
    stop_words = stopwords.words('english')
    return ' '.join([x.lower() for x in l if x.lower() not in stop_words])

#Run a cleanup: Remove anythig that is not an English" token. Typically, preparation step to create a dictionay
if outpath != '':
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    for f in os.listdir(inpath):
        with open(os.path.join(inpath,f),"r",encoding=iencoding) as infile, open(os.path.join(outpath,f+".clean"),"w",encoding=iencoding) as outfile:
            outfile.write(clean_doc(infile.read()))
