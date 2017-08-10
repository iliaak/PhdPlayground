#!/bin/bash

SRC="$1"
TRG="$2"
TOKENIZER="/home/peter/moses/mosesdecoder/scripts/tokenizer/tokenizer.perl" #configure this!
MGIZABIN="/home/peter/mgiza/mgizapp/bin" #configure this!

if [ $# -eq 0 ]
  then
      echo "Please supply source and targer language as args."
      exit
fi

if [ ! -f raw_corp.$SRC ]; then
    echo "Input file "+raw_corp.$SRC+" not found. Please make sure input file is called raw_corp.SRC and is in this folder."
    exit
fi
if [ ! -f raw_corp.$TRG ]; then
    echo "Input file "+raw_corp.$TRG+" not found. Please make sure input file is called raw_corp.TRG and is in this folder."
    exit
fi
if [ ! -f configfile ]; then
    echo "No configfile found. Please create one first and edit variables therein."
    exit
fi

echo "Did you modify the configfile properly (source and target language and number of available cpus?"
read -n 1 -p "Enter 'y' to proceed:" userinput
echo ""
if [ "$userinput" = "y" ]; then
    echo "Proceeding with word alignment."
else
    echo "Could not parse input '$userinput'. Please try again."
    exit
fi



echo "Starting tokenization..."
$TOKENIZER -l $SRC < raw_corp.$SRC > corp.tok.$SRC
$TOKENIZER -l $TRG < raw_corp.$TRG > corp.tok.$TRG
echo "Done."

echo "Lowercasing corpus..."
tr '[:upper:]' '[:lower:]' < corp.tok.$SRC > corp.tok.low.$SRC
tr '[:upper:]' '[:lower:]' < corp.tok.$TRG > corp.tok.low.$TRG
echo "Done."

echo "Creating HMM classes..."
"$MGIZABIN/mkcls" -n10 -pcorp.tok.low.$SRC -Vcorp.tok.low.$SRC.vcb.classes
"$MGIZABIN/mkcls" -n10 -pcorp.tok.low.$TRG -Vcorp.tok.low.$TRG.vcb.classes
echo "Done."

echo "Translating corpus into GIZA format..."
"$MGIZABIN/plain2snt" corp.tok.low.$SRC corp.tok.low.$TRG
echo "Done."

echo "Creating cooccurrence..."
"$MGIZABIN/snt2cooc" corp.tok.low."$SRC"_corp.tok.low.$TRG.cooc corp.tok.low.$SRC.vcb corp.tok.low.$TRG.vcb corp.tok.low."$SRC"_corp.tok.low.$TRG.snt
echo "Done."



echo "Aligning..."
"$MGIZABIN/mgiza" configfile
echo "Done. Finished."
