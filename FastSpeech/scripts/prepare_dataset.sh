#!/usr/bin/env bash

set -e

working_dir="$PWD"
# Download dataset
DATADIR="LJSpeech-1.1"
BZ2ARCHIVE="${DATADIR}.tar.bz2"
ENDPOINT="http://data.keithito.com/data/speech/$BZ2ARCHIVE"

cd ../..
if [ ! -d "$DATADIR" ]; then
  echo "dataset is missing, unpacking ..."
  if [ ! -f "$BZ2ARCHIVE" ]; then
    echo "dataset archive is missing, downloading ..."
    wget "$ENDPOINT"
  fi
  tar jxvf "$BZ2ARCHIVE"
fi

# Partition dataset
cd $working_dir
cp "../../${DATADIR}/metadata.csv" .
sed -n "1,13000p" "../../${DATADIR}/metadata.csv" > "metadata_train.csv"
sed -n "13001,13100p" "../../${DATADIR}/metadata.csv" > "metadata_test.csv"
