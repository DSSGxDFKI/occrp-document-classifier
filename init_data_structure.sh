#!/bin/bash

ROOT_PATH=$1

# create folder structure from zip
if [ ! -f data.zip ]
then
	echo "File data.zip not found"
    exit 1
fi

cp data.zip "$ROOT_PATH" || { echo "Could not find path $ROOT_PATH, exiting"; exit 1; }
cd "$ROOT_PATH" || { echo "Error with path $ROOT_PATH, exiting"; exit 1; }
unzip data.zip
rm data.zip

# rvl dataset download
# uncommment this only of you need to train feature extractors
# wget https://huggingface.co/datasets/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz
# tar -xf rvl-cdip.tar.gz -C input/rvl-cdip
# rm rvl-cdip.tar.gz