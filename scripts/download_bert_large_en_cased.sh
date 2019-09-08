#!/bin/bash

ROOT=$(cd `dirname $0`/../ && pwd)
DATA=$ROOT/data
mkdir -p $DATA

wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip -P $DATA
unzip $DATA/cased_L-24_H-1024_A-16.zip -d $DATA
rm $DATA/cased_L-24_H-1024_A-16.zip

python3 $ROOT/chainer-models/bert/convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $DATA/cased_L-24_H-1024_A-16/bert_model.ckpt \
  --npz_dump_path $DATA/cased_L-24_H-1024_A-16/bert_model.ckpt.npz
