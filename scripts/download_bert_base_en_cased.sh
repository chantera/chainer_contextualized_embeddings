#!/bin/bash

ROOT=$(cd `dirname $0`/../ && pwd)
DATA=$ROOT/data
mkdir -p $DATA

wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -P $DATA
unzip $DATA/cased_L-12_H-768_A-12.zip -d $DATA
rm $DATA/cased_L-12_H-768_A-12.zip

python3 $ROOT/chainer-models/bert/convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $DATA/cased_L-12_H-768_A-12/bert_model.ckpt \
  --npz_dump_path $DATA/cased_L-12_H-768_A-12/bert_model.ckpt.npz
