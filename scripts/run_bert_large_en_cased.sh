#!/bin/bash

source `dirname $0`/common_run_args.sh

python3 $ROOT/encode.py \
  --input $INPUT \
  --output $OUTPUT \
  --vocab $DATA/cased_L-24_H-1024_A-16/vocab.txt \
  --model $DATA/cased_L-24_H-1024_A-16/bert_model.ckpt.npz \
  --config $DATA/cased_L-24_H-1024_A-16/bert_config.json \
  --encoding bert_cased \
  --gpu $GPU
