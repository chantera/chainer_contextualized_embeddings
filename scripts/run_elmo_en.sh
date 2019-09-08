#!/bin/bash

source `dirname $0`/common_run_args.sh

python3 $ROOT/encode.py \
  --input $INPUT \
  --output $OUTPUT \
  --vocab $DATA/vocab-2016-09-10.txt \
  --model $DATA/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
  --config $DATA/elmo_2x4096_512_2048cnn_2xhighway_options.json \
  --encoding elmo \
  --gpu $GPU
