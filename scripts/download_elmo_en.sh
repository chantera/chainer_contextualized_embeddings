#!/bin/bash

ROOT=$(cd `dirname $0`/../ && pwd)
DATA=$ROOT/data
mkdir -p $DATA

wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json -P $DATA
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 -P $DATA
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt -P $DATA
