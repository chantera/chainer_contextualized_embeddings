#!/bin/bash

ROOT=$(cd `dirname $0`/../ && pwd)
DATA=$ROOT/data
INPUT=
OUTPUT=

usage() { echo "Usage: $0 --input file --output file"; }

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      if [[ $# -lt 2 ]]; then
        break
      fi
      INPUT=$2
      shift 2
      ;;
    -o|--output)
      if [[ $# -lt 2 ]]; then
        break
      fi
      OUTPUT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit
      ;;
    *)
      break
      ;;
  esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  usage
  exit 1
fi

python3 $ROOT/encode.py \
  --input $INPUT \
  --output $OUTPUT \
  --vocab $DATA/vocab-2016-09-10.txt \
  --model $DATA/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
  --config $DATA/elmo_2x4096_512_2048cnn_2xhighway_options.json
