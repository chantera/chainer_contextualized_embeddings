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
  --vocab $DATA/uncased_L-12_H-768_A-12/vocab.txt \
  --model $DATA/uncased_L-12_H-768_A-12/bert_model.ckpt.npz \
  --config $DATA/uncased_L-12_H-768_A-12/bert_config.json \
  --encoding bert_uncased
