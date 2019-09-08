#!/bin/bash

ROOT=$(cd `dirname $0`/../ && pwd)
DATA=$ROOT/data
INPUT=
OUTPUT=
GPU="-1"

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
    -g|--gpu)
      if [[ $# -lt 2 ]]; then
        break
      fi
      GPU=$2
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
