# chainer_contextualized_embeddings

This is a wrapper of [chainer/models](https://github.com/chainer/models) to get contextualized embeddings easily.

## Installation

```
git clone --recursive https://github.com/chantera/chainer_contextualized_embeddings
pip install -r requirements.txt
```

## Usage

### Obtain ELMo Embeddings

```
./scripts/download_elmo_en.sh
./scripts/run_elmo_en.sh -i input.txt -o output.hdf5 -g [GPU]
```

or

```
python encode.py \
  --input input.txt \
  --output output.hdf5 \
  --vocab data/vocab-2016-09-10.txt \
  --model data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
  --config data/elmo_2x4096_512_2048cnn_2xhighway_options.json \
  --encoding elmo \
  --gpu [GPU]
```

### Obtain BERT Embeddings

```
./scripts/download_bert_base_en_uncased.sh
./scripts/run_bert_base_en_uncased.sh -i input.txt -o output.hdf5 -g [GPU]
```

or

```
python encode.py \
  --input input.txt \
  --output output.hdf5 \
  --vocab data/uncased_L-12_H-768_A-12/vocab.txt \
  --model data/uncased_L-12_H-768_A-12/bert_model.ckpt.npz \
  --config data/uncased_L-12_H-768_A-12/bert_config.json \
  --encoding bert_uncased \
  --gpu [GPU]
```
