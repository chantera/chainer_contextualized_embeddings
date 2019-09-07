#!/usr/bin/env python

import argparse

import chainer
import h5py
import tqdm

import encoders


def _iter_line(iterable, skip_empty=True):
    for line in iterable:
        line = line.strip()
        if skip_empty and not line:
            continue
        yield line


def _iter_batch(samples, batch_size):
    batch = []
    iterator = iter(samples)
    while True:
        x = next(iterator, None)
        if x is None:
            break
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch.clear()
    if batch:
        yield batch


def run(encoder, input_file, output_file, batch_size=32):
    with chainer.using_config('train', False), \
            chainer.no_backprop_mode():
        sentence_id = 0
        n_lines = sum(1 for _ in _iter_line(open(input_file, 'r')))
        with open(input_file, 'r') as f_in, \
                h5py.File(output_file, 'w') as f_out:
            for batch in _iter_batch(
                    tqdm.tqdm(_iter_line(f_in), total=n_lines), batch_size):
                outputs = encoder.encode(batch)
                for output in outputs:
                    f_out.create_dataset(
                        '{}'.format(sentence_id),
                        output.shape, dtype='float32', data=output)
                    sentence_id += 1


def encode(input_file, output_file, vocab_file, model_file, config_file,
           encoding_type, batch_size=32):
    if encoding_type == 'elmo':
        encoder = encoders.ElmoEncoder(vocab_file, config_file, model_file)
    else:
        raise ValueError
    run(encoder, input_file, output_file, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    args = parser.parse_args()
    encode(args.input, args.output, args.vocab, args.model, args.config,
           'elmo', args.batchsize)
