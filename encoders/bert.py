import importlib
import re

import chainer
import numpy as np

from .encoder import Encoder

modeling = importlib.import_module('chainer-models.bert.modeling')
tokenization = importlib.import_module('chainer-models.bert.tokenization')


class BertEncoder(Encoder):
    # See: chainer-models/bert/extract_features.py

    def __init__(self, vocab_file, config_file, checkpoint_file,
                 do_lower_case=True, gpu=-1):
        config = modeling.BertConfig.from_json_file(config_file)
        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        bert = modeling.BertModel(config)
        self._model = modeling.BertExtracter(bert)
        ignore_names = ['output/W', 'output/b']
        chainer.serializers.load_npz(
            checkpoint_file, self._model, ignore_names=ignore_names)
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self._model.to_gpu()
        self._gpu = gpu

    def _preprocess(self, sentences):
        examples = read_examples(sentences)
        features = convert_examples_to_features(examples, self._tokenizer)
        self._features = features
        batch = make_batch(features, self._gpu)
        return batch

    def encode(self, sentences):
        inputs = self._preprocess(sentences)
        intermediates = self._model.get_all_encoder_layers(
            input_ids=inputs['input_ids'],
            input_mask=inputs['input_mask'],
            token_type_ids=inputs['input_type_ids'])
        outputs = self._postprosess(intermediates)
        return outputs

    def _postprosess(self, model_outputs):
        embeddings = self._extract_embeddings(model_outputs)
        outputs = []
        for i, feature in enumerate(self._features):
            vectors = []
            offset = 0
            for j, token in enumerate(feature['tokens']):
                if j > 0 and not token.startswith('##'):
                    if j - offset > 1:
                        v = self._pool_wordpieces(embeddings[i, offset:j])
                    else:
                        v = embeddings[i, offset]
                    vectors.append(v)
                    offset = j
            assert offset == j
            vectors.append(embeddings[i, offset])
            output = np.vstack(vectors)
            outputs.append(output)
        return outputs

    def _extract_embeddings(self, layer_outputs):
        # See: http://jalammar.github.io/illustrated-bert/
        return layer_outputs[-2].array

    def _pool_wordpieces(self, values):
        return values.mean(axis=0)


def read_examples(iterable):
    examples = []
    for line in iterable:
        line = tokenization.convert_to_unicode(line).strip()
        if not line:
            continue
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append((text_a, text_b))
    return examples


def convert_examples_to_features(examples, tokenizer):
    features = []
    for (ex_index, (text_a, text_b)) in enumerate(examples):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b) if text_b else None

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        features.append({'tokens': tokens,
                         'input_ids': np.array(input_ids, 'i'),
                         'input_mask': np.array(input_mask, 'i'),
                         'input_type_ids': np.array(input_type_ids, 'i')})
    return features


def make_batch(features, gpu):
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_input_ids.append(feature['input_ids'])
        all_input_mask.append(feature['input_mask'])
        all_input_type_ids.append(feature['input_type_ids'])

    def stack_and_to_gpu(data_list):
        sdata = chainer.functions.pad_sequence(
            data_list, length=None, padding=0).array
        return chainer.dataset.to_device(gpu, sdata)

    batch_input_ids = stack_and_to_gpu(all_input_ids).astype('i')
    batch_input_mask = stack_and_to_gpu(all_input_mask).astype('f')
    batch_input_type_ids = stack_and_to_gpu(all_input_type_ids).astype('i')
    return {'input_ids': batch_input_ids,
            'input_mask': batch_input_mask,
            'input_type_ids': batch_input_type_ids}
