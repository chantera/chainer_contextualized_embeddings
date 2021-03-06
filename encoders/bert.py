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
                 do_lower_case=True, layers=(-1, -2, -3, -4), merge=False,
                 gpu=-1):
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
        self._do_lower_case = do_lower_case
        self._layers = layers
        self._merge = merge
        self._gpu = gpu
        self.context = None

    def _preprocess(self, sentences):
        examples = read_examples(sentences)
        features = convert_examples_to_features(examples, self._tokenizer)
        batch = make_batch(features, self._gpu)
        self.context = {
            'sentences': sentences,
            'features': features,
            'tokens': None,
            'layers': self._layers,
        }
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
        if self._merge == 'wordpieces':
            outputs, tokens = self._merge_wordpieces(embeddings)
        elif self._merge == 'as_inputs':
            outputs, tokens = self._merge_as_input_tokens(embeddings)
        elif self._merge == 'raw' or self._merge is False:
            tokens = [f['tokens'] for f in self.context['features']]
            outputs = [embeddings[i, :, :len(s)] for i, s in enumerate(tokens)]
        else:
            raise ValueError(
                'merging `{}` is not supported'.format(self._merge))
        self.context['tokens'] = tokens
        return outputs  # [(n_layers, n_tokens, dim)]

    def _extract_embeddings(self, layer_outputs):
        embeddings = [layer_outputs[index].array for index in self._layers]
        embeddings = self._model.xp.stack(embeddings, axis=1)
        return chainer.cuda.to_cpu(embeddings)

    def _merge_wordpieces(self, embeddings):
        outputs = []
        tokens = []
        for i, feature in enumerate(self.context['features']):
            vectors = []
            seq = []
            offset = 0
            buffer = ''
            for j, token in enumerate(feature['tokens']):
                if token.startswith('##'):
                    token = token[2:]
                elif j > 0:
                    if j - offset > 1:
                        v = self._pool_wordpieces(embeddings[i, :, offset:j])
                    else:
                        v = embeddings[i, :, offset]
                    vectors.append(v)
                    seq.append(buffer)
                    buffer = ''
                    offset = j
                buffer += token
            assert offset == j
            vectors.append(embeddings[i, :, offset])
            seq.append(buffer)
            output = np.stack(vectors, axis=1)
            outputs.append(output)
            tokens.append(seq)
        return outputs, tokens

    def _merge_as_input_tokens(self, embeddings):
        outputs = []
        tokens = []
        for i, (feature, sentence) in enumerate(
                zip(self.context['features'], self.context['sentences'])):
            if self._do_lower_case:
                sentence = sentence.lower()
            sentence = sentence.split()
            tid = 0
            vectors = []
            seq = []
            offset = 1
            buffer = ''
            iterator = enumerate(feature['tokens'])
            next(iterator)  # skip [CLS]
            for j, token in iterator:
                if buffer == sentence[tid]:
                    if j - offset > 1:
                        v = self._pool_wordpieces(embeddings[i, :, offset:j])
                    else:
                        v = embeddings[i, :, offset]
                    vectors.append(v)
                    seq.append(buffer)
                    buffer = ''
                    offset = j
                    tid += 1
                if token == "[SEP]":
                    token = "|||"
                elif token == "[UNK]":
                    token = sentence[tid]
                elif token.startswith('##'):
                    token = token[2:]
                buffer += token
            assert offset == j and token == "|||"
            assert len(seq) == len(sentence)
            output = np.stack(vectors, axis=1)
            outputs.append(output)
            tokens.append(seq)
        return outputs, tokens

    def _pool_wordpieces(self, values):
        return values.mean(axis=1)


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
