import json
import importlib

from chainer import cuda

from .encoder import Encoder

elmo_libs = importlib.import_module('chainer-models.elmo-chainer.bilm')


class ElmoEncoder(Encoder):
    # See: chainer-models/elmo-chainer/bilm/elmo.py

    def __init__(self, vocab_file, options_file, weight_file, gpu=-1):
        with open(options_file, 'r') as f:
            options = json.load(f)
        max_word_length = options['char_cnn']['max_characters_per_token']
        self._batcher = elmo_libs.data.Batcher(vocab_file, max_word_length)
        self._model = elmo_libs.elmo.Elmo(
            options_file, weight_file, num_output_representations=1,
            requires_grad=False, do_layer_norm=False, dropout=0.)
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self._model.to_gpu()

    def _preprocess(self, sentences):
        sentences = [s.split() for s in sentences]
        char_ids = self._batcher.batch_sentences(
            sentences, add_bos_eos=False)
        return self._model.xp.asarray(char_ids)

    def encode(self, sentences):
        inputs = self._preprocess(sentences)
        intermediates = self._model.forward(inputs)
        outputs = self._postprosess(intermediates)
        return outputs

    def _postprosess(self, model_outputs):
        mb_embedding_layers = model_outputs['elmo_layers']
        mb_mask = model_outputs['mask']
        mb_concat_embedding_layers = cuda.to_cpu(self._model.xp.stack(
            [mb_emb.array for mb_emb in mb_embedding_layers], axis=1))
        outputs = []
        for mask, concat_embedding_layers \
                in zip(mb_mask, mb_concat_embedding_layers):
            # remove pads
            length = int(mask.sum())
            concat_embedding_layers = concat_embedding_layers[:, :length]
            outputs.append(concat_embedding_layers)
        return outputs
