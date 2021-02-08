import logging
import os
import re

from pathlib import Path
import tensorflow as tf

def read_vocab_file(vocab_file):
    with Path(vocab_file).open() as f:
        token2id = {t.strip(): i for i, t in enumerate(f,1)}
        token2id["UNK"]=0
        token2id["PAD"]=len(token2id)
    return token2id

def read_labels_file(labels_file):
    with Path(labels_file).open() as f:
        label2id = {t.strip(): i for i, t in enumerate(f)}
    return label2id


def read_files(input_files, callback=None):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist.', f)
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.rstrip('\n')
                if not line:
                    continue
                if callback:
                    callback(line)
        logging.info('Read file %s finished.', f)
    logging.info('Read all files finished.')


def read_labelled_files(sentences_file, labels_file, sep=' '):
    features, labels = [], []
    def collect_tokens_fn(line):
        tokens = re.split(sep, line)
        features.append(tokens)

    def collect_tags_fn(line):
        tags = re.split(sep, line)
        labels.append(tags)

    read_files(sentences_file, callback=collect_tokens_fn)
    read_files(labels_file, callback=collect_tags_fn)

    return features, labels


class TokenMapper:
    def __init__(self, vocab_file):
        self.token2id = read_vocab_file(vocab_file)
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.unk_token = 'UNK'
        self.unk_id = self.token2id[self.unk_token]
        self.pad_token = 'PAD'
        self.pad_id = self.token2id[self.pad_token]

    def encode(self, tokens):
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.id2token.get(_id, self.unk_token) for _id in ids]
        return tokens


class LabelMapper:
    def __init__(self, labels_file):
        self.label2id = read_labels_file(labels_file)
        self.id2label = {v: k for k, v in self.label2id.items()}

    def encode(self, labels):
        ids = [self.label2id.get(label, 0) for label in labels]
        return ids

    def decode(self, ids):
        labels = [self.id2label.get(_id, 'O') for _id in ids]
        return labels


class DatasetBuilder:

    def __init__(self, token_mapper, label_mapper, **kwargs):
        self.token_mapper = token_mapper
        self.label_mapper = label_mapper
        self.feature_pad_id = self.token_mapper.pad_id
        self.label_pad_id = self.label_mapper.label2id['O']

    def build_labelled_dataset(self, sentences_file, labels_file, batch_size):
        features, labels = read_labelled_files(sentences_file, labels_file)
        features = [self.token_mapper.encode(x) for x in features]
        labels = [self.label_mapper.encode(x) for x in labels]
        features = tf.ragged.constant(features, dtype=tf.int32)
        labels = tf.ragged.constant(labels, dtype=tf.int32)
        x_dataset = tf.data.Dataset.from_tensor_slices(features)
        # convert ragged tensor to tensor
        x_dataset = x_dataset.map(lambda x: x)
        y_dataset = tf.data.Dataset.from_tensor_slices(labels)
        y_dataset = y_dataset.map(lambda y: y)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.shuffle(buffer_size=10000000, reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]),
            padding_values=(self.token_mapper.pad_id, self.label_mapper.label2id['O'])
        )
        return dataset

    def build_valid_dataset(self, input_file, batch_size):
        return self.build_labelled_dataset(sentences_file, labels_file, batch_size, buffer_size, repeat=repeat)
