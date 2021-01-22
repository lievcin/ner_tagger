import os
from pathlib import Path
path = Path(__file__).parent

import sys
sys.path.append(".")
from src.utils import get_project_root
ROOT_DIR = get_project_root()

import json
from optparse import OptionParser

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, Lambda
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras_crf import CRF

import csv
from sklearn.metrics import classification_report

# Logging
import logging
Path("{}/results".format(path)).mkdir(exist_ok=True)


sys.path.append(str(Path('.').absolute().parent))
from models.utils import tags_dictionaries, words_dictionaries, fwords, ftags, parse_fn, generator_fn, save_model


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    output_shapes = ([None], [None])
    output_types = (tf.int32, tf.int32)
    output_signature = (
        tf.TensorSpec(shape=([None]), dtype=tf.int32),
        tf.TensorSpec(shape=([None]), dtype=tf.int32))

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags, word2idx, tag2idx),
        # output_signature=output_signature # this is awaiting for tf2.4.1 release for CRF compatibility.
        output_shapes=output_shapes, output_types=output_types
    )

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    shapes = (tf.TensorShape([None]),tf.TensorShape([None]))
    dataset = (dataset
               .padded_batch(batch_size=params["batch_size"],
                             padded_shapes=([params["max_len"]], [params["max_len"]]),
                             padding_values=(params['vocab_size']-1,params['pad_index'])
                            )
               .prefetch(1))
    return dataset

def model_fn(params):
    input_word = Input(shape=(params["max_len"],))
    input_mask = Lambda(lambda x: tf.greater(x, 0))(input_word)
    model = Embedding(input_dim=params['vocab_size'], output_dim=params["embeddings_dim"], input_length=params["max_len"])(input_word)
    model = SpatialDropout1D(params["dropout"])(model)
    model = Bidirectional(LSTM(units=params["lstm_size"], return_sequences=True, recurrent_dropout=params["dropout"]))(model)
    logits = Dense(params["labels_size"], activation=None)(model)
    crf = CRF(params["labels_size"])
    outputs = crf(logits, mask=input_mask)
    model = tf.keras.Model(inputs=input_word, outputs=outputs)
    model.compile(loss=crf.neg_log_likelihood, metrics=[crf.accuracy], optimizer='adam')

    return model

def write_predictions(name, idx2tag, datadir, writedir):
    y_true, y_pred = [],[]
    target_names = list(idx2tag.values())

    with open('{}/results/{}.preds.csv'.format(writedir, name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        golds_gen = generator_fn(fwords(datadir, name), ftags(datadir, name), word2idx, tag2idx)
        dataset = functools.partial(input_fn, fwords(datadir, name), ftags(datadir, name), params, shuffle_and_repeat=False)()
        preds_gen = model.predict(dataset)
        for golds, preds in zip(golds_gen, preds_gen):

            (words, tags) = golds
            preds = np.argmax(preds, axis=-1)[:len(words)]
            for word, tag, tag_pred in zip(words, tags, preds):
                y_true.append(idx2tag[tag])
                y_pred.append(idx2tag[tag_pred])
                writer.writerow([idx2word[word], idx2tag[tag], idx2tag[tag_pred]])
    return y_true, y_pred

def write_classification_report(datadir, y_true, y_pred, name):
    with open("{}/{}.metric-report.txt".format(datadir, name), "w") as f:
        report = classification_report(y_true, y_pred)
        f.write(report)


if __name__ == '__main__':
    usage = """
    Usage: can do one of three things: train, test or interactive. It uses the GMB by default,
    but using the -c option will tell it to use the Conll dataset.( not implemented yet)
    Usage scenarios:
    1) train [-g] [-c]: This command trains our bidirectional lsmt model against the Conll (or GMB if -g option is used)
    1) test [-g] [-c]: This command loads a trained model and tests it against the the Conll (or GMB if -g option is used)
    """

    parser = OptionParser(usage)
    parser.add_option("-c", "--conll", default=False,
        action = "store_true", dest = "use_conll", help = 'Use conll dataset. Default is false')
    parser.add_option("-g", "--gmb", default=True,
        action = "store_true", dest = "use_gmb", help = 'Use GMB dataset. Default is true')

    options, args = parser.parse_args()

    if len(args)==0 or args[0].lower().strip() not in ["train", "test"]:
        print('Command not recognized.')
        print(usage)
        sys.exit(-1)
    else:
        cmd = args[0].lower().strip()

    if options.use_conll:
        DATADIR = "{}/data/processed_data/conll".format(ROOT_DIR)
    else:
        DATADIR = "{}/data/processed_data/gmb".format(ROOT_DIR)

    params = {
        "dim": 100,
        "dropout": 0.2,
        "max_len": 60,
        "epochs": 3,
        "batch_size": 32,
        "buffer": 15000,
        "lstm_size": 100,
        "words": str(Path(DATADIR, "vocabulary.txt")),
        "tags": str(Path(DATADIR, "tags.txt")),
        "embeddings_dim": 50
    }

    tag2idx, idx2tag , tags_len= tags_dictionaries(params)
    word2idx, idx2word, vocab_size = words_dictionaries(params)
    params['vocab_size']=vocab_size
    params['pad_index']=tag2idx['O']
    params["labels_size"]=tags_len

    with Path("{}/results/params.json".format(path)).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    if cmd=="train":
        dataset = functools.partial(input_fn, fwords(DATADIR, 'train'), ftags(DATADIR, 'train'), params, shuffle_and_repeat=True)()
        valid_dataset = functools.partial(input_fn, fwords(DATADIR, 'test'), ftags(DATADIR, 'test'), params, shuffle_and_repeat=True)()
        model = model_fn(params)
        model.fit(dataset, validation_data=valid_dataset, epochs=params["epochs"])
        save_model(model, path)
        predictions = model.predict(valid_dataset)
        y_true, y_pred = write_predictions('train', idx2tag, DATADIR, path)
        _ = write_classification_report(path, y_true, y_pred, 'train')
    elif cmd=="test":
        model=tf.keras.models.load_model("{}/results/model".format(path), compile=False)
        model.compile()
        valid_dataset = functools.partial(input_fn, fwords(DATADIR, 'test'), ftags(DATADIR, 'test'), params, shuffle_and_repeat=False)()
        predictions = model.predict(valid_dataset)
        y_true, y_pred = write_predictions('test', idx2tag, DATADIR, path)
        _ = write_classification_report(path, y_true, y_pred, 'test')