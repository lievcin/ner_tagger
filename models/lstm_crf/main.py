import os
from pathlib import Path
path = Path(__file__).parent

import sys
sys.path.append(".")
from src.utils import get_project_root
ROOT_DIR = get_project_root()

from optparse import OptionParser

import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional

# Logging
import logging
Path("{}/results".format(path)).mkdir(exist_ok=True)
# tf.compat.v1.logging.set_verbosity(logging.DEBUG)
# handlers = [
#     logging.FileHandler('results/main.log'),
#     logging.StreamHandler(sys.stdout)
# ]
# logging.getLogger('tensorflow').handlers = handlers

sys.path.append(str(Path('.').absolute().parent))
from models.utils import tags_dictionaries, words_dictionaries

def fwords(name):
    return str(Path(DATADIR, "{}.sentences.csv".format(name)))

def ftags(name):
    return str(Path(DATADIR, "{}.labels.csv".format(name)))

def parse_fn(line_words, line_tags):
    words = np.array([word2idx.get(w, 0) for w in line_words.strip().split()])
    tags = np.array([tag2idx[t] for t in line_tags.strip().split()])
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return words, tags

def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)

def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    output_signature = (
        tf.TensorSpec(shape=([None]), dtype=tf.int32),
        tf.TensorSpec(shape=([None]), dtype=tf.int32))

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_signature=output_signature
    )

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    shapes = (tf.TensorShape([None]),tf.TensorShape([None]))
    dataset = (dataset
               .padded_batch(batch_size=32,
                             padded_shapes=([params["max_len"]], [params["max_len"]]),
                             padding_values=(params['vocab_size']-1,params['pad_index'])
                            )
               .prefetch(1))
    return dataset

def model_fn(params):
    input_word = Input(shape=(params["max_len"],))
    model = Embedding(input_dim=params['vocab_size'], output_dim=params["embeddings_dim"], input_length=params["max_len"])(input_word)
    model = SpatialDropout1D(params["dropout"])(model)
    model = Bidirectional(LSTM(units=params["lstm_size"], return_sequences=True, recurrent_dropout=params["dropout"]))(model)
    out = TimeDistributed(Dense(params["labels_size"], activation="softmax"))(model)
    model = Model(input_word, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def model_save(model):
    model_json = model.to_json()
    with open("{}/model.json".format(path), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("{}/model.h5".format(path))
    logging.info("Saved model to disk!")

if __name__ == '__main__':
    usage = """
    Usage: can do one of three things: train, test or interactive. It uses the GMB by default,
    but using the -c option will tell it to use the Conll dataset.( not implemented yet)
    Usage scenarios:
    1) train [-g] [-c]: This command trains our bidirectional lsmt model against the Conll (or GMB if -g option is used)
    """

    parser = OptionParser(usage)
    parser.add_option("-c", "--conll", default=False,
        action = "store_true", dest = "use_conll", help = 'Use conll dataset. Default is false')
    parser.add_option("-g", "--gmb", default=True,
        action = "store_true", dest = "use_gmb", help = 'Use GMB dataset. Default is true')

    options, args = parser.parse_args()

    if len(args)==0 or args[0].lower().strip() not in ['train']:
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

    if cmd=="train":
        dataset = functools.partial(input_fn, fwords('train'), ftags('train'), params, shuffle_and_repeat=True)()
        model = model_fn(params)
        model.fit(dataset, epochs=1)
        model_save(model)