import tensorflow as tf
# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import LSTM, Embedding, Dense
# from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# import numpy as np
# from sklearn.metrics import classification_report
import json
import functools


from pathlib import Path
Path('results').mkdir(exist_ok=True)



# path = Path(__file__).parent

# import sys
# sys.path.append(".")
# from datasets.dataset import Dataset

# class LSTMCRF:

#     def __init__(self, vocabulary_size, labels_size, embeddings_size=50, lstm_units=100, dropout=0.2):
#         input_word = Input(shape=(embeddings_size,))
#         model = Embedding(input_dim=vocabulary_size, output_dim=embeddings_size)(input_word)
#         model = SpatialDropout1D(dropout)(model)
#         model = Bidirectional(LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=dropout))(model)
#         out = TimeDistributed(Dense(labels_size, activation="softmax"))(model)
#         self.model = Model(input_word, out)
#         self.checkpoint = ModelCheckpoint("{}/model.h5".format(path), monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')

#     def fit(self, data):
#         self.model.compile(optimizer="adam",
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"])

#         X_train, X_test, y_train, y_test = data.Xy

#         self.model.fit(
#             x=X_train,
#             y=y_train,
#             validation_data=(X_test,y_test),
#             batch_size=32,
#             epochs=3,
#             callbacks=[self.checkpoint, EarlyStopping()],
#             verbose=1
#         )

#     def test(self, data):
#         _, X, _, y = data.Xy
#         self.model.evaluate(X, y)
#         n_sequences, sequence_len = X.shape
#         predictions = self.model.predict(X)
#         flat_predictions = []
#         flat_gold = []
#         with open("{}/predictions.txt".format(path), "w") as f:
#             f.write(f"WORD\tPREDICTION\tGOLD STANDARD\n")
#             for sequence_i in range(n_sequences):
#                 f.write("\n")
#                 for word_i in range(sequence_len):
#                     word = data.idx2word[X[sequence_i, word_i]]
#                     prediction_i = np.argmax(predictions[sequence_i, word_i, :])
#                     prediction = data.idx2tag[prediction_i]
#                     flat_predictions.append(prediction)
#                     truth = data.idx2tag[y[sequence_i, word_i]]
#                     flat_gold.append(truth)
#                     f.write(f"{word}\t{prediction}\t{truth}\n")
#         with open("{}/metric-report.txt".format(path), "w") as f:
#             report = classification_report(flat_gold, flat_predictions)
#             f.write(report)

#     def save(self):
#         # serialize model to JSON
#         model_json = self.model.to_json()
#         with open("{}/model.json".format(path), "w") as json_file:
#             json_file.write(model_json)
#         # serialize weights to HDF5
#         self.model.save_weights("{}/model.h5".format(path))
#         print("Saved model to disk")


# if __name__ == "__main__":
#     dataset = Dataset(path="./data/processed_data/gmb-1.0.0.csv", maxlen=50)
#     model = LSTMCRF(dataset.vocabulary_size, dataset.labels_size)
#     model.fit(dataset)
#     model.test(dataset)
#     model.save()
#     # print(path)


DATADIR = './data/processed_data/gmb'


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags

def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)

def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn():
    pass


if __name__ == '__main__':

    params = {
        "dim": 300,
        "dropout": 0.5,
        # "num_oov_buckets": 1,
        "epochs": 3,
        "batch_size": 32,
        "buffer": 15000,
        "lstm_size": 100,
        "words": str(Path(DATADIR, "vocabulary.txt")),
        "tags": str(Path(DATADIR, "tags.txt"))
    }

    with Path("results/params.json").open("w") as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, "{}.sentences.txt".format(name)))

    def ftags(name):
        return str(Path(DATADIR, "{}.labels.txt".format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('test'), ftags('test'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True) # TODO take this outside?
    # hook = tf.contrib.estimator.stop_if_no_increase_hook(
    #     estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf) #, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # # Write predictions to file
    # def write_predictions(name):
    #     Path('results/score').mkdir(parents=True, exist_ok=True)
    #     with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
    #         test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
    #         golds_gen = generator_fn(fwords(name), ftags(name))
    #         preds_gen = estimator.predict(test_inpf)
    #         for golds, preds in zip(golds_gen, preds_gen):
    #             ((words, _), tags) = golds
    #             for word, tag, tag_pred in zip(words, tags, preds['tags']):
    #                 f.write(b' '.join([word, tag, tag_pred]) + b'\n')
    #             f.write(b'\n')

    for name in ['train', 'test']:
        write_predictions(name)