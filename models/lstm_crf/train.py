import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
from sklearn.metrics import classification_report

from pathlib import Path
path = Path(__file__).parent

import sys
sys.path.append(".")
from datasets.dataset import Dataset

class LSTMCRF:

    def __init__(self, vocabulary_size, labels_size, embeddings_size=50, lstm_units=100, dropout=0.2):
        input_word = Input(shape=(embeddings_size,))
        model = Embedding(input_dim=vocabulary_size, output_dim=embeddings_size)(input_word)
        model = SpatialDropout1D(dropout)(model)
        model = Bidirectional(LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=dropout))(model)
        out = TimeDistributed(Dense(labels_size, activation="softmax"))(model)
        self.model = Model(input_word, out)
        self.checkpoint = ModelCheckpoint("{}/model.h5".format(path), monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')

    def fit(self, data):
        self.model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        X_train, X_test, y_train, y_test = data.Xy

        self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test,y_test),
            batch_size=32, 
            epochs=3,
            callbacks=[self.checkpoint, EarlyStopping()],
            verbose=1
        )           

    def test(self, data):
        _, X, _, y = data.Xy
        self.model.evaluate(X, y)
        n_sequences, sequence_len = X.shape
        predictions = self.model.predict(X)
        flat_predictions = []
        flat_gold = []
        with open("{}/predictions.txt".format(path), "w") as f:
            f.write(f"WORD\tPREDICTION\tGOLD STANDARD\n")
            for sequence_i in range(n_sequences):
                f.write("\n")
                for word_i in range(sequence_len):
                    word = data.idx2word[X[sequence_i, word_i]]
                    prediction_i = np.argmax(predictions[sequence_i, word_i, :])
                    prediction = data.idx2tag[prediction_i]
                    flat_predictions.append(prediction)
                    truth = data.idx2tag[y[sequence_i, word_i]]
                    flat_gold.append(truth)
                    f.write(f"{word}\t{prediction}\t{truth}\n")
        with open("{}/metric-report.txt".format(path), "w") as f:
            report = classification_report(flat_gold, flat_predictions)
            f.write(report)

    def save(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("{}/model.json".format(path), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("{}/model.h5".format(path))
        print("Saved model to disk")        
            

if __name__ == "__main__":
    dataset = Dataset(path="./data/processed_data/gmb-1.0.0.csv", maxlen=50)
    model = LSTMCRF(dataset.vocabulary_size, dataset.labels_size)
    model.fit(dataset)
    model.test(dataset)
    model.save()
    # print(path)