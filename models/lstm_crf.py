import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional

# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from livelossplot.inputs.tf_keras import PlotLossesCallback

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
            epochs=1,
            # callbacks=callbacks,
            verbose=1
        )           

    def save(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")        
            

if __name__ == "__main__":
    dataset = Dataset(path="./data/processed_data/gmb-1.0.0.csv", maxlen=50)
    model = LSTMCRF(dataset.vocabulary_size, dataset.labels_size)
    model.fit(dataset)
    # model.test(dataset)
    model.save()