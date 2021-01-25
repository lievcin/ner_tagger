import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf import CRF

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Lambda, Bidirectional, SpatialDropout1D, Dense
from tensorflow.keras.optimizers import Adam


def BiLSTMCRF(vocab_size, embedding_size, hidden_size=256, dropout=0.1, num_classes=3, lr=5e-5):

    sequence_input = Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size)(sequence_input)
    dropout = SpatialDropout1D(dropout)(embedding)
    bilstm = Bidirectional(
        LSTM(units=hidden_size // 2, return_sequences=True)
        )
    outputs = bilstm(dropout)
    logits = Dense(num_classes, activation=None)(outputs)
    crf = CRF(num_classes)
    outputs = crf(inputs=logits, mask=sequence_mask)
    model = Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[crf.accuracy],
        optimizer=Adam(lr))

    model.summary()
    return model