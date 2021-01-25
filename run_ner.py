import argparse
import logging
import os
import tensorflow as tf
from models.bilstm_crf.model import BiLSTMCRF
from datasets.dataset import TokenMapper, LabelMapper, DatasetBuilder

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bilstm-crf', choices=[
        'bilstm-crf'
    ])
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--label_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--train_input_sentences_file', type=str, default=None)
    parser.add_argument('--train_input_labels_file', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_input_sentences_file', type=str, default=None)
    parser.add_argument('--valid_input_labels_file', type=str, default=None)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--embedding_size', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    args, _ = parser.parse_known_args()
    return args

def choose_model(args):
    if args.model == 'bilstm-crf':
        assert args.vocab_size is not None, "vocab_size must be set when using bilstm-crf model!"
        assert args.embedding_size is not None, "embedding_size must be set when using bilstm-crf model!"
        model = BiLSTMCRF(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            num_classes=args.num_classes,
            lr=args.learning_rate)
        path = 'models/bilstm_crf/results'
        return model, path


def build_dataset(args):
    token_mapper = TokenMapper(vocab_file=args.vocab_file)
    label_mapper = LabelMapper(labels_file=args.label_file)
    num_classes = len(label_mapper.label2id)

    dataset = DatasetBuilder(token_mapper, label_mapper)
    train_dataset = dataset.build_labelled_dataset(
        sentences_file=args.train_input_sentences_file,
        labels_file=args.train_input_labels_file,
        batch_size=args.train_batch_size)
    if not args.valid_input_sentences_file:
        return train_dataset, None, num_classes
    valid_dataset = dataset.build_labelled_dataset(
        sentences_file=args.valid_input_sentences_file,
        labels_file=args.valid_input_labels_file,
        batch_size=args.valid_batch_size)
    return train_dataset, valid_dataset, num_classes


def train_model(model, train_dataset, valid_dataset, model_dir, args):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tensorboard_logdir = os.path.join(model_dir, 'logs')
    saved_model_dir = os.path.join(model_dir, 'export', '{epoch}')
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss' if valid_dataset is not None else 'loss'),
            tf.keras.callbacks.TensorBoard(tensorboard_logdir),
            tf.keras.callbacks.ModelCheckpoint(
                saved_model_dir,
                save_best_only=False,
                save_weights_only=False)
        ]
    )

if __name__ == "__main__":
    args = add_arguments()
    train_dataset, valid_dataset, num_classes = build_dataset(args)
    args.num_classes=num_classes
    model, model_dir = choose_model(args)
    train_model(model, train_dataset, valid_dataset, model_dir, args)