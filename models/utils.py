from pathlib import Path
import numpy as np

def tags_dictionaries(params):
    with Path(params['tags']).open() as f:
        tag2idx = {t.strip(): i for i, t in enumerate(f)}
        idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag, len(tag2idx)

def words_dictionaries(params):
    with Path(params['words']).open() as f:
        word2idx = {t.strip(): i for i, t in enumerate(f,1)}
        word2idx["UNK"]=0
        word2idx["ENDPAD"]=len(word2idx)
        idx2word = {i: t for t, i in word2idx.items()}
    return word2idx, idx2word, len(word2idx)

def fwords(datadir, name):
    return str(Path(datadir, "{}.sentences.csv".format(name)))

def ftags(datadir, name):
    return str(Path(datadir, "{}.labels.csv".format(name)))

def parse_fn(line_words, line_tags, word2idx, tag2idx):
    words = np.array([word2idx.get(w, 0) for w in line_words.strip().split()])
    tags = np.array([tag2idx[t] for t in line_tags.strip().split()])
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return words, tags

def generator_fn(words, tags, word2idx, tag2idx):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags, word2idx, tag2idx)

def save_model(model, path):
    model.save("{}/results/model".format(path))