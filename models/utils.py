from pathlib import Path

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