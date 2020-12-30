import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
    
class Dataset(object):
    def __init__(self, path, maxlen):
        """
        Defines self.df as a df with the loaded dataset (for now) ... TODO complete docstring
        Args:
            path: path to the dataset file to load
        """         
        self.data = pd.read_csv(path)
        self.data = self.data.fillna(method="ffill")
        self.maxlen = maxlen
        
        self.vocabulary = list(set(self.data["Word"].values)) + ["ENDPAD"]
        self.vocabulary_size = len(self.vocabulary)
        self.labels = list(set(self.data["Tag"].values))
        self.labels_size = len(self.labels)

        self.word2idx = {w: i for i, w in enumerate(self.vocabulary)}
        self.idx2word = {i: w for i, w in enumerate(self.vocabulary)}
        self.tag2idx = {t: i for i, t in enumerate(self.labels)}
        self.idx2tag = {i: t for i, t in enumerate(self.labels)}        
        self.sentences = self.make_sentences()
        
    def make_sentences(self):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        grouped = self.data.groupby("Sentence #").apply(agg_func)
        sentences = [s for s in grouped]
        return sentences
    
    def make_sequences(self):
        X = [[self.word2idx[w[0]] for w in s] for s in self.sentences]
        y = [[self.tag2idx[w[2]] for w in s] for s in self.sentences]
        return X,y
        
    def pad_sequences(self, X, y):
        X = pad_sequences(maxlen=self.maxlen, sequences=X, padding="post", value=self.vocabulary_size-1)
        y = pad_sequences(maxlen=self.maxlen, sequences=y, padding="post", value=self.tag2idx["O"])
        return X,y

    @property
    def Xy(self, test_size=0.2, random_state=42):
        X, y = self.make_sequences()
        X, y = self.pad_sequences(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
