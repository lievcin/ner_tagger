import os
import csv
from utils import check_dir, make_sentences
import numpy as np
import pandas as pd

def transform(source_path):
    rows = []
    sentence_count = 1
    new_sentence=True

    for root, __subFolders, files in os.walk(source_path):
        for file in files:
            if file.endswith('.tags'):
                for line in open(os.path.join(root, file), encoding='utf-8'):
                    line = line.split()
                    if len(line) >= 5 and new_sentence==True:
                        row = [sentence_count, line[0], line[1], line[4]]
                        new_sentence=False
                        rows.append(row)
                    elif len(line) >= 5:
                        row = [sentence_count, line[0], line[1], line[4]]
                        rows.append(row)
                    else:
                        new_sentence = True
                        sentence_count += 1
    return rows, sentence_count

def main():

    source_path = "./data/gmb-1.0.0"
    columns = ["sentence_idx", "Word", "POS", "Tag"]

    rows, sentence_count  = transform(source_path)
    sentence_idx = np.array(range(sentence_count))

    # split into train and test files. this will help with keeping the generators simple,
    # plus this should really be done at the ETL stage of the pipeline anyway!
    test_idx = np.random.choice(np.array(range(sentence_count)), size=int(sentence_count*0.2), replace=False)
    train_idx = np.setdiff1d(sentence_idx,test_idx)

    # check that the directory to store the data exists, if not create it.
    check_dir("./data/processed_data/gmb/")
    df_train = pd.DataFrame(data=[s for s in rows if s[0] in train_idx], columns=columns)
    train_sentences, train_labels = make_sentences(df_train, group_col="sentence_idx", word_col="Word", tag_col="Tag")
    train_sentences.to_csv("./data/processed_data/gmb/train.sentences.csv", index=False, header=False)
    train_labels.to_csv("./data/processed_data/gmb/train.labels.csv", index=False, header=False)

    vocab = df_train["Word"].unique() # TODO change this to be a full list and add a frequency filter.
    tags = sorted(df_train["Tag"].unique(), reverse=True)

    with open("./data/processed_data/gmb/vocabulary.txt", "w", newline="") as f:
        f.write("\n".join(vocab))

    with open("./data/processed_data/gmb/tags.txt", "w", newline="") as f:
        f.write("\n".join(tags))

    del (df_train, train_sentences, train_labels, vocab, tags)

    check_dir("./data/processed_data/gmb/")
    df_test = pd.DataFrame(data=[s for s in rows if s[0] in test_idx], columns=columns)
    test_sentences, test_labels = make_sentences(df_test, group_col="sentence_idx", word_col="Word", tag_col="Tag")
    test_sentences.to_csv("./data/processed_data/gmb/test.sentences.csv", index=False, header=False)
    test_labels.to_csv("./data/processed_data/gmb/test.labels.csv", index=False, header=False)
    del (df_test, test_sentences, test_labels)


if __name__ == "__main__":
    main()