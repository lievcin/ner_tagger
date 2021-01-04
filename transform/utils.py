import os

def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_sentences(df, group_col, word_col, tag_col):
	df["sentences"] = df.groupby(group_col)[word_col].transform(lambda x : ' '.join(x))
	df["tags"] = df.groupby(group_col)[tag_col].transform(lambda x : ' '.join(x))
	df = df[["sentences", "tags"]].drop_duplicates(inplace=False)
	return df["sentences"], df["tags"]