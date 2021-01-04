import os
import csv

def transform(gmb_dir):
    sentences = []
    sentence_count = 1
    new_sentence=True

    for root, __subFolders, files in os.walk(gmb_dir):
        for file in files:
            if file.endswith('.tags'):
                rows = []
                for line in open(os.path.join(root, file), encoding = 'utf-8'):
                    line = line.split()
                    if len(line) >= 5 and new_sentence==True:
                        row = ["Sentence: {}".format(sentence_count), line[0], line[1], line[4]]
                        new_sentence=False
                        sentences.append(row)
                    elif len(line) >= 5:
                        row = ["", line[0], line[1], line[4]]
                        sentences.append(row)
                    else:
                        new_sentence = True
                        sentence_count += 1
    return sentences

def main():

    gmb_dir = './data/gmb-1.0.0'
    sentences  = transform(gmb_dir)

    with open("./data/processed_data/gmb-1.0.0.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sentence #", "Word", "POS", "Tag"])
        writer.writerows(sentences)

if __name__ == "__main__":
    main()