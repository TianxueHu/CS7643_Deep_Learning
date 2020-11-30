# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import csv
import string
from collections import Counter, defaultdict
import random
import numpy as np

random.seed(1)


def main():
    
    train_data = []
    train_words = []
    val_data = []
    train_labels = []
    val_labels = []

    with open('../datasets/CoLA/train.tsv', 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for i, line in enumerate(reader):
            label = line[1]
            text = line[3].translate(str.maketrans('', '', string.punctuation)).lower().split()
            if i < 7000:
                train_data.append(text)
                train_labels.append(int(label))
                train_words.extend(text)
            else:
                val_data.append(text)
                val_labels.append(int(label))

    word_occs = Counter(train_words)
    vocab = [k for k, v in dict(word_occs).items() if v >= 5]
    inxs = [i for i in range(1, len(vocab))]
    vocab_lookup = dict(zip(vocab, inxs))
    vocab_lookup['UNK'] = 0
    vocab_lookup['CLS'] = len(vocab_lookup)
    vocab_lookup['PAD'] = len(vocab_lookup)
    print(len(vocab_lookup)) #1542 
    # indices ranging from [0, 1541]

    # pad (note the proper means of doing this is to only pad to max length in batch, during training each time a batch is drawn. We take a shortcut here)
    lengths = [len(l) for l in train_data + val_data]
    max_length = max(lengths)

    # convert each sentence to its vocabulary lookup
    def construct_instance(instance):
        return [vocab_lookup[token] if token in vocab_lookup else vocab_lookup['UNK'] for token in instance] + [vocab_lookup['CLS']] + [vocab_lookup['PAD']]*(max_length-len(instance))
    train_inxs = np.array([construct_instance(i) for i in train_data])
    val_inxs = np.array([construct_instance(i) for i in val_data])
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # save the objects
    np.save('../datasets/train_inxs', train_inxs)
    np.save('../datasets/val_inxs', val_inxs)
    np.save('../datasets/train_labels', train_labels)
    np.save('../datasets/val_labels', val_labels)

    with open('../datasets/word_to_ix.csv', 'w') as f:
        writer = csv.writer(f)
        for k, v in vocab_lookup.items():
            writer.writerow([k,v])

if __name__ == '__main__':
    main()
