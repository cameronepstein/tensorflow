from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
imdb = keras.datasets.imdb

# Assign training and testing data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

# Get words and their corresponding number codes
word_index = imdb.get_word_index()

# Add values to all words in word_index.items
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 #unknown
word_index["<UNUSED>"] = 3

# Reverse all word index words and numeric values
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Return string value of numeric sequences
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))
