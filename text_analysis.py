from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
print(" training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0], len(train_data[1]))
