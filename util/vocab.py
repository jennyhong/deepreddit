import csv
import numpy as np
import random

def randvec(d, lower=-0.5, upper=0.5):
  return np.array([random.uniform(lower, upper) for i in range(d)])

def get_word_indices(vocab_dir, vocab_filename):
  ######## create word vectors from vocab_filename
  # If a word in vocab_filename is not in the GloVe dataset, then
  # it is initialized to a random vector using randvec.
  #
  # This allows us to use special tokens like <UNK>, which are not in GloVe.
  #
  # wv is a np.array of dimension n_words by d
  # word_to_num is a dict() { word : i } where i is its row id in wv
  # num_to_word is a dict() { i : word } where i is the index of a row in wv
  with open(vocab_dir + vocab_filename, 'r') as f:
    words = [line.strip() for line in f]
  word_to_num = dict()
  num_to_word = dict()
  for i, w in enumerate(words):
    word_to_num[w] = i
    num_to_word[i] = w
  return word_to_num, num_to_word

def load_wv(data_dir='vocab/', glove_filename='glove.6B.50d.txt',
  vocab_filename='redditVocabBase.txt', d=50):
  ######## create glove_dict from glove_filename
  # { word : numpy array }
  glove_dict = dict()
  with open(data_dir + glove_filename, 'r') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for line in reader:
      glove_dict[line[0]] = np.array(list(map(float, line[1: ])))

  word_to_num, num_to_word = get_word_indices(data_dir, vocab_filename)
  return wv, word_to_num, num_to_word