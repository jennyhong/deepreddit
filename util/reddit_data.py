import csv
import numpy as np
import random

def randvec(d, lower=-0.5, upper=0.5):
  return np.array([random.uniform(lower, upper) for i in range(d)])

def load_wv(data_dir='vocab/', glove_filename='glove.6B.50d.txt',
  vocab_filename='redditVocabBase.txt', d=50):
  ######## create glove_dict from glove_filename
  # { word : numpy array }
  glove_dict = dict()
  with open(data_dir + glove_filename, 'r') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for line in reader:
      glove_dict[line[0]] = np.array(list(map(float, line[1: ])))

  ######## create word vectors from vocab_filename
  # If a word in vocab_filename is not in the GloVe dataset, then
  # it is initialized to a random vector using randvec.
  #
  # This allows us to use special tokens like <UNK>, which are not in GloVe.
  #
  # wv is a np.array of dimension n_words by d
  # word_to_num is a dict() { word : i } where i is its row id in wv
  # num_to_word is a dict() { i : word } where i is the index of a row in wv
  wvs = list()
  with open(data_dir + vocab_filename, 'r') as f:
    words = [line.strip() for line in f]
  word_to_num = dict()
  num_to_word = dict()
  for i, w in enumerate(words):
    word_to_num[w] = i
    num_to_word[i] = w
  for word in words:
    vector = glove_dict.get(word.strip(), randvec(d))
    wvs.append(vector)
  wv = np.array(wvs)

  return wv, word_to_num, num_to_word

def load_dataset(filename, word_to_num, class_to_num,
  min_sentence_length=10, full_sentence_length=10):
  X = list()
  y = list()
  with open(filename, 'r') as f:
    for line in f.readlines():
      items = line.strip().split('\t')
      
      # In Will's reddit dataset, all comments start with <EOS>
      # TODO: Make sure this generalizes if we need to use another dataset
      words = items[9].split()[1:]
      
      if len(words) < min_sentence_length:
        # TODO: Pad sentences longer than min_sentence_length to full_sentence_length
        # For now, we'll just skip anything shorter than full_sentence_length
        continue
      # Add the data point to x
      x = list()
      for i in xrange(full_sentence_length):
        word = words[i]
        if word not in word_to_num:
          word = '<UNK>'
        x.append(word_to_num[word])
      X.append(np.array(x))

      # Add the data point to y
      class_name = items[0]
      y.append(class_to_num[class_name])
  return np.array(X), np.array(y)
