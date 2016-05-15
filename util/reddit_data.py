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
  min_length=10, full_length=10):
  """
  min_length: All comments shorter than this number of words
          are ignored entirely.

  We use npoints to mean the number of comments in filename
  longer than min_length. So we return

  X: np.array of size (npoints, full_length)
      Each sentence shorter than full_length is padded with the
      token '<FILLER>' until it is full_length, so that X can be
      a single matrix, rather than a list of length npoints.
      Each row of x stores the indices of the words in the sentence.
  y: np.array of size (npoints)
      The indices of the classes of each data point.
  lengths: np.array of size (npoints)
      The original length of each sentence.
  """
  X = list()
  y = list()
  lengths = list()
  with open(filename, 'r') as f:
    for line in f.readlines():
      items = line.strip().split('\t')
      # In Will's reddit dataset, all comments start with <EOS>
      # TODO: Make sure this generalizes if we need to use another dataset
      words = items[9].split()[1:]
      
      if len(words) < min_length:
        continue

      if len(words) < full_length:
        lengths.append(len(words))
      else:
        lengths.append(full_length)

      while len(words) < full_length:
        words.append('<FILLER>')
      
      x = list() # Add the data point to x
      for i in xrange(full_length):
        word = words[i]
        if word not in word_to_num:
          word = '<UNK>'
        x.append(word_to_num[word])
      X.append(np.array(x))

      # Add the data point to y
      class_name = items[0]
      y.append(class_to_num[class_name])
  return np.array(X), np.array(y), np.array(lengths)

def data_iterator(orig_X, orig_lengths, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_lengths = orig_lengths[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_lengths = orig_lengths
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    lengths = data_lengths[batch_start:batch_start + batch_size]
    y = data_y[batch_start:batch_start + batch_size]
    yield x, y, lengths
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
