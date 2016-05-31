import collections
import numpy as np
import vocab

def generate_onehot(y, num_classes):
  y_t = np.zeros((len(y), num_classes))
  y_t[np.arange(len(y)), y] = 1
  return y_t

class DataLoader:

  def __init__(self, config):
    self.config = config
    self.load_data()

  def load_data(self):
    self.wv, word_to_num, num_to_word = vocab.load_wv(data_dir=self.config.vocab_dir)
    self.num_vocab, self.wv_dim = self.wv.shape
    self.class_names = self.config.class_names
    self.config.num_classes = len(self.class_names) #TODO: make modular lolololol
    num_to_class = dict(enumerate(self.class_names))
    class_to_num = {v:k for k,v in num_to_class.iteritems()}

    self.X_train, self.y_train, self.lengths_train = self.load_datafile(self.config.train_file,
      word_to_num, class_to_num, min_length=10, full_length=self.config.lstm_size)
    self.y_train = generate_onehot(self.y_train, self.config.num_classes)
    if self.config.debug:
      n_sample = len(self.X_train) / 1024
      if n_sample > 0:
        self.X_train = self.X_train[::n_sample]
        self.y_train = self.y_train[::n_sample]

    self.X_val, self.y_val, self.lengths_val = self.load_datafile(self.config.val_file,
      word_to_num, class_to_num, min_length=10, full_length=self.config.lstm_size)
    self.y_val = generate_onehot(self.y_val, self.config.num_classes)
    if self.config.debug:
      n_sample = len(self.X_val) / 1024
      if n_sample > 0:
        self.X_val = self.X_val[::n_sample]
        self.y_val = self.y_val[::n_sample]

    self.X_test, self.y_test, self.lengths_test = self.load_datafile(self.config.test_file,
      word_to_num, class_to_num, min_length=10, full_length=self.config.lstm_size)
    self.y_test = generate_onehot(self.y_test, self.config.num_classes)

  def load_datafile(self, filename, word_to_num, class_to_num, 
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

  def load_data_text(self, filename):
    X = []
    with open(filename, 'r') as f:
      items = line.strip().split('\t')
      words = items[9].split()[1:]
      X.append(collections.Counter(words))
    return X



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