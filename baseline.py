import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from model import LanguageModel
from util import reddit_data

class Config(object):
  dropout = 0.9
  # The number of hidden units in each LSTM cell
  hidden_size = 10
  # The number of time-steps we propagate forward
  # For now, it is a constant length
  lstm_size = 3
  batch_size = 5

class BaselineModel(LanguageModel):

  def load_data(self):
    self.wv, word_to_num, num_to_word = reddit_data.load_wv()
    self.num_vocab, self.wv_dim = self.wv.shape
    self.class_names = ['Jokes', 'communism']
    self.num_to_class = dict(enumerate(self.class_names))
    class_to_num = {v:k for k,v in self.num_to_class.iteritems()}

    self.X_train, self.y_train = reddit_data.load_dataset('data/babyTrain',
      word_to_num, class_to_num, min_sentence_length=5, full_sentence_length=5)

    self.X_test, self.y_test = reddit_data.load_dataset('data/babyTest',
      word_to_num, class_to_num, min_sentence_length=5, full_sentence_length=5)

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.lstm_size])
    self.labels_placeholder = tf.placeholder(tf.int32, [None, self.config.lstm_size])
    self.dropout_placeholder = tf.placeholder(tf.float32)

  def add_embedding(self):
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
      embeddings = tf.get_variable("Embedding", (self.num_vocab, self.config.hidden_size))
      window = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      inputs = [tf.squeeze(inpt) for inpt in tf.split(1, self.config.lstm_size, window)]
      return inputs

  def add_rnn_model(self, inputs):
    lstm = rnn_cell.BasicLSTMCell(self.config.lstm_size)
    state = tf.zeros([self.config.batch_size, lstm.state_size])
    return state

  def __init__(self, config):
    self.config = config
    self.load_data()
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_rnn_model(self.inputs)
    # Use the last hidden state of the RNN to classify
    # y = self.add_classifier_model(self.rnn_outputs[-1])

def test_baseline_model():
  c = Config()
  b = BaselineModel(c)
  print 'Reached the end of the test! Nothing broke.'

if '__main__' == __name__:
  test_baseline_model()