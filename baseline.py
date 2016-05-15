import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.ops.seq2seq import sequence_loss


from model import LanguageModel
from util import reddit_data
import sys

class Config(object):
  dropout = 0.9
  # The number of hidden units in each LSTM cell
  hidden_size = 7
  # The number of time-steps we propagate forward
  # For now, it is a constant length
  lstm_size = 10
  batch_size = 51
  learning_rate = 0.001
  max_epochs = 1

class BaselineModel(LanguageModel):

  def load_data(self):
    self.wv, word_to_num, num_to_word = reddit_data.load_wv()
    self.num_vocab, self.wv_dim = self.wv.shape
    self.class_names = ['Jokes', 'communism']
    self.config.num_classes = len(self.class_names) #TODO: make modular lolololol
    self.num_to_class = dict(enumerate(self.class_names))
    class_to_num = {v:k for k,v in self.num_to_class.iteritems()}

    self.X_train, self.y_train, self.lengths_train = reddit_data.load_dataset('data/babyTrain',
      word_to_num, class_to_num, min_length=10, full_length=10)
    self.y_train = generate_labels_tensor(self.y_train, self.config.num_classes)
    print self.X_train.shape, self.y_train.shape

    self.X_test, self.y_test, self.lengths_test = reddit_data.load_dataset('data/babyTest',
      word_to_num, class_to_num, min_length=10, full_length=10)
    self.y_test = generate_labels_tensor(self.y_test, self.config.num_classes)
    print self.X_test.shape, self.y_test.shape

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.lstm_size])
    self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.num_classes])
    self.dropout_placeholder = tf.placeholder(tf.float32)
    self.early_stop_times = tf.placeholder(tf.int32, [None]) #??

  def add_embedding(self):
    with tf.device('/cpu:0'), tf.variable_scope('embedding'):
      embeddings = tf.Variable(tf.convert_to_tensor(self.wv, dtype=tf.float32), name="Embedding")
      window = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      inputs = [tf.squeeze(inpt, squeeze_dims=[1]) for inpt in tf.split(1, self.config.lstm_size, window)]
      return inputs

  def add_rnn_model(self, inputs):
    cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)  
    #print inputs[0].get_shape()
    self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)
    outputs, self.final_state = tf.nn.rnn(cell, inputs, initial_state=None, sequence_length=self.early_stop_times, dtype=tf.float32)
    print 'initial/final state shapes:', self.initial_state.get_shape(), self.final_state.get_shape()
    return self.final_state

  def add_classifier_model(self, rnn_output):
    print 'rnn_output has shape', rnn_output.get_shape()
    U = tf.get_variable("U", (self.config.hidden_size * 2, self.config.num_classes))
    b = tf.get_variable("b", (self.config.num_classes))
    output = tf.matmul(rnn_output, U) + b
    print 'classifier_output has shape', output.get_shape()
    return output

  def add_loss_op(self, classifier_output):
    print classifier_output.get_shape(), self.labels_placeholder.get_shape()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classifier_output, self.labels_placeholder))
    return loss

  def add_training_op(self, loss):
    opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
    train_op = opt.minimize(loss)
    return train_op

  def __init__(self, config):
    self.config = config
    self.load_data()
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_output = self.add_rnn_model(self.inputs)
    # Use the last hidden state of the RNN to classify
    self.output = self.add_classifier_model(self.rnn_output)
    self.prediction = tf.nn.softmax(tf.cast(self.output, 'float64'))
    output = tf.reshape(tf.concat(1, self.output), [-1, self.config.num_classes])
    print 'outputs:', output.get_shape()
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)

  def run_epoch(self, session, x, y, lengths, train_op=None):
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    #total_steps = sum(1 for x in ptb_iterator(data, self.config.batch_size, self.config.lstm_size))
    batch_len = len(x) // self.config.batch_size
    total_steps = (batch_len - 1) // self.config.lstm_size
    total_loss = []
    state = self.initial_state.eval()
    #for step, (x, y) in enumerate(ptb_iterator(data, self.config.batch_size, self.config.lstm_size)):
    print x.shape, y.shape
    
    feed = {self.input_placeholder : x,
      self.labels_placeholder : y,
      self.initial_state : state,
      self.dropout_placeholder : self.config.dropout,
      self.early_stop_times : lengths} # TODO: get num_time_steps for each input in x
    loss, state, _ = session.run([self.calculate_loss, self.final_state, train_op], feed_dict=feed)
    total_loss.append(loss)

    verbose = True
    """
    if verbose:#and step % verbose == 0:
      sys.stdout.write('\r{} / {} : pp = {}'.format(total_steps, np.exp(np.mean(total_loss))))
      sys.stdout.flush()
    """

    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def ptb_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    yield (x, y)

def generate_labels_tensor(y, num_classes):
  y_t = np.zeros((len(y), num_classes))
  y_t[np.arange(len(y)), y] = 1
  return y_t

def test_baseline_model():
  c = Config()
  b = BaselineModel(c)
  init = tf.initialize_all_variables()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0

    session.run(init)
    for epoch in xrange(c.max_epochs):
      print 'Epoch {}'.format(epoch)
      #start = time.time()
      train_pp = b.run_epoch(session, b.X_train, b.y_train, b.lengths_train, train_op=b.train_step)
      test_pp = b.run_epoch(session, b.X_test, b.y_test, b.lengths_test) # TODO: change validation to test later
      print 'Training perplexity: {}'.format(train_pp)
      print 'Testing perplexity: {}'.format(test_pp)

  print 'Reached the end of the test! Nothing broke.'

if '__main__' == __name__:
  test_baseline_model()