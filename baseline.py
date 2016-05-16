import numpy as np
import sys
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from config import Config
from model import LanguageModel
from util import reddit_data

class BaselineModel(LanguageModel):

  def load_data(self):
    self.wv, word_to_num, num_to_word = reddit_data.load_wv(data_dir=self.config.vocab_dir)
    self.num_vocab, self.wv_dim = self.wv.shape
    self.class_names = ['Jokes', 'communism']
    self.config.num_classes = len(self.class_names) #TODO: make modular lolololol
    self.num_to_class = dict(enumerate(self.class_names))
    class_to_num = {v:k for k,v in self.num_to_class.iteritems()}

    self.X_train, self.y_train, self.lengths_train = reddit_data.load_dataset(self.config.train_file,
      word_to_num, class_to_num, min_length=10, full_length=10)
    self.y_train = reddit_data.generate_onehot(self.y_train, self.config.num_classes)

    self.X_val, self.y_val, self.lengths_val = reddit_data.load_dataset(self.config.val_file,
      word_to_num, class_to_num, min_length=10, full_length=10)
    self.y_val = reddit_data.generate_onehot(self.y_val, self.config.num_classes)

    self.X_test, self.y_test, self.lengths_test = reddit_data.load_dataset(self.config.test_file,
      word_to_num, class_to_num, min_length=10, full_length=10)
    self.y_test = reddit_data.generate_onehot(self.y_test, self.config.num_classes)

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

    # Calculate classification accuracy
    one_hot_prediction = tf.argmax(self.prediction, 1)
    correct_prediction = tf.equal(tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

    output = tf.reshape(tf.concat(1, self.output), [-1, self.config.num_classes])
    print 'outputs:', output.get_shape()
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)

  def run_epoch(self, session, input_data, input_labels, input_lengths, train_op=None, verbose=10):
    orig_X, orig_y = input_data, input_labels
    dropout = self.config.dropout
    if not train_op:
      train_op = tf.no_op()
      dropout = 1
    total_steps = sum(1 for x in reddit_data.data_iterator(input_data, input_lengths,
                      input_labels, self.config.batch_size, self.config.num_classes))
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    state = self.initial_state.eval()
    for step, (x, y, lengths) in enumerate(reddit_data.data_iterator(input_data, input_lengths,
                      input_labels, self.config.batch_size, self.config.num_classes)):      
      feed = {self.input_placeholder : x,
              self.labels_placeholder : y,
              self.initial_state : state,
              self.dropout_placeholder : dropout,
              self.early_stop_times : lengths}
      loss, state, total_correct, _ = session.run(
              [self.calculate_loss, self.final_state, self.correct_predictions, train_op],
              feed_dict=feed)
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)

    if verbose and step % verbose == 0:
      sys.stdout.write('\r{} / {} : pp = {}'.format(step, total_steps, np.exp(np.mean(total_loss))))
      sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
      sys.stdout.flush()
    loss = np.exp(np.mean(total_loss))
    acc = total_correct_examples / float(total_processed_examples)
    return loss, acc 

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
      start = time.time()

      train_pp, train_acc = b.run_epoch(session, b.X_train, b.y_train, b.lengths_train, train_op=b.train_step)
      val_pp, val_acc = b.run_epoch(session, b.X_val, b.y_val, b.lengths_val) # TODO: change validation to test later
      print 'Training perplexity: {}'.format(train_pp)
      print 'Training accuracy: {}'.format(train_acc)
      print 'Validation perplexity: {}'.format(val_pp)
      print 'Validation accuracy: {}'.format(val_acc)

      if val_pp < best_val_pp:
        best_val_pp = val_pp
        best_val_epoch = epoch
      print 'Total time: {}'.format(time.time() - start)

    test_pp, test_acc = b.run_epoch(session, b.X_test, b.y_test, b.lengths_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print 'Test accuracy: {}'.format(test_acc)
    print '=-=' * 5

  print 'Reached the end of the test! Nothing broke.'

if '__main__' == __name__:
  test_baseline_model()