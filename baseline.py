import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import time

from config import Config
from model import LanguageModel
from util import reddit_data

class BaselineModel(LanguageModel):

  def load_data(self, dataLoader):
    self.wv = dataLoader.wv
    self.num_vocab, self.wv_dim = dataLoader.num_vocab, dataLoader.wv_dim
    self.class_names = dataLoader.class_names
    self.config.num_classes = dataLoader.config.num_classes

    self.X_train, self.y_train, self.lengths_train = dataLoader.X_train, dataLoader.y_train, dataLoader.lengths_train
    self.X_val, self.y_val, self.lengths_val = dataLoader.X_val, dataLoader.y_val, dataLoader.lengths_val
    self.X_test, self.y_test, self.lengths_test = dataLoader.X_test, dataLoader.y_test, dataLoader.lengths_test

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
    with tf.variable_scope('classifier') as scope:
      U = tf.get_variable("U", (self.config.hidden_size * 2, self.config.num_classes))
      b = tf.get_variable("b", (self.config.num_classes))
    output = tf.matmul(rnn_output, U) + b
    print 'classifier_output has shape', output.get_shape()
    return output

  def add_loss_op(self, classifier_output):
    print classifier_output.get_shape(), self.labels_placeholder.get_shape()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classifier_output, self.labels_placeholder))
    with tf.variable_scope('classifier', reuse=True) as scope:
      U = tf.get_variable("U")
    loss += self.config.l2_reg * tf.nn.l2_loss(U)
    return loss

  def add_training_op(self, loss):
    opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    train_op = opt.minimize(loss)
    return train_op

  def __init__(self, config, dataLoader):
    self.config = config
    self.learning_rate = self.config.learning_rate
    self.load_data(dataLoader)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_output = self.add_rnn_model(self.inputs)
    # Use the last output of the RNN to classify
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

  def run_epoch(self, session, input_data, input_labels, input_lengths,
    train_op=None, verbose=10, new_model=False):
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
      loss, state, total_correct, predictions = session.run(
              [self.calculate_loss, self.final_state, self.correct_predictions, self.one_hot_prediction],
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
    return loss, acc, total_loss, state, predictions

def train_baseline_model(model):
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_acc = 0
    best_val_epoch = 0

    # for lr annealing
    prev_epoch_loss = float('inf')

    session.run(init)
    for epoch in xrange(model.config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()

      train_pp, train_acc, loss_history, _, _ = model.run_epoch(session,
        model.X_train, model.y_train, model.lengths_train, train_op=model.train_step)
      val_pp, val_acc, _, _, _ = model.run_epoch(session,
        model.X_val, model.y_val, model.lengths_val)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Training accuracy: {}'.format(train_acc)
      print 'Validation perplexity: {}'.format(val_pp)
      print 'Validation accuracy: {}'.format(val_acc)

      # lr annealing
      epoch_loss = np.mean(loss_history)
      if epoch_loss > prev_epoch_loss * model.config.anneal_threshold:
        model.learning_rate /= model.config.anneal_by
        print 'annealed learning rate to %f' % model.learning_rate
      prev_epoch_loss = epoch_loss

      if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch
        if not os.path.exists(model.config.weights_dir):
          os.makedirs(model.config.weights_dir)
        saver.save(session, model.config.weights_file())

      print 'Total time: {}'.format(time.time() - start)
  return best_val_acc

def test_baseline_model(model):
  saver = tf.train.Saver()
  with tf.Session() as session:
    saver.restore(session, model.config.weights_file())
    return model.run_epoch(session, model.X_test, model.y_test, model.lengths_test)

def main():
  config = Config()
  dataLoader = reddit_data.DataLoader(config)
  baselineModel = BaselineModel(config, dataLoader)
  best_val_pp = train_baseline_model(baselineModel)
  test_baseline_model(baselineModel)
  print config
  print best_val_pp

  print 'Reached the end of the test! Nothing broke.'

if '__main__' == __name__:
  main()
