class Config(object):
  batch_size = 50
  class_names = ['Jokes', 'communism']
  # class_names = ['science', 'space', 'creepy', 'Documentaries', 'gaming', 'nosleep', 'sports', 'television', 'askscience', 'books', 'history', 'TwoXChromosomes', 'dataisbeautiful', 'InternetIsBeautiful', 'Jokes', 'nottheonion', 'tifu', 'UpliftingNews', 'Art', 'EarthPorn', 'OldSchoolCool', 'photoshopbattles', 'DIY', 'food', 'GetMotivated', 'LifeProTips', 'personalfinance', 'philosophy', 'WritingPrompts', 'Futurology', 'gadgets']
  debug = False
  dropout = 0.9 # not currently used
  # The number of hidden units in each LSTM cell
  hidden_size = 20
  # The number of time-steps we propagate forward
  # For now, it is a constant length
  lstm_size = 10
  learning_rate = 0.001
  max_epochs = 2
  l2_reg = 0.1

  vocab_dir = 'vocab/'
  data_dir = 'data/'
  train_file = data_dir + 'babyTrain'
  val_file = data_dir + 'babyVal'
  test_file = data_dir + 'babyTest'

  weights_dir = data_dir + 'weights/'

  def weights_file(self):
    weights_filename = 'hidden=%d_l2=%f_lr=%f.weights' % (self.hidden_size,
      self.l2_reg, self.learning_rate)
    return self.weights_dir + weights_filename

  def __str__(self):
    return '\n'.join([
        'Batch size ' + str(self.batch_size),
        'Dropout ' + str(self.dropout),
        'Hidden size ' + str(self.hidden_size),
        'LSTM size ' + str(self.lstm_size),
        'Learning rate ' + str(self.learning_rate),
        'L2 Reg ' + str(self.l2_reg),
        'Training dataset ' + self.train_file
      ])
