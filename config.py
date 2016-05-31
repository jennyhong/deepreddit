class Config(object):
  batch_size = 50
  class_names = ['science', 'space', 'creepy', 'Documentaries', 'gaming', 'nosleep', 'sports', 'television', 'askscience', 'books', 'history', 'TwoXChromosomes', 'dataisbeautiful', 'InternetIsBeautiful', 'Jokes', 'nottheonion', 'tifu', 'UpliftingNews', 'Art', 'EarthPorn', 'OldSchoolCool', 'photoshopbattles', 'DIY', 'food', 'GetMotivated', 'LifeProTips', 'personalfinance', 'philosophy', 'WritingPrompts', 'Futurology', 'gadgets']
  debug = False
  dropout = 0.9 # not currently used
  hidden_size = 20 # The number of hidden units in each LSTM cell
  lstm_size = 30
  learning_rate = 0.001
  max_epochs = 30
  l2_reg = 0.001
  anneal_by = 1.2
  anneal_threshold = 0.99

  dataset = 'small'

  vocab_dir = '../../dfs-leon/rc/vocab/'
  data_dir = '../../dfs-leon/rc/data/'
  train_file = data_dir + 'train31class.small'
  val_file = data_dir + 'val31class.small'
  test_file = data_dir + 'test31class.small'

  weights_dir = data_dir + 'weights/'

  def weights_file(self):
    weights_filename = 'hidden=%d_l2=%f_lr=%f_anneal=%f_lstmsize=%d_epochs=%d.%s.weights' % (self.hidden_size, \
      self.l2_reg, self.learning_rate, self.anneal_by, \
      self.lstm_size, self.max_epochs, self.dataset)
    return self.weights_dir + weights_filename

  def __str__(self):
    return '\n'.join([
        'Batch size ' + str(self.batch_size),
        'Dropout ' + str(self.dropout),
        'Hidden size ' + str(self.hidden_size),
        'LSTM size ' + str(self.lstm_size),
        'Learning rate ' + str(self.learning_rate),
        'L2 Reg ' + str(self.l2_reg),
        'Annealing ' + str(self.anneal_by),
        'Training dataset ' + self.train_file
      ])
