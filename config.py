class Config(object):
  batch_size = 50
  class_names = ['science', 'space', 'creepy', 'Documentaries', 'gaming', 'nosleep', 'sports', 'television', 'askscience', 'books', 'history', 'TwoXChromosomes', 'dataisbeautiful', 'InternetIsBeautiful', 'Jokes', 'nottheonion', 'tifu', 'UpliftingNews', 'Art', 'EarthPorn', 'OldSchoolCool', 'photoshopbattles', 'DIY', 'food', 'GetMotivated', 'LifeProTips', 'personalfinance', 'philosophy', 'WritingPrompts', 'Futurology', 'gadgets']
  debug = False
  dropout = 0.9 # not currently used
  # The number of hidden units in each LSTM cell
  hidden_size = 100
  # The number of time-steps we propagate forward
  # For now, it is a constant length
  lstm_size = 10
  learning_rate = 0.001
  max_epochs = 10

  vocab_dir = '../../dfs-leon/rc/vocab/'
  train_file = '../../dfs-leon/rc/data/train31class.small'
  val_file = '../../dfs-leon/rc/data/val31class.small'
  test_file = '../../dfs-leon/rc/data/test31class.small'
