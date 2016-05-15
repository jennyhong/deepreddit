class Config(object):
  dropout = 0.9
  # The number of hidden units in each LSTM cell
  hidden_size = 7
  # The number of time-steps we propagate forward
  # For now, it is a constant length
  lstm_size = 10
  batch_size = 20
  learning_rate = 0.001
  max_epochs = 10
  vocab_dir = 'vocab/'
  train_file = 'data/babyTrain'
  val_file = 'data/babyVal'
  test_file = 'data/babyTest'