import tensorflow as tf

from baseline import BaselineModel, train_baseline_model, test_baseline_model
from config import Config

def main():
  hidden_sizes = [10, 20, 30]
  for hidden_size in hidden_sizes:
    config = Config()
    config.hidden_size = hidden_size
    with tf.Graph().as_default():
      baselineModel = BaselineModel(config)
      train_baseline_model(baselineModel)
    print config

if '__main__' == __name__:
  main()