from multiprocessing import Pool
import tensorflow as tf

from baseline import BaselineModel, train_baseline_model, test_baseline_model
from config import Config
from util import reddit_data

def train_baseline_thread(config, dataLoader):
  model = BaselineModel(config, dataLoader)
  return train_baseline_model(model)

def main():
  # dataLoader is created outside, because for now, none of the 
  # hyperparameters we search over below affects the reading of the data.
  # Reading data is incredibly slow, so we want to do it only once over
  # all models, if the data will be the same over all models.
  # If we ever do change things like LSTM size, which affect reading in 
  # the data, that should be the outermost loop, so that we have to
  # read in data as few times as possible
  pool = Pool()
  default_config = Config()
  dataLoader = reddit_data.DataLoader(default_config)
  with open('results.txt', 'a+') as f:
    hidden_sizes = [10, 20, 30, 50]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    annealing = [1.0, 1.2, 1.5]
    l2_regs = [0.001, 0.01, 0.1, 1.0]
    for hidden_size in hidden_sizes:
      for learning_rate in learning_rates:
        for anneal_by in annealing:
          for l2_reg in l2_regs:
            config = Config()
            config.hidden_size = hidden_size
            config.learning_rate = learning_rate
            config.anneal_by = anneal_by
            config.l2_reg = l2_reg
            with tf.Graph().as_default():
              # baselineModel = BaselineModel(config, dataLoader)
              # best_val_pp = train_baseline_model(baselineModel)
              result = pool.apply_async(train_baseline_thread, [config, dataLoader])
              best_val_acc = result.get()
              f.write(str(config) + '\n')
              f.write('Best validation accuracy: ' + str(best_val_acc) + '\n')
              f.write('=-=' * 5 + '\n')
              f.write('=-=' * 5 + '\n')

if '__main__' == __name__:
  main()