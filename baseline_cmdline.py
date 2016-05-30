import argparse
import os
import tensorflow as tf

from baseline import BaselineModel, train_baseline_model, test_baseline_model
from config import Config
from util import reddit_data

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--hiddensize", type=int, help="LSTM hidden size")
  parser.add_argument("--lr", type=float, help="learning rate")
  parser.add_argument("--annealby", type=float, help="annealing")
  parser.add_argument("--l2reg", type=float, help="L2 regularization")
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  config = Config()
  dataLoader = reddit_data.DataLoader(config)
  config.hidden_size = args.hiddensize
  config.learning_rate = args.lr
  config.anneal_by = args.annealby
  config.l2_reg = args.l2reg
  with tf.Graph().as_default():
    baselineModel = BaselineModel(config, dataLoader)
    best_val_acc = train_baseline_model(baselineModel)
    if not os.path.exists('accuracies'):
      os.makedirs('accuracies')
    filename = 'accuracies/' + \
      'hidden=%d_l2=%f_lr=%f_anneal=%f.acc' % (config.hidden_size,
      config.l2_reg, config.learning_rate, config.anneal_by)
    with open(filename, 'a+') as f:
      f.write(str(config) + '\n')
      f.write('Best validation accuracy: ' + str(best_val_acc) + '\n')
      f.write('=-=' * 5 + '\n')
      f.write('=-=' * 5 + '\n')

if '__main__' == __name__:
  main()