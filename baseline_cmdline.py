import argparse
import numpy as np
import os
import sklearn.metrics
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
  parser.add_argument("--lstmsize", type=int, help="LSTM size: max number of words for one data point")
  parser.add_argument("--epochs", type=int, help="Max number of epochs to train for")
  parser.add_argument("--test", dest='test', action='store_true', help="Running this mode will run predictions on test data using a model already trained with the given parameters.")
  parser.add_argument("--testfile", help="Filename of custom test file (test mode only)")
  parser.add_argument("--print-confusion", dest='print_confusion', action='store_true', help="Write confusion matrix to file (text mode only).")
  args = parser.parse_args()
  return args

def get_config(args):
  config = Config()
  if args.hiddensize:
    config.hidden_size = args.hiddensize
  if args.lr:
    config.learning_rate = args.lr
  if args.annealby:
    config.anneal_by = args.annealby
  if args.l2reg:
    config.l2_reg = args.l2reg
  if args.lstmsize:
    config.lstm_size = args.lstmsize
  if args.epochs:
    config.max_epochs = args.epochs
  if args.testfile:
    config.test_file = args.testfile
  return config

def main():
  args = parse_args()
  config = get_config(args)
  dataLoader = reddit_data.DataLoader(config)
  baselineModel = BaselineModel(config, dataLoader)
  if args.test:
    test_pp, test_acc, final_state, predictions, labels, probs = test_baseline_model(baselineModel)
    # filename = config.test_file + '.predictions.big'
    # with open(filename, 'w+') as f:
    #   for pred in predictions:
    #     f.write(baselineModel.class_names[pred] + '\n')
    #   f.write('test_pp ' + str(test_pp) + '\n')
    #   f.write('test_acc ' + str(test_acc) + '\n')
    #   f.write(str(predictions))
    with open(config.test_file + '.analyze', 'w+') as f:
      f.write('test_pp ' + str(test_pp) + '\n')
      f.write('test_acc ' + str(test_acc) + '\n')
      f.write('True labels\tPredictions\tPost text\n')
      for i in xrange(len(predictions)):
        f.write(baselineModel.class_names[np.argmax(labels[i])] + '\t')
        f.write(baselineModel.class_names[predictions[i]] + '\t')
        for word_idx in baselineModel.X_test[i]:
          f.write(baselineModel.num_to_word[word_idx] + ' ')
        f.write('\n')
    if args.print_confusion:
      confusion = sklearn.metrics.confusion_matrix(np.argmax(labels, 1), predictions)
      np.save(config.test_file + '.confusion', confusion)
  else:
    best_val_acc = train_baseline_model(baselineModel)
    if not os.path.exists('accuracies'):
      os.makedirs('accuracies')
    filename = 'accuracies/' + \
      'hidden=%d_l2=%f_lr=%f_anneal=%f_lstmsize=%d_epochs=%d.acc' % (config.hidden_size,
      config.l2_reg, config.learning_rate, config.anneal_by,
      config.lstm_size, config.max_epochs)
    with open(filename, 'a+') as f:
      f.write(str(config) + '\n')
      f.write('Best validation accuracy: ' + str(best_val_acc) + '\n')
      f.write('=-=' * 5 + '\n')
      f.write('=-=' * 5 + '\n')

if '__main__' == __name__:
  main()
