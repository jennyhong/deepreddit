# TODO: Import stuff
from model import LanguageModel
from util import reddit_data

class Config(object):
  hidden_size = 10

class BaselineModel(LanguageModel):

  def load_data(self):
    self.wv, word_to_num, num_to_word = reddit_data.load_wv()
    self.class_names = ['Jokes', 'communism']
    self.num_to_class = dict(enumerate(self.class_names))
    class_to_num = {v:k for k,v in self.num_to_class.iteritems()}

    self.X_train, self.y_train = reddit_data.load_dataset('data/babyTrain',
      word_to_num, class_to_num, min_sentence_length=5, full_sentence_length=5)