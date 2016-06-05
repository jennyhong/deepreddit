import collections
import lda
import nltk
import numpy as np
import scipy.sparse
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer


from config import Config
from util import vocab

config = Config()

num_topics = len(config.class_names)
num_iters = 1000
num_top_words = 15

stop_words = set()
print 'Creating counts matrix via CountVectorizer'
with open('stopwords.txt', 'r') as f:
  for line in f:
    stop_words.add(line.strip())
stop_words = stop_words.union(text.ENGLISH_STOP_WORDS)
stop_words = stop_words.union(['eos', '<EOS>', '<UNK>', 'unk', 'like', 'just', 'special', '<SPECIAL>', 'did', 'll', 've', 's', 't', 'nt'])

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.snowball.SnowballStemmer('english')

for i in xrange(5):
  train_filename = '../../dfs-leon/rc/data/test31classlevels%d.med' % i
  print 'Reading in', train_filename
  vectorizer = CountVectorizer(min_df=10, stop_words=stop_words)
  X = []
  with open(train_filename) as train_file:
    for line in train_file:
      items = line.strip().split('\t')
      words = items[-1].split()[1:]
      words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
      words = [word for word in words if len(word) > 2]
      if len(words) < 10: continue
      X.append(' '.join(words[:20]))
  X = vectorizer.fit_transform(X)
  print 'Beginning LDA fitting'
  model = lda.LDA(n_topics=num_topics, n_iter=num_iters, random_state=1)
  model.fit(X)
  output_filename = 'lda_output_levels%d.med.clipped' % i
  print 'Writing to file', output_filename
  num_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
  num_to_word = np.array([num_to_word[idx] for idx in xrange(len(num_to_word))])
  with open(output_filename, 'w+') as f:
    for idx, words in enumerate(model.topic_word_):
      topic_words = num_to_word[np.argsort(words)][:-(num_top_words+1):-1]
      f.write('Topic {}: {}\n'.format(idx, ' '.join(topic_words)))

