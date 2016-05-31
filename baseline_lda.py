import collections
import numpy as np
import lda

from config import Config
from util import vocab

config = Config()

num_topics = len(config.class_names)
num_iters = 1000
num_top_words = 15


word_to_num, num_to_word = vocab.get_word_indices(config.vocab_dir, 'redditVocabBase.txt')
num_to_word = np.array([num_to_word[i] for i in xrange(len(word_to_num))])

X = np.zeros((0, len(word_to_num)))
with open(config.train_file) as train_file:
	for line in train_file:
		items = line.strip().split('\t')
		words = items[9].split()[1:]
		words = [word if word in word_to_num else '<UNK>' for word in words]
		word_indices = [word_to_num[word] for word in words]
		counts = collections.Counter(word_indices)
		x_1 = np.array([counts[i] if i in counts else 0 for i in xrange(len(word_to_num))])
		if sum(x_1) > 0:
			x_1 = np.reshape(x_1, (1, len(x_1)))
			X = np.append(X, x_1, axis=0)
X = X.astype(int)

model = lda.LDA(n_topics=num_topics, n_iter=num_iters, random_state=1)
model.fit(X)

with open('lda_output.txt', 'a+') as f:
  for i, words in enumerate(model.topic_word_):
    topic_words = num_to_word[np.argsort(words)][:-(num_top_words+1):-1]
    f.write('Topic {}: {}\n'.format(i, ' '.join(topic_words)))
