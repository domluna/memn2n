from __future__ import absolute_import
from __future__ import print_function

from process_data import load_challenge, vectorize_data
import tensorflow as tf
import numpy as np
from models.memn2n import MemN2N

from itertools import chain

# challenge data
dir_1k = "data/tasks_1-20_v1-2/en/"
dir_10k = "data/tasks_1-20_v1-2/en-10k/"

# if time is enabled: vocab_size = vocab_size + size of memory
# total words = total words + 1, the extra 1 is for time words

# FROM Matlab impl

# word, sentence, story
# story (20, 1000, 1000)

# questions (10, 1000)

# word, question -- represents the sentence of the question
# qstory (20, 1000)
#
# if we have time enabled the last memory slot is used for that
#

train, test = load_challenge(dir_1k, 4)

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# sentence_size = 0
# story_maxlen = 0
# sentence_size = max(map(len, chain.from_iterable(x for x, _, _ in train + test)))
# story_maxlen = max(map(len, (x for x, _, _ in train + test)))
# print("max sentence length: {}, max story length: {}".format(sentence_size, story_maxlen))

vocab_size = len(word_idx) + 1
sentence_size = 30
memory_size = 50
embedding_size = 40

trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print(trainS[0])
print(trainQ[0])
print(trainA[0])

stories = tf.placeholder(tf.int32, [None, sentence_size], name="stories")
query = tf.placeholder(tf.int32, [sentence_size], name="query")
answer = tf.placeholder(tf.float32, [None, vocab_size], name="answer")

model = MemN2N(vocab_size, sentence_size, memory_size, embedding_size)

# cross entropy loss + Adam optimizer
pred = model(stories, query)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, answer))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Init all variables
init = tf.initialize_all_variables()

# TODO: fix tf.shape problems
# TODO: try to make input cleaner
# TODO: variable scoping, make sure it's legit
with tf.Session() as sess:
    sess.run(init)

    # TODO: epochs
    print(trainS[0].shape, trainQ[0].shape)
    ans = np.reshape(trainA[0], (1, -1))
    sess.run(optimizer, feed_dict={stories: trainS[0], query: trainQ[0], answer: ans})
    c = sess.run(cost, feed_dict={stories: trainS[0], query: trainQ[0], answer: ans})
    print('Cost', c)
