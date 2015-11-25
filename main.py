from __future__ import absolute_import
from __future__ import print_function

from process_data import load_challenge, vectorize_data
from sklearn.cross_validation import train_test_split
from models.memn2n import MemN2N
from itertools import chain

import tensorflow as tf
import numpy as np

# challenge data
dir_1k = "data/tasks_1-20_v1-2/en/"
dir_10k = "data/tasks_1-20_v1-2/en-10k/"
train, test = load_challenge(dir_1k, 1)

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# sentence_size = 0
# story_maxlen = 0
# sentence_size = max(map(len, chain.from_iterable(x for x, _, _ in train + test)))
# story_maxlen = max(map(len, (x for x, _, _ in train + test)))
# print("max sentence length: {}, max story length: {}".format(sentence_size, story_maxlen))

batch_size = 32
vocab_size = len(word_idx) + 1
sentence_size = 30
memory_size = 50
embedding_size = 40

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=0.10)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

stories = tf.placeholder(tf.int32, [None, memory_size, sentence_size], name="stories")
query = tf.placeholder(tf.int32, [None, sentence_size], name="query")
answer = tf.placeholder(tf.float32, [None, vocab_size], name="answer")

print(S.shape)
print(valS.shape)
print(valQ.shape)
print(valA.shape)

# params
learning_rate = 1e-2
epochs = 30
n_data = trainS.shape[0]

model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size)

# functions
pred = model(stories, query)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, answer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(answer, 1)), "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        total_cost = 0.0
        for start in range(0, batch_size, n_data):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            total_cost += sess.run(cost, feed_dict={stories: s, query: q, answer: a})
            sess.run(optimizer, feed_dict={stories: s, query: q, answer: a})
        # Cost
        print('Epoch {0}: Total Cost {1}'.format(i+1, total_cost))

        # correct prediction
        acc = sess.run(accuracy, feed_dict={stories: valS, query: valQ, answer: valA})
        print('Validation Accuracy {0}'.format(acc))
