from __future__ import absolute_import
from __future__ import print_function

from process_data import load_challenge, vectorize_data
from sklearn.cross_validation import train_test_split
from memn2n.memn2n import MemN2N
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
learning_rate = 0.01
epochs = 70
n_data = trainS.shape[0]

model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size)

# functions
pred = model(stories, query)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, answer))
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(answer, 1)), "float"))
pred_idxs = tf.argmax(pred, 1)
target_idxs = tf.argmax(answer, 1)

# summaries
logdir = '/tmp/memn2n-logs'
summary_validation_accuracy = tf.scalar_summary('validation_accuracy', accuracy)
summary_validation_cost = tf.scalar_summary('validation_cost', cost)
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
    for t in range(epochs):
        total_cost = 0.0
        for start in range(0, batch_size, n_data):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t, _ = sess.run([cost, train_op], feed_dict={stories: s, query: q, answer: a})
            total_cost += cost_t

        summary = sess.run(merged_summary_op, feed_dict={stories: valS, query: valQ, answer: valA})
        summary_writer.add_summary(summary, t)
        # Cost
        print('-----------------------')
        print('Epoch', t+1)
        # print('Epoch {0}: Total Cost {1:.4f}'.format(i+1, total_cost))

        # correct prediction
        val_cost = sess.run(cost, feed_dict={stories: valS, query: valQ, answer: valA})
        val_acc = sess.run(accuracy, feed_dict={stories: valS, query: valQ, answer: valA})
        print('Validation Cost   {0:10.4f}'.format(val_cost))
        print('Validation Accuracy   {0:10.4f}'.format(val_acc))

        # print('Test Accuracy         {0:10.4f}'.format(test_acc))
        # pi = sess.run(pred_idxs, feed_dict={stories: valS, query: valQ})
        # ai = sess.run(target_idxs, feed_dict={answer: valA})
        # print("Prediction Indices", pi)
        # print("Target Indices", ai)
        print('-----------------------')
