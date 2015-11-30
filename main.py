from __future__ import absolute_import
from __future__ import print_function

from process_data import load_challenge, vectorize_data
from sklearn import cross_validation, metrics
from memn2n.memn2n import MemN2N
from itertools import chain

import tensorflow as tf
import numpy as np

# random seed
seed = 1234

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
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=seed)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

stories = tf.placeholder(tf.int32, [None, memory_size, sentence_size], name="stories")
query = tf.placeholder(tf.int32, [None, sentence_size], name="query")
answer = tf.placeholder(tf.int32, [None, vocab_size], name="answer")

print(S.shape)
print(valS.shape)
print(valQ.shape)
print(valA.shape)

# params
epochs = 70
n_data = trainS.shape[0]

tf.set_random_seed(seed)

# summaries
#logdir = '/tmp/memn2n-logs'
#summary_validation_accuracy = tf.scalar_summary('validation_error', error_op)
#summary_validation_cost = tf.scalar_summary('validation_cost', cost)
#merged_summary_op = tf.merge_all_summaries()
#summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)

# numerics_op = tf.add_check_numerics_ops()
# print(numerics_op)

val_labels = np.argmax(valA, axis=1)

#model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size)
with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size, session=sess)

    for t in range(epochs):
        total_cost = 0.0
        for start in range(0, batch_size, n_data):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.partial_fit(s, q, a)
            total_cost += cost_t

        #summary = sess.run(merged_summary_op, feed_dict={stories: valS, query: valQ, answer: valA})
        #summary_writer.add_summary(summary, t)
        # Cost

        val_preds = model.predict(valS, valQ)
        val_acc = metrics.accuracy_score(val_preds, val_labels)
        print('-----------------------')
        print('Epoch', t+1)
        print('Total Cost:', total_cost)
        print('Validation Accuracy:', val_acc)
        print("Prediction Indices", val_preds)
        print("Labels Indices", val_labels)
        print('-----------------------')
