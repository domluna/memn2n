from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_challenge, vectorize_data
from sklearn import cross_validation, metrics
from memn2n.memn2n import MemN2N
from itertools import chain

import tensorflow as tf
import numpy as np

seed = 1234
challenge_n = 2
print("Started Challenge:", challenge_n)

# challenge data
data_dir = "data/tasks_1-20_v1-2/en/"
train, test = load_challenge(data_dir, challenge_n)

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

batch_size = 32
vocab_size = len(word_idx) + 1
max_story_size = max(map(len, (s for s, _, _ in train + test)))
mean_story_size = int(np.mean(map(len, (s for s, _, _ in train + test))))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train + test)))
query_size = max(map(len, chain.from_iterable(q for _, q, _ in train + test)))
sentence_size = max(query_size, sentence_size)
memory_size = min(50, mean_story_size)
embedding_size = 20

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=seed)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print(trainS.shape)

# params
epochs = 200
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(seed)

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, embedding_size, session=sess, hops=3)
    for t in range(epochs):
        total_cost = 0.0
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a)
            total_cost += cost_t

        train_preds = []
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            pred = model.predict(s, q)
            train_preds += list(pred)

        val_preds = model.predict(valS, valQ)
        train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
        val_acc = metrics.accuracy_score(val_preds, val_labels)

        print('-----------------------')
        print('Epoch', t+1)
        print('Total Cost:', total_cost)
        print('Training Accuracy:', train_acc)
        print('Validation Accuracy:', val_acc)
        print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)

