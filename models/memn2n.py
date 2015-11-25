from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

# I: (input feature map) - convert sentence to input repr
# G: (generalization) - update current memory state given input
# O: (output feature map) - compute output given input and memory
# R: (response) - decode output to give final response to user

def position_encoding(sentence_size, embedding_size):
    # Postion Encoding from section 4.1 of [1]
    encoding = np.ones((sentence_size, embedding_size))
    J = sentence_size+1
    d = embedding_size+1
    for j in range(1, J):
        for k in range(1, d):
            encoding[j-1, k-1] = 1 - (j/J) - (k/d)*(1 - 2*j/J)
    return tf.constant(encoding, dtype=tf.float32)

class MemN2N(object):
    """
    End-To-End Memory Network as described in [1].
[1] http://arxiv.org/abs/1503.08895
    """
    def __init__(self, batch_size, vocab_size, sentence_size,
        memory_size,
        embedding_size,
        hops=1,
        init=tf.random_normal_initializer(stddev=0.1),
        name='MemN2N'):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.hops = hops
        self.name = name

        with tf.variable_scope(name):
            # Embeddings
            self.A = tf.Variable(init([vocab_size, embedding_size]))
            self.B = tf.Variable(init([vocab_size, embedding_size]))
            self.C = tf.Variable(init([vocab_size, embedding_size]))

            # output weight matrix
            self.W = tf.Variable(init([embedding_size, vocab_size]))
            self.H = tf.Variable(init([embedding_size, embedding_size]))

            # Memory
            # self.memory = tf.Variable(tf.zeros([batch_size, memory_size, sentence_size]))

            # Postion Encoding from section 4.1 of [1]
            self.encoding = position_encoding(sentence_size, embedding_size)

    def input_module(self, i_emb, u):
        # u -> (batch_size, embedding_size)
        # i_mem -> (batch_size, memory_size, embedding_size)
        # probs -> (batch_size, memory_size)
        i_mem = tf.reduce_sum(i_emb * self.encoding, 2)
        ts = tf.transpose(i_mem, perm=[0, 2, 1])
        uu = tf.expand_dims(u, -1)
        dotted = tf.reduce_sum(ts * uu, 1)
        probs = tf.nn.softmax(dotted)
        return probs

    def output_module(self, probs, o_emb):
        # probs -> (batch_size, memory_size)
        # o_mem -> (batch_size, memory_size, embedding_size)
        # o -> (batch_size, embedding_size)
        o_mem = tf.reduce_sum(o_emb * self.encoding, 2)
        o = tf.reduce_sum(o_mem * tf.expand_dims(probs, -1), 1)
        return o

    def __call__(self, stories, queries):
        with tf.variable_scope(self.name):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_0 = tf.reduce_sum(q_emb * self.encoding, 1)
            inputs = [u_0]
            for _ in range(self.hops):
                i_emb = tf.nn.embedding_lookup(self.A, stories)
                o_emb = tf.nn.embedding_lookup(self.C, stories)

                u_k = inputs[-1]
                probs = self.input_module(i_emb, u_k)
                o_k = self.output_module(probs, o_emb)
                u_k_next = o_k + tf.matmul(u_k, self.H)
                inputs.append(u_k_next)

            # (batch_size, vocab_size)
            pred = tf.matmul(inputs[-1], self.W)
            return pred
