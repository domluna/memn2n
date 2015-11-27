from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    # Postion Encoding from section 4.1 of [1]
    encoding = np.ones((embedding_size, sentence_size))
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - le / 2) * (j - ls / 2)
    encoding = 1 + 4 * encoding / le / ls
    return tf.constant(np.transpose(encoding), dtype=tf.float32)

# TODO: make it so the nil part of the embedding matrices is 0 at all times
# This seems kind of hacky -> see if there's a better way
class MemN2N(object):
    """
    End-To-End Memory Network as described in [1].
    [1] http://arxiv.org/abs/1503.08895
    """
    def __init__(self, batch_size, vocab_size, sentence_size,
        memory_size,
        embedding_size,
        hops=3,
        init=tf.random_normal_initializer(stddev=0.1),
        name='MemN2N'):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.hops = hops
        self.name = name

        # Embeddings
        with tf.variable_scope(self.name):
            self.A = tf.Variable(init([vocab_size, embedding_size]), name="A")
            self.B = tf.Variable(init([vocab_size, embedding_size]), name="B")
            self.C = tf.Variable(init([vocab_size, embedding_size]), name="C")

            # output weight matrix
            self.H = tf.Variable(init([embedding_size, embedding_size]), name="H")
            self.W = tf.Variable(init([embedding_size, vocab_size]), name="W")

            # Memory
            # self.memory = tf.Variable(tf.zeros([batch_size, memory_size, sentence_size]))

            # Postion Encoding from section 4.1 of [1]
            self.encoding = position_encoding(sentence_size, embedding_size)

    def reset_nil_embedding(self):
        with tf.name_scope("reset_nil_embedding"):
            new_A = tf.concat(0, [tf.zeros([1, self.embedding_size]), tf.slice(self.A, [1, 0], [-1, -1])])
            new_B = tf.concat(0, [tf.zeros([1, self.embedding_size]), tf.slice(self.B, [1, 0], [-1, -1])])
            new_C = tf.concat(0, [tf.zeros([1, self.embedding_size]), tf.slice(self.C, [1, 0], [-1, -1])])
            tf.assign(self.A, new_A)
            tf.assign(self.B, new_B)
            tf.assign(self.C, new_C)

    def input_module(self, i_emb, u):
        # u -> (batch_size, embedding_size)
        # i_mem -> (batch_size, memory_size, embedding_size)
        # probs -> (batch_size, memory_size)
        # uu -> (batch_size, 1, embedding_size)
        with tf.name_scope("input_module"):
            i_mem = tf.reduce_sum(i_emb * self.encoding, 2)
            u_temp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
            dotted = tf.reduce_sum(i_mem * u_temp, 2)
            return tf.nn.softmax(dotted)

    def output_module(self, probs, o_emb):
        # probs -> (batch_size, memory_size)
        # o_mem -> (batch_size, memory_size, embedding_size)
        # o -> (batch_size, embedding_size)
        with tf.name_scope("output_module"):
            o_mem = tf.reduce_sum(o_emb * self.encoding, 2)
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            o_mem_temp = tf.transpose(o_mem, [0, 2, 1])
            return tf.reduce_sum(o_mem_temp * probs_temp, 2)

    def __call__(self, stories, queries):
        with tf.variable_scope(self.name):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_0 = tf.reduce_sum(q_emb * self.encoding, 1)
            i_emb = tf.nn.embedding_lookup(self.A, stories)
            o_emb = tf.nn.embedding_lookup(self.C, stories)
            inputs = [u_0]
            for k in range(self.hops):
                u_k = inputs[k]
                probs = self.input_module(i_emb, u_k)
                o_k = self.output_module(probs, o_emb)
                u_k_next = o_k + tf.matmul(u_k, self.H)
                inputs.append(u_k_next)

            self.reset_nil_embedding()
            return tf.matmul(inputs[-1], self.W)
