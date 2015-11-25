from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from six.moves import range

# TODO: think about the core components structure
# I: (input feature map) - convert sentence to input repr
# G: (generalization) - update current memory state given input
# O: (output feature map) - compute output given input and memory
# R: (response) - decode output to give final response to user

def position_encoding(sentence_size, embedding_size):
    # Postion Encoding from section 4.1 of [1]
    encoding = np.zeros((sentence_size, embedding_size))
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
    def __init__(self, vocab_size, sentence_size,
        memory_size,
        embedding_size,
        hops=1,
        init=tf.random_normal_initializer(stddev=0.1),
        name='MemN2N'):

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
            self.input_memory = Memory(memory_size, embedding_size, name="input_memory")
            self.output_memory = Memory(memory_size, embedding_size, name="output_memory")

            # Postion Encoding from section 4.1 of [1]
            self.encoding = position_encoding(sentence_size, embedding_size)

    def _step(self, ns, i_emb, q_emb, o_emb):
        """
        sentences -> 2D tensor
        query -> 1D tensor
        answer -> 1D tensor

        Assume input has been preprocessed.
        """
        tiled = tf.tile(self.encoding, tf.pack([ns, 1]))
        encs = tf.reshape(tiled, tf.pack([ns, self.sentence_size, self.embedding_size]))

        new_memory_input = tf.reduce_sum(i_emb * encs, 1)
        new_memory_output = tf.reduce_sum(o_emb * encs, 1)

        # process query
        u = tf.reduce_sum(q_emb * self.encoding, 0)

        # self.input_memory(new_memory_input)
        probs = tf.nn.softmax(tf.matmul(new_memory_input, tf.expand_dims(u, -1)))

        # self.output_memory(new_memory_output)
        o = tf.reduce_sum(probs * new_memory_output, 0)

        # hu = tf.matmul(self.H, tf.expand_dims(u, -1))
        # return o + tf.squeeze(hu)
        return o + u

    def __call__(self, sentences, query):
        with tf.variable_scope(self.name):
            # embeddings
            i_emb = tf.nn.embedding_lookup(self.A, sentences)
            q_emb = tf.nn.embedding_lookup(self.B, query)
            o_emb = tf.nn.embedding_lookup(self.C, sentences)

            ns = tf.shape(sentences)[0]
            # res = None
            # for i in range(self.hops):
            res = self._step(ns, i_emb, q_emb, o_emb)

            pred = tf.matmul(tf.expand_dims(res, 0), self.W)

            # reset memory
            # self.input_memory.reset()
            # self.output_memory.reset()

            return pred

class Memory(object):
    def __init__(self, memory_size, embedding_size, name="Memory"):
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.name = name

        with tf.variable_scope(self.name):
            self.memory = tf.Variable(tf.zeros([memory_size, embedding_size]), trainable=False)

    def reset(self):
        """
        Resets memory values to all zeros.
        """
        with tf.variable_scope(self.name):
            tf.assign(self.memory, tf.zeros_like(self.memory))

    def __call__(self, new_memory):
        """
        Update memory.

        Inputs:
        - new_memory: 2D Tensor, (memory_size, embedding_size)

        Outputs:
        - Updated memory, 2D Tensor, (memory_size, embedding_size)
        """
        with tf.variable_scope(self.name):
            # update memory
            ns = tf.shape(new_memory)[0]
            old_memory = tf.slice(self.memory, tf.pack([ns, 0]), [-1, -1])
            updated_memory = tf.concat(0, [new_memory, old_memory])
            tf.assign(self.memory, updated_memory)
