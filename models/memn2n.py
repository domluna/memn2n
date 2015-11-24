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
#
# I: This should take care of truncating/padding + sentence encoding
# G: just update the memory
# O: sentence encoding

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
        scope=None):

        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.hops = hops

        with tf.variable_scope(scope or type(self).__name__, initializer=init):

            # input_embedding = tf.get_variable('input_embedding',
            #     [vocab_size, embedding_size])
            # query_embedding = tf.get_variable('query_embedding',
            #     [vocab_size, embedding_size])
            # output_embedding = tf.get_variable('output_embedding',
            #     [vocab_size, embedding_size])
            #
            # W = tf.get_variable('W', [embedding_size, vocab_size])
            # H = tf.get_variable('H', [embedding_size, embedding_size])

            # memory = Memory(memory_size, embedding_size)

            # Postion Encoding from section 4.1 of [1]
            # encoding = position_encoding(sentence_size, embedding_size)

            # Embeddings
            self.input_embedding = tf.Variable(init([vocab_size, embedding_size]))
            self.query_embedding = tf.Variable(init([vocab_size, embedding_size]))
            self.output_embedding = tf.Variable(init([vocab_size, embedding_size]))

            # output weight matrix
            self.W = tf.Variable(init([embedding_size, vocab_size]))
            self.H = tf.Variable(init([embedding_size, embedding_size]))

            # Memory
            self.memory = Memory(memory_size, embedding_size)

            # Postion Encoding from section 4.1 of [1]
            self.encoding = position_encoding(sentence_size, embedding_size)

    def _step(self, sentences, query, reuse=None):
        """
        sentences -> 2D tensor
        query -> 1D tensor
        answer -> 1D tensor

        Assume input has been preprocessed.
        """
        # with tf.variable_scope(scope or type(self).__name__, reuse=True):

        # embeddings
        i_emb = tf.nn.embedding_lookup(self.input_embedding, sentences)
        q_emb = tf.nn.embedding_lookup(self.query_embedding, query)
        o_emb = tf.nn.embedding_lookup(self.output_embedding, sentences)

        ns = tf.shape(sentences)[0]
        ns = 2
        # ns = sentences.get_shape()[0]
        tiled = tf.tile(self.encoding, [ns, 1])
        encs = tf.reshape(tiled, [ns, self.sentence_size, self.embedding_size])

        new_memory_input = tf.reduce_sum(i_emb * encs, 1)
        new_memory_output = tf.reduce_sum(o_emb * encs, 1)

        # process query
        u = tf.reduce_sum(q_emb * self.encoding, 0)

        # (memory_size x 1)
        input_memory = self.memory(new_memory_input, scope="input")
        probs = tf.nn.softmax(tf.matmul(input_memory, tf.expand_dims(u, -1)))

        # output (embedding_size x 1)
        output_memory = self.memory(new_memory_output, scope="output")
        o = tf.reduce_sum(probs * output_memory, 0)

        hu = tf.matmul(self.H, tf.expand_dims(u, -1))
        return o + tf.squeeze(hu)

    def __call__(self, sentences, query):
        for i in range(self.hops):
            query = self._step(sentences, query)
        pred = tf.matmul(tf.expand_dims(query, 0), self.W)
        return pred
        # return tf.reshape(pred, [-1])

class Memory(object):
    def __init__(self, memory_size, embedding_size):
        self.memory_size = memory_size
        self.embedding_size = embedding_size

    def access_memory(self, name="memory"):
        return tf.get_variable(name,
            shape=[self.memory_size, self.embedding_size],
            initializer=tf.zeros_initializer,
            trainable=False)

    def reset(self, scope=None):
        """
        Resets memory values to all zeros.
        """
        with tf.variable_scope(scope or type(self).__name__):
            memory = self.access_memory()
            tf.assign(memory, tf.zeros_like(memory))

    def __call__(self, new_memory, scope=None):
        """
        Update memory.

        Inputs:
        - new_memory a 2D Tensor, (memory_size, embedding_size)

        Outputs:
        - updated memory, 2D Tensor, (memory_size, embedding_size)
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=None):
            # update memory
            ns = tf.shape(new_memory)[0]
            memory = self.access_memory()
            old_memory = tf.slice(memory, [2, 0], [-1, -1])
            tf.assign(memory, tf.concat(0, [new_memory, old_memory]))
            return memory
