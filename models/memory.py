from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# TODO: Bag of Words Memory
class MemoryBagOfWords(object):
    def __init__(self, sentence_size, memory_size, embedding_size):
        pass

class MemoryPositionEncoding(object):
    def __init__(self, sentence_size, memory_size, embedding_size):
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size

        # Postion Encoding from section 4.1 of [1]
        _encoding = np.zeros((sentence_size, embedding_size))
        J = sentence_size+1
        d = embedding_size+1
        for j in range(1, J):
            for k in range(1, d):
                _encoding[j-1, k-1] = 1 - (j/J) - (k/d)*(1 - 2*j/J)
        self.encoding = tf.constant(_encoding)

    @property
    def sentence_size(self):
        return self._sentence_size

    @property
    def memory_size(self):
        return self._memory_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def access_memory(self, name="memory"):
        return tf.get_variable(name,
            shape=[self.memory_size, self.embedding_size],
            initializer=tf.zeros_initializer,
            trainable=False)

    def __call__(self, sentences, scope=None):
        """
        Update memory.

        Inputs: sentences is a 3D Tensor (n_sentences, n_words, embedding_size).

        If the number of sentences is greater than the memory size we truncate
        such that the most recent sentences are used.

        Ex. We have a memory size of 50 and 167 sentences then we take the last
        50 sentences.

        Outputs: updated memory, 2D Tensor (memory_size, embedding_size)
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=True):
            # truncate to the max number of sentences we can hold
            ns = tf.shape(senences)[0]
            start = max(0, ns - memory_size)
            trunc_sentences = tf.slice(sentences, [start, 0, 0], [-1, -1, -1])
            ns = tf.shape(trunc_sentences)[0]

            # update memory
            memory = self.access_memory()
            encs = tf.reshape(tf.tile(self.encoding, ns), [ns, self.sentence_size, self.embedding_size])
            old_memory = tf.slice(memory, [ns, 0], [-1, -1])
            new_memory = tf.reduce_sum(trunc_sentences * encs, 1)
            memory.assign(np.concat(0, [new_memory, old_memory]))
            return memory

    def reset(self):
        """
        Resets memory values to all zeros.
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=True):
            memory = self.access_memory()
            memory.assign(tf.zeros_like(memory))
