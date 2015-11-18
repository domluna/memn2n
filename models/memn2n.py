from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from six.moves import range

def softmax_1d(x):
    exped = tf.exp(x)
    s = tf.reduce_sum(exped)
    return exped / s

class MemN2NCell(object):
    """
    End-To-End Memory Network as described in [1].

    [1] http://arxiv.org/abs/1503.08895
    """
    def __init__(self, vocab_size, sentence_maxlen,
        emb_size=40,
        mem_size=50,
        enable_time=True,
        use_bow=False,
        randomize_time = 0.1,
        initializer=tf.random_normal_initializer(stddev=0.1)):
        pass

    def __call__(self, arg):
        pass

class MemN2N(object):
    """
    End-To-End Memory Network as described in [1].

    [1] http://arxiv.org/abs/1503.08895
    """
    def __init__(self, vocab_size, sentence_maxlen,
        emb_size=40,
        mem_size=50,
        enable_time=True,
        hops=3,
        # use_bow=False,
        randomize_time = 0.1,
        share_type='adjacent',
        initializer=tf.random_normal_initializer(stddev=0.1)):

        self.vocab_size = vocab_size
        self.mem_size = mem_size
        self.emb_size = emb_size

        # Embeddings
        self.input_emb = tf.Variable(initializer([vocab_size, emb_size]))
        self.query_emb = tf.Variable(initializer([vocab_size, emb_size]))
        self.output_emb = tf.Variable(initializer([vocab_size, emb_size]))

        # output weight matrix
        self.output_weights = tf.Variable(initializer([emb_size, vocab_size]))

        # Temporal Encoding from section 4.1 of [1]
        # TODO: figure this out
        # output_te = tf.reduce_sum(o_emb[i], 0) + self.output_time_enc[i]
        # self.input_time_enc = tf.Variable(initializer((mem_size, emb_size)))
        # self.output_time_enc = tf.Variable(initializer((mem_size, emb_size)))

        # Memory
        self.input_mem = tf.Variable(tf.zeros((mem_size, emb_size)),
            trainable=False)
        self.output_mem = tf.Variable(tf.zeros((mem_size, emb_size)),
            trainable=False)

        # Postion Encoding from section 4.1 of [1]
        _sentence_enc = np.zeros((sentence_maxlen, emb_size))
        J = sentence_maxlen+1
        d = emb_size+1
        for j in range(1, J):
            for k in range(1, d):
                _sentence_enc[j-1, k-1] = 1 - (j/J) - (k/d)*(1 - 2*j/J)
        self.sentence_enc = tf.constant(_sentence_enc)

    def _step(sentences, query, answer):
        """
        sentences -> 2D tensor
        query -> 1D tensor
        answer -> 1D tensor

        First dim of all of these is the batch size?
        """
        # batch, sentence, word, embval
        i_emb = tf.nn.embedding_lookup(self.input_emb, sentences)
        q_emb = tf.nn.embedding_lookup(self.query_emb, question)
        o_emb = tf.nn.embedding_lookup(self.output_emb, answer)

        # TODO: figure out reverse-order temporal thing
        # TODO: we can batch the sentence thing
        # if we have 50 memory slots and 160 sentences, we only use
        # the last 50 sentences. Dependent on when the question is asked!
        n = tf.shape(sentences)[0]
        for i in range(n):
            # input memory representation
            input_pe = tf.reduce_sum(i_emb[i] * self.sentence_enc, 0) #
            input_enc_val = input_pe
            input_tmp = tf.slice(self.input_mem, [0, 0], [self.mem_size-1, -1])
            new_input_mem = tf.concat(0, [tf.expand_dims(input_enc_val, 0), input_tmp])
            self.input_mem.assign(new_input_mem)

            # output memory representation
            output_pe = tf.reduce_sum(o_emb[i] * self.sentence_enc, 0)
            output_enc_val = output_pe
            output_tmp = tf.slice(self.output_mem, [0, 0], [self.mem_size-1, -1])
            new_output_mem = tf.concat(0, [tf.expand_dims(output_enc_val, 0), output_tmp])
            self.output_mem.assign(new_output_mem)

        # process question
        u = tf.reduce_sum(q_emb * self.sentence_enc, 0)

        # (mem_size x 1)
        probs = softmax_1d(tf.matmul(self.input_mem, tf.reshape(u, [self.emb_size, 1])))

        # output (emb_size x 1)
        o = tf.reduce_sum(probs * self.output_mem, 0)

        # TODO: if we have rnn sharing add matrix H to matmul u
        return o + u


    def compute(sentences, query, answer):
        # The output of each forward pass is used as the query input
        # for the next hop.
        for i in range(self.hops):
            query = self._step(sentences, query, answer)

        # TODO: do something with answer here?
        return softmax_1d(tf.matmul(tf.reshape(o + u, [self.emb_size, 1]),
            self.output_weights))
