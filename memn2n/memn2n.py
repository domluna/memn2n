from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    # Postion Encoding from section 4.1 of [1]
    E = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            E[i-1, j-1] = (i - le / 2) * (j - ls / 2)
    E = 1 + 4 * E / le / ls
    return np.transpose(E)

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
        clip_norm=40,
        epochs=20,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(1e-2),
        session=tf.Session(),
        name='MemN2N'):

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._clip_norm = clip_norm
        self._epochs = epochs
        self._init = initializer
        self._opt = optimizer
        self._name = name

        g = tf.Graph()
        self._g = g
        with self._g.as_default():
        
            self._build_vars()
            self._build_inputs()
            
            init_op = tf.initialize_all_variables()
            
            self._sess = session
            sess.run(init_op)
            
    def _build_inputs(self):
        self.stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self.queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self.answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        
    def _build_vars(self):
        # Embeddings
        nil_word_slot = tf.stop_gradient(tf.zeros([1, self._embedding_size]))
        A = tf.concat(0, [ nil_word_slot, self._init(self._vocab_size-1, self._embedding_size) ])
        B = tf.concat(0, [ nil_word_slot, self._init(self._vocab_size-1, self._embedding_size) ])
        C = tf.concat(0, [ nil_word_slot, self._init(self._vocab_size-1, self._embedding_size) ])
        self.A = tf.Variable(A, name="A")
        self.B = tf.Variable(B, name="B")
        self.C = tf.Variable(C, name="C")

        # Weight Matrices
        self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
        self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

        # Postion Encoding from section 4.1 of [1]
        self.E = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="position_encoding")
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
    def _update_params(self):
        pass
            
    def fit(self):
        pass

    def partial_fit(self):
        pass
        
    def predict(self):
        pass
        
    def _inference(self):
        q_emb = tf.nn.embedding_lookup(self.B, queries)
        u_0 = tf.reduce_sum(q_emb * self.E, 1)
        inputs = [u_0]
        for k in range(self.hops):
            u_k = inputs[k]
            i_emb = tf.nn.embedding_lookup(self.A, stories)
            o_emb = tf.nn.embedding_lookup(self.C, stories)
            probs = self._input_module(i_emb, u_k)
            o_k = self._output_module(probs, o_emb)
            u_k_next = tf.nn.relu(o_k + tf.matmul(u_k, self.H))
            inputs.append(u_k_next)

        out = tf.matmul(inputs[-1], self.W)
        return out
        
    def _loss(self, logits, targets):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets, name="xentropy")
        loss_func = tf.reduce_sum(cross_entropy, name="xentropy_sum")
        return loss_func

    def _input_module(self, i_emb, u):
        # u -> (batch_size, embedding_size)
        # i_mem -> (batch_size, memory_size, embedding_size)
        # probs -> (batch_size, memory_size)
        # uu -> (batch_size, 1, embedding_size)
        with tf.name_scope("input_module"):
            i_mem = tf.reduce_sum(i_emb * self.E, 2)
            u_temp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
            dotted = tf.reduce_sum(i_mem * u_temp, 2)
            return tf.nn.softmax(dotted)

    def _output_module(self, probs, o_emb):
        # probs -> (batch_size, memory_size)
        # o_mem -> (batch_size, memory_size, embedding_size)
        # o -> (batch_size, embedding_size)
        with tf.name_scope("output_module"):
            o_mem = tf.reduce_sum(o_emb * self.E, 2)
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            o_mem_temp = tf.transpose(o_mem, [0, 2, 1])
            return tf.reduce_sum(o_mem_temp * probs_temp, 2)

            
