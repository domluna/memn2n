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

def zero_nil_slot(t, name=None):
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        #s = tf.shape(t)[1]
        z = tf.zeros([1, 20])
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

# TODO: change this to not be magic number
# from paper: variance = n / (1 + t)^v, n = {0.01, 0.3, 1.0}, v = 0.55
# training step t
def add_gradient_noise(t, stddev, name=None):
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """
    End-To-End Memory Network as described in [1].
    [1] http://arxiv.org/abs/1503.08895
    """
    def __init__(self, batch_size, vocab_size, sentence_size,
        memory_size,
        embedding_size,
        hops=1,
        clip_norm=40,
        epochs=100,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
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

        self.build_inputs()
        self.build_vars()

        # loss op
        logits = self.forward(self._stories, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="xentropy")
        loss_op = tf.reduce_sum(cross_entropy, name="loss_op")

        # training op
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads_and_vars = self._opt.compute_gradients(loss_op, vars)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name == self.A.name or v.name == self.B.name or v.name == self.C.name:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self._clip_norm), gv[1]) for gv in nil_grads_and_vars]
        noised_grads_and_vars = [(add_gradient_noise(gv[0], 0.1), gv[1]) for gv in clipped_grads_and_vars]
        train_op = self._opt.apply_gradients(nil_grads_and_vars, global_step=self.global_step, name="train_op")

        # predict op
        predict_op = tf.argmax(logits, 1, name="predict_op")

        self._loss_op = loss_op
        self._predict_op = predict_op
        self._train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        
    def build_vars(self):
        # Embeddings
        nil_word_slot = tf.stop_gradient(tf.zeros([1, self._embedding_size]))
        A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
        B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
        C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])

        self.A = tf.Variable(A, name="A")
        self.B = tf.Variable(B, name="B")
        self.C = tf.Variable(C, name="C")

        # Weight Matrices
        self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
        self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

        # Postion Encoding from section 4.1 of [1]
        self.E = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="position_encoding")

        # Global step used in training
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    def fit(self, stories, queries, answers):
        n_data = np.shape(stories)[0]
        for t in range(self._epochs):
            start = 0
            total_loss = 0.0
            for start in range(0, n_data, self._batch_size):
                end = start + self_.batch_size
                train_S = stories[start:end]
                train_Q = queries[start:end]
                train_A = answers[start:end]
                feed_dict = {self._stories: train_S, self._queries: train_Q, self._answers: train_A}
                loss_t, _ = self._sess.run([self._loss_op, self._train_op], feed_dict=feed_dict)
                total_cost += loss_t

            print('Epoch %d: training loss %f', t+1, total_cost)

    def partial_fit(self, stories, queries, answers):
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss_t, _ = self._sess.run([self._loss_op, self._train_op], feed_dict=feed_dict)
        return loss_t
        
    def predict(self, stories, queries):
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self._predict_op, feed_dict=feed_dict)

    def forward(self, stories, queries):
        """
        A forward pass of the Memory Network.
        """
        q_emb = tf.nn.embedding_lookup(self.B, queries)
        u_k = tf.reduce_sum(q_emb * self.E, 1)
        for k in range(self._hops):
            i_emb = tf.nn.embedding_lookup(self.A, stories)
            o_emb = tf.nn.embedding_lookup(self.C, stories)
            probs = self.input_module(i_emb, u_k)
            o_k = self.output_module(probs, o_emb)
            u_k = o_k + tf.matmul(u_k, self.H)

        out = tf.matmul(u_k, self.W)
        #out = tf.Print(out, [tf.slice(t, [0, 0], [1, -1]) for t in [self.A, self.B, self.C]], message="before nil embeddings")
        return out
        
    def input_module(self, i_emb, u):
        # u -> (batch_size, embedding_size)
        # i_mem -> (batch_size, memory_size, embedding_size)
        # probs -> (batch_size, memory_size)
        # uu -> (batch_size, 1, embedding_size)
        with tf.name_scope("input_module"):
            i_mem = tf.reduce_sum(i_emb * self.E, 2)
            u_temp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
            dotted = tf.reduce_sum(i_mem * u_temp, 2)
            return tf.nn.softmax(dotted)

    def output_module(self, probs, o_emb):
        # probs -> (batch_size, memory_size)
        # o_mem -> (batch_size, memory_size, embedding_size)
        # o -> (batch_size, embedding_size)
        with tf.name_scope("output_module"):
            o_mem = tf.reduce_sum(o_emb * self.E, 2)
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            o_mem_temp = tf.transpose(o_mem, [0, 2, 1])
            return tf.reduce_sum(o_mem_temp * probs_temp, 2)

            
