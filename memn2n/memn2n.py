"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    """ 
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (1 - (j /ls)) - (i / le)*(1 - ((2 * j) / ls))
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        hops=1,
        max_gradient_norm=40.0,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        encoding=position_encoding,
        session=tf.Session(),
        name='MemN2N'):

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_gradient_norm = max_gradient_norm
        self._init = initializer
        self._opt = optimizer
        self._name = name

        self._build_inputs()
        self._build_vars()
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        tf.add_to_collection('losses', cross_entropy_sum)
        loss_op = tf.add_n(tf.get_collection('losses'), name='loss_op')
        #loss_op = tf.no_op(cross_entropy_sum, name="loss_op")

        # training op
        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(add_gradient_noise(g, 0.001), v) for g,v in grads_and_vars]
        grads_and_vars = [(tf.clip_by_norm(g, self._max_gradient_norm), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name == self.A.name or v.name == self.B.name or v.name == self.C.name:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        
    def _build_vars(self):
        with tf.variable_scope(self._name):
            with tf.variable_scope("Embeddings"):
                nil_word_slot = tf.zeros([1, self._embedding_size])
                A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                self.A = tf.Variable(A, name="A")
                self.B = tf.Variable(B, name="B")
                self.C = tf.Variable(C, name="C")
            with tf.variable_scope("Weights"):
                self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
                self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

            tf.add_to_collection('losses', tf.nn.l2_loss(self.A, name="L2_loss_A"))
            tf.add_to_collection('losses', tf.nn.l2_loss(self.B, name="L2_loss_B"))
            tf.add_to_collection('losses', tf.nn.l2_loss(self.C, name="L2_loss_C"))
            tf.add_to_collection('losses', tf.nn.l2_loss(self.W, name="L2_loss_W"))
            tf.add_to_collection('losses', tf.nn.l2_loss(self.H, name="L2_loss_H"))

    def _inference(self, stories, queries):
        with tf.name_scope("inference"):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_k = tf.reduce_sum(q_emb * self._encoding, 1)
            for _ in range(self._hops):
                i_emb = tf.nn.embedding_lookup(self.A, stories)
                o_emb = tf.nn.embedding_lookup(self.C, stories)
                # Memories
                m = tf.reduce_sum(i_emb * self._encoding, 2, name="m")
                c = tf.reduce_sum(o_emb * self._encoding, 2, name="c")
                probs = self._input_module(m, u_k, name="probs")
                o_k = self._output_module(c, probs, name="o_k")
                u_k = tf.matmul(u_k, self.H) + o_k
                u_k = tf.nn.relu(u_k)
            return tf.matmul(u_k, self.W, name="logits")
        
    def _input_module(self, m, u, name=None):
        with tf.name_scope("input_module"):
            # Currently tensorflow does not support reduce_dot, so this
            # is a little hack to get around that.
            u_temp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])
            dotted = tf.reduce_sum(m * u_temp, 2)
            # Because we pad empty memories to conform to a memory_size
            # we add a large enough negative value such that the softmax
            # value of the empty memory is 0.
            # Otherwise, empty memories, depending on the memory_size will
            # have a larger and larger impact.
            bs = tf.shape(dotted)[0]
            tt = tf.fill(tf.pack([bs, self._memory_size]), -1000.0)
            cond = tf.not_equal(dotted, 0.0)
            # Returns softmax probabilities, acts as an attention mechanism
            # to signal the importance of memories.
            return tf.nn.softmax(tf.select(cond, dotted, tt), name=name)

    def _output_module(self, c, probs, name=None):
        with tf.name_scope("output_module"):
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            c_temp = tf.transpose(c, [0, 2, 1])
            return tf.reduce_sum(c_temp * probs_temp, 2, name=name)

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Inputs
        ------

        stories: Tensor (None, memory_size, sentence_size)
        queries: Tensor (None, sentence_size)
        answers: Tensor (None, vocab_size)

        Returns
        -------

        loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss
        
    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Inputs
        ------

        stories: Tensor (None, memory_size, sentence_size)
        queries: Tensor (None, sentence_size)

        Returns
        -------

        answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Inputs
        ------

        stories: Tensor (None, memory_size, sentence_size)
        queries: Tensor (None, sentence_size)

        Returns
        -------

        answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Inputs
        ------

        stories: Tensor (None, memory_size, sentence_size)
        queries: Tensor (None, sentence_size)

        Returns
        -------

        answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)
