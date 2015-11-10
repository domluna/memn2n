from __future__ import absolute_import
from __future__ import division

import autograd.numpy as np
import autograd.numpy.random as nr
import tensorflow as tf
from six.moves import range

def iid_gaussian(mean=0., std=1.):
    return lambda s: nr.randn(*s) * std + mean

def iid_uniform(low, high):
    return lambda s: nr.rand(*s) * (high - low) + low

def softmax(x, axis=1, keepdims=True):
    out = np.exp(x)
    return out / np.sum(out, axis=axis, keepdims=keepdims)

def logsoftmax(x, axis=1, keepdims=True):
    return np.log(softmax(x, axis, keepdims))

class MemN2N(object):
    def __init__(self, vocab_size, sentence_maxlen,
            embed_size=50,
            mem_size=50,
            hops=3,
            enable_time=True,
            # use_bow=False,
            randomize_time = 0.1,
            share_type='adjacent',
            init=iid_gaussian(mean=0., std=0.1)):

        self.vocab_size = vocab_size
        self.mem_size = mem_size
        self.embed_size = embed_size
        self.hops = hops

        self.params = {}
        self.params['A'] = init((embed_size, vocab_size))
        self.params['B'] = init((embed_size, vocab_size))
        self.params['C'] = init((embed_size, embed_size))
        self.params['W'] = init((vocab_size, embed_size))

        # memory
        self.input_mem = np.zeros((mem_size, embed_size))
        self.output_mem = np.zeros((mem_size, embed_size))

        # temporal encodings
        self.params['TA'] = init((mem_size, embed_size))
        self.params['TC'] = init((mem_size, embed_size))

        # position encoding
        self.PE = np.zeros((embed_size, sentence_maxlen))
        J = sentence_maxlen+1
        d = embed_size+1
        for k in range(1, d):
            for j in range(1, J):
                self.PE[k-1, j-1] = 1 - (j/J) - (k/d)*(1 - 2*j/J)


    # story -> 3d array
    # question -> 2d array
    def fprop(story, question):
        """
            story and question are lists of the ids words
            corresponding to the vocabulary.

            len(story) and the story themself should be
            padded to fixed length by a null symbol prior calling
            fprop.

            Ex. len(story) == mem_size

            length of the story themselves should be the length of the longest
            sentence in the story.

            m_i = sum_j (l_j * Ax_ij)

            Ax_ij is the the embedding of the jth word in the ith sentence. It
            has dimension embed_size.

            We lookup the value of x_ij in A and multiply it by
            l_j

            l_j is a column vector with the structure:

            l_kj = (1 - j/J) - (k/d)(1 - 2j/J)     # 1-based indexing

            iterate over k

            For 0-based indexing:

                j = j + 1           ????
                d = embed_size
                k/d = (0, 1]

            J is the number of words in the sentence, d is the dimension
            of the embedding (embed_size).

            In addition we encode temporal information through special matrices
            TA and TOM.

            m_i = sum_j (Ax_ij) + TA(i)

            story should be indexed in reverse, reflecting their
            relative distance from the question so that story[0] is
            the last sentence of the story.

        """

        A = self.params['A']
        B = self.params['B']
        C = self.params['C']
        W = self.params['W']
        TA = self.params['TA']
        TC = self.params['TC']
        IM = self.input_mem
        OM = self.output_mem

        Bx = B[:, question]

        # TODO: figure out reverse-order temporal thing
        for i, s in enumerate(story):
            # input memory representation
            A_i = A[:, s] # (embed_size x sentence_maxlen)
            pe = np.sum(A_i * self.PE, axis=1) # position encoding
            te = np.sum(A_i, axis=1) + TA[i] # temporal encoding
            IM[1:, :] = IM[-1, :]
            IM[0, :] = te + pe

            # output memory representation
            C_i = C[:, s] # (embed_size x sentence_maxlen)
            pe = np.sum(C_i * self.PE, axis=1) # position encoding
            te = np.sum(C_i, axis=1) + TC[i] # temporal encoding
            OM[1:, :] = OM[-1, :]
            OM[0, :] = te + pe

        # process question
        u = np.sum(B[:, question] * self.PE, axis=1)

        probs = softmax(np.dot(IM, u), axis=0) # (mem_size x 1)

        # output
        o = np.sum(probs.reshape(-1,1) * OM, axis=0)

        # TODO: generalize for k hops
        a_hat = softmax(np.dot(W, o + u), axis=0) # (vocab_size x 1)
        return a_hat


    # TODO: multiple words
    # currently get mean log loss of answer?
    def loss_func(self, story, question, answer):
        answers_pred = fprop(story, question)
        return -np.mean(np.log(answers_pred[answer]))

    def pred_func(self, story, question):
        answers_pred = fprop(story, question)
        pass

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
