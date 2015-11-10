from __future__ import absolute_import
from __future__ import print_function

from process_data import load_challenge
from autograd.util import quick_grad_check
from autograd import grad
from models.memn2n import MemN2N

from itertools import chain

# challenge data
dir_1k = "data/tasks_1-20_v1-2/en/"
dir_10k = "data/tasks_1-20_v1-2/en-10k/"

train, test = load_challenge(dir_1k, 1)

# represent story?
# 3D array
# (story, sentence, word)
#
# so we need to figure out the max number of sentences in any story
# and the max number of words in any sentence

# size of memory
# size of vocab
# batch size
# size of total words
#
# if time is enabled: vocab_size = vocab_size + size of memory
# total words = total words + 1, the extra 1 is for time words

print(train[0][0])
print(train[0][1])
print(train[0][2])

# TODO: vocab, padding, and turns words to id
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
vocab_size = len(word_idx)

print(word_idx)

# sentence_maxlen = 0
# story_maxlen = 0
sentence_maxlen = max(map(len, chain.from_iterable(x for x, _, _ in train + test)))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
print("max sentence length: {}, max story length: {}".format(sentence_maxlen, story_maxlen))

model = MemN2N(vocab_size, sentence_maxlen)
