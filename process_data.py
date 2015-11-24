from __future__ import absolute_import

import os
import re
import numpy as np

def load_challenge(data_dir, n, only_supporting=False):
    '''Load the nth challenge. There are currently 20 challenges in total.

    Returns a tuple containing the training and testing data for the challenge.
    '''
    assert n > 0 and n < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'
    train_file = [f for f in files if s.format(n) in f and 'train' in f][0]
    test_file = [f for f in files if s.format(n) in f and 'test' in f][0]

    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)

    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


# TODO: determine what the value of only_supporting is
def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            a = tokenize(a)
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries and pads them to sentence_size.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for sentence in story:
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)
        ss = ss[:memory_size]
        S.append(ss)
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq
        a = answer[0]
        y = np.zeros(len(word_idx) + 1) # 0 is reversed for nil word
        y[word_idx[a]] = 1
        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S, dtype=np.int32), np.array(Q, dtype=np.int32), np.array(A, dtype=np.int32)
