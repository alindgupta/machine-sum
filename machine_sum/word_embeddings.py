import os
import string
import pickle
import functools
from collections import Counter
import collections
from typing import Generator
import random

# external dependancies
import numpy as np
import nltk
import tensorflow as tf


# inline with cython
def is_numeric(x: str) -> bool:
    """ Check if a string is coercible to float """
    try:
        float(x)
        return True
    except ValueError:
        return False

data_index = 0


def batcher(data, batch_size, num_skips, skip_window):
    """ Generate batches """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


class Embedder:

    # tokens to filter from corpus
    # include stopwords and punctuation
    invalid_tokens = set((nltk.corpus.stopwords.words('english')) +
                         list(string.punctuation))

    def __init__(self, filename: str):

        # assert the file exists
        assert os.path.isfile(filename)

        self.filename = filename        # text file with corpus
        self.input_size = 0             # number of tokens to embed
        self.embedding_size = 128       # size of embedding vector
        self.num_epochs = 10            # number of epochs to train
        self.batch_size = 50           # batch size
        self.learning_rate = 1e-2       # learning rate
        self._data = []
        self._vocabulary = {}
        self.corpus = []

    @functools.lru_cache(maxsize=1)
    def set_attributes(self, max_fraction=1000) -> None:
        """
        Generate data 
        
        :param max_fraction - fraction of corpus size to embed
        for example, for corpus size of length L,
        max_fraction of 1 will embed the entire corpus
        max_fraction of 10 will embed the L/10th  most frequent words
        max_fraction of 100 will embed the L/100th  most frequent words
        Recommended: 1000-10000
        
        """
        try:
            with open(self.filename) as file_handle:  # -> List[str]
                for line in file_handle:    # bottleneck loop, use cython
                    self._data += [token for token in nltk.word_tokenize(line)
                                   if token not in Embedder.invalid_tokens
                                   and not is_numeric(token)]
        except IOError as err:
            print('Unable to read file ', self.filename)
            print(err)

        max_input_count = len(self._data) // max_fraction    # -> int
        self.input_size = max_input_count

        counter = iter(Counter(set(self._data)).most_common(max_input_count - 1))
        self._vocabulary = {token: token_id for token_id, token in
                            enumerate([item for item, _ in counter], 1)}
        self._vocabulary['UNK'] = 0

        self.corpus = [self._vocabulary[i] if i in self._vocabulary
                       else self._vocabulary['UNK'] for i in self._data]

    def dump(self):
        """ Pickle all fields """
        with open(r'dumpObject.pickle', 'w') as pk_handle:
            pickle.dump(self.__dict__, pk_handle)

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def data(self):
        return self._data

    @property
    def embeddings(self):
        if not self._embeddings:
            print('Embeddings not set, perhaps you have not run embed()')
        return self._embeddings

    @functools.lru_cache(maxsize=1)
    def embed(self):
        # with tf.name_scope('summary'):
        #    summary = tf.summary.FileWriter(self.logdir)

        with tf.Graph().as_default():

            # placeholders for batching
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(
                            tf.int32,
                            [self.batch_size],
                            name='inputs')
                labels = tf.placeholder(
                        tf.int32,
                        [self.batch_size, 1],
                        name='labels')

            # specify cpu instructions
            with tf.device('/cpu:0'):
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable(
                                    'embeddings',
                                    [self.input_size, self.embedding_size],
                                    initializer=tf.random_normal_initializer(
                                        mean=0.0,
                                        stddev=1.0))
                    _embed = tf.nn.embedding_lookup(embeddings, inputs)

                with tf.name_scope('nc-weights'):
                    nce_w = tf.get_variable(
                            'nce-w',
                            [self.input_size, self.embedding_size],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(
                                mean=0.0,
                                stddev=1.0/np.sqrt(self.input_size)))
                    nce_b = tf.get_variable(
                            'nce-b',
                            [self.input_size],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

                with tf.name_scope('loss'):
                    nce_loss = tf.nn.nce_loss(
                            nce_w,
                            nce_b,
                            labels,
                            _embed,
                            num_sampled=5,
                            num_classes=self.input_size
                            )
                    loss = tf.reduce_mean(nce_loss)

                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(
                            self.learning_rate
                            ).minimize(loss)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    average_loss = 0.0

                    epoch = 0
                    for step in range(1000):
                        batch_inputs, batch_outputs = batcher(self.corpus,
                                                              self.batch_size, 2, 4)
                        _, loss_value = sess.run([optimizer, loss],
                                                 feed_dict={inputs: batch_inputs,
                                                            labels: batch_outputs})
                        average_loss += loss_value
                        if step % 100 == 0:
                            print(average_loss/100)
                    return embeddings.eval()
