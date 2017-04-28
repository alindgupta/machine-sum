#!/usr/bin/python3

import os
import pickle
from functools import lru_cache
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_utils import DataUtils
from skipgrams import exec_ngrams

class WordEmbeddings:
    def __init__(self, args):
        self.args = args
        with DataUtils(self.filename) as duf:
            self.text = duf.process()

    @classmethod
    def loop_func(cls):
        # XXX: skipgrams implementation
        raise NotImplementedError

    @property
    @lru_cache(maxsize=None)
    def embeddings(self):
        if not self._embeddings:
            print("Please call the embed method first")
        else:
            return self._embeddings

    def embed(self, loop_function=loop_func):

        def summarizer():
            with tf.name_scope('summary'):
                summary = tf.summary.FileWriter(self.logdir)

        with tf.Graph().as_default():
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(tf.int32,
                        shape=[self.batch_size],
                        name='inputs')
                labels = tf.placeholder(tf.int32,
                        shape=[self.batch_size, 1],
                        name='labels')

            with tf.device('/cpu:0'):
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable('embeddings',
                            shape=[self.input_size, self.embedding_size],
                            initializer=tf.random_normal_initializer())
                    _embed = tf.nn.embedding_lookup(embeddings, inputs)

            with tf.name_scope('noise-contr-estimation'):
                nce_w = tf.get_variable('nce_w',
                        [self.input_size, self.embedding_size],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(1.0 / np.sqrt(self.input_size)))
                nce_b = tf.get_variable('nce_b',
                        [self.input_size],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))
                nce_loss = tf.nn.nce_loss(nce_w,
                        nce_b,                      # nce
                        labels,
                        inputs,
                        num_sampled=10,     # probably negative sampling
                        num_classes=self.input_size,
                        )
                loss = tf.reduce_mean(nce_loss)
                tf.summary.scalar('nce-loss', loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            with tf.Session() as sess:
                num_epochs = 0
                sess.run(tf.global_variable_initializer())
                average_loss = 0.0
                while num_epochs < self.num_epochs:
                    gen_batch = loop_func(self.data)

                    try:
                        x, y = next(gen_batch)
                    except StopIteration:
                        gen_batch = loop_func(self.data)
                        epochs += 1

                    feed_dict = {inputs: x, labels: y}
                    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val
                self._embeddings = embeddings

        def plot_embeddings(self):
            tsne = TSNE(n_components=2, init='pca')
            squashed_embeddings = tsne.fit_transform(self._embeddings[ :500, :])
            labels = (self.reverse_vocab[i] for i in range(500))
            plt.figure()
            for i, label in enumerate(self.dictionary):
                x, y = squashed_embeddings[i, :]
                plt.scatter(x, y)
                plt.annotate(labels, xy=[x, y], textcoords='offset points')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f',
            default=None,
            help='Text file or directory')
    parser.add_argument('--embedding_size',
            default=256,
            help='Size of the embedding layer')
    parser.add_argument('--learning_rate',
            default=1e-1,
            help='Learning rate')
    parser.add_argument('--num_epochs',
            default=10,
            help='Number of epochs for training')
    parser.add_argument('--logdir',
            default=None,
            help='Directory to log tensorflow summaries')
    parser.add_argument()
    args = parser.parse_args()


if __name__ == '__main__':
    main()

