""" Implementation of word2vec using skipgrams in tensorflow """

import os
import argparse
from typing import List, Generator

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class WordEmbeddings:
    """ Embed words from a corpus into vector """

    def __init__(self, filename, logdir, input_size):

        assert os.path.isfile(args.filename), 'Could not find file ' \
                '{}'.format(args.filename)

        self.filename = filename
        self.logdir = logdir
        self.embedding_size = 128
        self.input_size = input_size
        self.num_epochs = 10
        self.verbose = True
        self.batch_size = 256
        self.learning_rate = 1e-1

        # run the data utils function to retrive the following
        self.data = None
        self.vocabulary = None
        self.reverse_vocabulary = None

        # embeddings output
        self._embeddings = None

    @property
    def get_embeddings(self):
        """ """
        if self._embeddings:
            return self._embeddings
        else:
            print('Please run the embed method first')
            return None

    @classmethod
    def loop_func(cls) -> Generator:
        """ """
        pass

    def embed(self):
        """ """

        # initialize tensorboard summarizer
        def summarizer():
            with tf.name_scope('summary'):
                summary = tf.summary.FileWriter(self.logdir)

        
        # set up graph
        with tf.Graph('graph').as_default():
            
            # placeholders for input and labels
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(
                        tf.int32,
                        [self.batch_size],
                        name='inputs')
                labels = tf.placeholder(
                        tf.int32,
                        [self.batch_size],
                        name='labels')

            with tf.device('/cpu:0'):

                # initialize random word embeddings
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable(
                            'embeddings',
                            [self.input_size, self.embedding_size),
                            initializer=tf.random_normal_initializer(
                                mean=0.0,
                                stddev=1.0
                                )]
                    _embed = tf.nn.embedding_lookup(embeddings, inputs)

                with tf.name_scope('nce-weights'):
                    nce_w = tf.get_variable(
                            'nce_w',
                            [self.input_size, self.embedding_size],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(
                                mean=0.0,
                                stddev=1.0 / np.sqrt(self.input_size)
                                ))
                    nce_b = tf.get_variable('nce_b',
                            [self.input_size],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
                
                with tf.name_scope('loss'):
                    nce_loss = tf.nn.nce_loss(
                            nce_w,
                            nce_b,
                            labels,
                            inputs,
                            num_sampled=50,         # neg samples
                            num_classes=self.input_size
                            )
                    loss = tf.reduce_mean(nce_loss)
                    tf.summary_scalar('nce-loss', loss)

                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(self.learning_rate).
                    minimize(loss)

                
                # create a session
                with tf.Session() as sess:
                    num_epochs = 0
                    sess.run(tf.global_variables_initializer())
                    average_loss = 0.0

                    while num_epochs < self.num_epochs:

                        # refactor to a method
                        gen_batch = loop_func()
                        try:
                            x, y = next(gen_batch)
                        except StopIteration:
                            gen_batch = loop_func()
                            x, y = next(gen_batch)
                            epochs += 1
                            continue

                    batch_dict = {inputs: batch_inputs, outputs: batch_outputs}
                    _, loss_val = sess.run([optimizer, loss], feed_dict=batch_dict)
                    average_loss += loss_val

        self._embeddings = embeddings
    
    def plot(self, woi=''):
        """ 
        params
        ------
        woi: the word of interest

        This method will plot the 500 closest words to woi
        if provided by user. Else, it will plot the first 500
        words in the embeddings matrix
        
        """
        
        if not self._embeddings:
            raise Exception('No embeddings to plot, maybe you haven\'t run embed')

        if woi: 
            if not woi in self.vocabulary:
                raise ValueError('{} could not be found in corpus'.format(woi))

            woi_embedding = tf.nn.embedding_lookup(self._embeddings,
                    self.vocabulary[woi])
            
            # compute cosine distance
            norm = tf.sqrt(tf.reduce_mean(
                tf.square(self._embeddings),
                axis=1,
                keep_dims=True
                ))
            normalized_embeddings = tf.div(self._embeddings, norm)
            cosdist = tf.matmul(woi_embedding, normalized_embeddings)
            plt_embeddings = np.argsort(cosdist)[:500, :]
            pass

        else:
            embeddings = 


        tsne = TSNE(n_components=2, init='pca')
        transf_embeds = tsne.fit_transform(embeddings)

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--filepath',
            type=str,
            default=None,
            help='Absolute filepath for corpus'
            )
    parser.add_argument(
            '--logdir',
            type=str,
            default=None,
            help='Directory for logging tensorflow data'
            )
    parser.add_argument(
            '--embedding_size',
            type=int,
            default=256,
            help='Size of the embedding vectors'
            )
    parser.add_argument(
            '--num_epochs',
            type=int,
            default=10,
            help='Number of epochs to run'
            )

if __name__ == '__main__':
    main()

