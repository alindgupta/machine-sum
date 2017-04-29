import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

class GRUModel:
    def __init__(self, args):
        self._args = args

    def _add_placeholders(self):
        args = self._args
        self._corpus = tf.placeholder(tf.int32,
                [args.batch_size, args.timesteps],
                name='corpus')
        self._targets = tf.placeholder(tf.int32,
                [args.batch_size, args.timesteps],
                name='targets')

        def _init_gru(self):

            with tf.variable_scope('gru'):
                inputs = tf.unstack(tf.transpose(self._corpus))
                
            for layer in range(self._args.layers):
                with tf.variable_scope('layer_{}'.format(layer)):
                    cell = GRUCell(self._args.state_size,
                            initializer=tf.truncated_normal_initializer(0.0, 1.0),
                            state_is_tuple=False)

                with tf.variable_scope('out'):
                    pass


