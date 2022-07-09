import os
import tensorflow as tf
import numpy

class Non_Local(object):
    def __init__(self, is_training = True, scope="non_local"):
        self.is_training = is_training
        self.scope = scope

    def model(self, x, depth, embed=True, softmax=False, maxpool=2):
        with tf.variable_scope(self.scope):
            if embed:
                a = tf.layers.conv2d(x, depth, 1, strides=1, padding='same', name='embA')
                b = tf.layers.conv2d(x, depth, 1, strides=1, padding='same', name='embB')
            else:
                a, b = x, x
            g_orig = g = tf.layers.conv2d(x, depth, 1, strides=1, padding='same', name='g')

            if maxpool is not False and maxpool > 1:
                b = tf.layers.max_pooling2d(b, [maxpool, maxpool], strides=maxpool, name='pool')
                g = tf.layers.max_pooling2d(g, [maxpool, maxpool], strides=maxpool, name='pool')

            # Flatten from (B,H,W,C) to (B,HW,C) or similar
            a_flat = tf.reshape(a, [tf.shape(a)[0], -1, tf.shape(a)[-1]])
            b_flat = tf.reshape(b, [tf.shape(b)[0], -1, tf.shape(b)[-1]])
            g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
            a_flat.set_shape([a.shape[0], a.shape[1] * a.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
            b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
            g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
            # Compute f(a, b) -> (B,HW,HW)
            f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
            if softmax:
                f = tf.nn.softmax(f)
            else:
                f = f / tf.cast(tf.shape(f)[-1], tf.float32)
            # Compute f * g ("self-attention") -> (B,HW,C)
            fg = tf.matmul(f, g_flat)
            # Expand and fix the static shapes TF lost track of.
            fg = tf.reshape(fg, tf.shape(g_orig))
            # fg.set_shape(g.shape)  # NOTE: This actually appears unnecessary.

            fg = tf.layers.conv2d(fg, x.shape[-1], 1, strides=1, name='fgup')
            n = x + fg

            return n

