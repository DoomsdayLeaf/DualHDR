import os
import tensorflow as tf
import numpy

class RFB(object):
    def __init__(self, is_training=True, scope="RBF"):
        self.is_training = is_training
        self.scope = scope

    def conv_bn_relu(self, x, out_ch, kernel_size, stride=1, dilation=1, bias=False, use_bn=True, use_relu=True):
        n = tf.layers.conv2d(x, out_ch, kernel_size, strides=stride, padding='same', dilation_rate=dilation, use_bias=bias)
        if use_bn:
            n = tf.layers.batch_normalization(n, epsilon=1e-5, momentum=0.01, trainable=True)
        if use_relu:
            n = tf.nn.relu(n)
        return n

    def model(self, x, in_ch, out_ch, stride=1, scale=0.1, visual=1):
        with tf.variable_scope(self.scope):
            inter_ch = in_ch // 8
            n0 = self.conv_bn_relu(x, 2 * inter_ch, kernel_size=1, stride=stride, dilation=visual)
            n0 = self.conv_bn_relu(n0, 2 * inter_ch, 3, stride, use_relu=False)

            n1 = self.conv_bn_relu(x, inter_ch, kernel_size=1, stride=1)
            n1 = self.conv_bn_relu(n1, 2 * inter_ch, kernel_size=3, stride=stride)
            n1 = self.conv_bn_relu(n1, 2 * inter_ch, kernel_size=3, stride=1, dilation=visual+1, use_relu=False)

            n2 = self.conv_bn_relu(x, inter_ch, kernel_size=1, stride=1)
            n2 = self.conv_bn_relu(n2, (inter_ch // 2) * 3, kernel_size=3, stride=1)
            n2 = self.conv_bn_relu(n2, 2 * inter_ch, kernel_size=3, stride=stride)
            n2 = self.conv_bn_relu(n2, 2 * inter_ch, kernel_size=3, stride=1, dilation=2*visual + 1, use_relu=False)

            n = tf.concat([n0, n1, n2], -1)
            n = self.conv_bn_relu(n, out_ch, kernel_size=1, stride=1, use_relu=False)
            short = self.conv_bn_relu(x, out_ch, kernel_size=1, stride=1, use_relu=False)
            out = tf.nn.relu(n * scale + short)

            return out

