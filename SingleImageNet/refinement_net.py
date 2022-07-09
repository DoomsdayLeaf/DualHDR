import tensorflow as tf
from Block.differential_net import Differential_net
from Block.rfb import RFB
from Block.non_local import Non_Local
from ops import guidedfilter

class Refinement_net(object):
    def __init__(self, is_train=True, batch_size=8, scope = 'dual_ref', model_name="AEncoder",
                 use_diff=False,
                 use_rfb = False,
                 use_non_local=False,
                 use_bd=False):
        self.is_train = is_train
        self.model_name = model_name   # AEncoder, res_net
        self.scope = scope
        self.use_diff = use_diff
        self.use_rfb = use_rfb
        self.use_non_local = use_non_local
        self.use_bd = use_bd
        self.batch_size = batch_size

    def inference(self, input_images):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images)

    def down(self, x, in_ch, outChannels, filterSize, name=""):
        x = tf.layers.average_pooling2d(x, 2, 2)

        # RFB 0
        if self.use_rfb:
            rbf_model = RFB(self.is_train, scope="RFB" + name)
            x = rbf_model.model(x, in_ch, outChannels)
        else:
            x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
            x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)

        return x

    def up(self, x, in_ch, outChannels, skpCn, name=""):
        x = tf.image.resize_bilinear(x, 2*tf.shape(x)[1:3])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x

    def res_block(self, x, c, name):
        with tf.variable_scope(name):
            n = tf.layers.conv2d(tf.nn.leaky_relu(x), c, 3, 1, 'same', name='conv/0')
            n = tf.layers.conv2d(tf.nn.leaky_relu(n), c, 3, 1, 'same', name='conv/1')
            n = x + n
        return n

    def _build_model(self, input_images):
        if self.model_name == 'res_net':
            ch = 64
            n = tf.layers.conv2d(input_images, ch, 3, 1, 'same', name='ref0/conv0')

            for i in range(3):
                n = self.res_block(n, ch, 'ref'+ str(i))

            n = tf.layers.conv2d(tf.nn.leaky_relu(n), 3, 3, 1, 'same', name='refn/conv0')
            return n
        else:
            x = tf.nn.leaky_relu(tf.layers.conv2d(input_images, 16, 7, 1, 'same'), 0.1)
            s1 = tf.nn.leaky_relu(tf.layers.conv2d(x, 16, 7, 1, 'same'), 0.1)

            with tf.variable_scope("down"):
                s2 = self.down(s1, 16, 32, 5, "0")
                s3 = self.down(s2, 32, 64, 3, "1")
                s4 = self.down(s3, 64, 128, 3, "2")
                x = self.down(s4, 128, 128, 3, "3")

            # non_local 0
            if self.use_non_local:
                non_local_model = Non_Local(is_training=self.is_train)
                x = non_local_model.model(x, 128)

            # diff 1.0
            if self.use_diff:
                diff_net = Differential_net(is_train=self.is_train, batch_size=self.batch_size)
                x = diff_net.inference(x, 128)

            with tf.variable_scope("up"):
                x = self.up(x, 128, 128, s4, "0")
                x = self.up(x, 128, 64, s3, "1")
                x = self.up(x, 64, 32, s2, "2")
                x = self.up(x, 32, 16, s1, "3")
            x = tf.layers.conv2d(x, 3, 3, 1, 'same')

            output = input_images[..., 0:3] + x
            return output