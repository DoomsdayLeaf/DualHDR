import tensorflow as tf

class Differential_net(object):
    def __init__(self, is_train=True, batch_size=4, scope = 'differential_net'):
        self.is_train = is_train
        self.scope = scope
        self.batch_size = batch_size

    def inference(self, input_images, in_ch):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images, in_ch)

    def cu_block(self, x, in_ch, model="single-scale"):
        with tf.variable_scope("cu_block" + model):
            # scale conv
            if model.find("mutil-scale") != -1:
                n_3x3 = tf.nn.relu(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(1,1), padding='same')(x))
                n_5x5 = tf.nn.relu(tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=(1,1), padding='same')(x))
                n_7x7 = tf.nn.relu(tf.keras.layers.DepthwiseConv2D(kernel_size=7, strides=(1,1), padding='same')(x))
                n = n_3x3 + n_5x5 + n_7x7
            else:
                n = tf.nn.relu(tf.nn.depthwise_conv2d(x, filter=[7, 7, 1, 1], strides=[1, 1, 1, 1], padding='SAME'))

            # se unit
            reduction = 2
            n_se = tf.keras.layers.GlobalAvgPool2D()(n)
            n_se = tf.reshape(n_se, [self.batch_size, in_ch])
            n_se = tf.layers.dense(n_se, in_ch // reduction, use_bias=False, activation=tf.nn.relu)
            n_se = tf.layers.dense(n_se, in_ch, use_bias=False, activation=tf.nn.sigmoid)
            n_se = tf.reshape(n_se, [self.batch_size, 1, 1, in_ch])
            n = tf.multiply(n, n_se)

            n =  tf.nn.relu(tf.layers.conv2d(n, in_ch, 1, 1, 'same'))
            out = tf.add(x, n)

            return out

    def _build_model(self, x, in_ch):
        with tf.variable_scope(self.scope):
            n1 = tf.nn.relu(tf.layers.conv2d(x, in_ch, 5, 1, 'same'))
            n = tf.nn.relu(tf.layers.conv2d(n1, in_ch * 2, 5, 2, 'same'))
            n = tf.nn.relu(tf.layers.conv2d(n, in_ch * 2, 1, 1, 'same'))

            for i in range(3):
                n = self.cu_block(n, in_ch * 2, "mutil-scale" + str(i))
            n = tf.depth_to_space(n, 2, name='pixel_shuffle')
            n = tf.concat([n1, n], -1)
            n = tf.nn.relu(tf.layers.conv2d(n, in_ch, 1, 1, 'same'))
            return n