import logging
import inspect
import time

logging.basicConfig(level=logging.INFO)
import os
import tensorflow as tf
from SingleImageNet import util, hallucination_net
from DataProcess import dataset
from Dual_SingleImageNet import clipping_net
import numpy as np


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "SingleImageNet/vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.VGG_MEAN = [103.939, 116.779, 123.68]

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

# --- graph
class Dual_Hal_Synth(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        self._clip = lambda x: tf.clip_by_value(x, 0, 1)

        self.batch_size = 8
        self.hdr_prefix = args.hdr_prefix

        self.it_num = 100000
        self.logdir_path = "ckpt/dBD_iter3_hal_synth_100k_d10/"
        self.tfrecords_path = args.tf_records_log_dir
        self.deq_ckpt = args.pre_deq_ckpt
        self.lin_ckpt = args.pre_lin_ckpt
        self.hal_ckpt = args.pre_hal_ckpt
        self.ref_ckpt = args.pre_ref_ckpt

        self.epsilon = 0.001
        self.dual_lamada = 10
        self.use_iter_net = True
        self.use_clipped_net_in = False

    def rand_quantize(
            self,
            img,  # [b, h, w, c]
            is_training):
        b, h, w, c, = util.get_tensor_shape(img)

        const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

        bit = tf.cond(is_training, lambda: const_bit, lambda: const_bit)
        s = (2 ** bit) - 1

        img = self._clip(img)
        img = tf.round(s * img) / s
        img = self._clip(img)

        return img

    def get_final(self, network, x_in):
        sb, sy, sx, sf = x_in.get_shape().as_list()
        y_predict = network.outputs

        # Highlight mask
        thr = 0.05
        alpha = tf.reduce_max(x_in, reduction_indices=[3])
        alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
        alpha = tf.reshape(alpha, [-1, sy, sx, 1])
        alpha = tf.tile(alpha, [1, 1, 1, 3])

        # Linearied input and prediction
        x_lin = tf.pow(x_in, 2.0)
        y_predict = tf.exp(y_predict) - 1.0 / 255.0

        # Alpha blending
        y_final = (1 - alpha) * x_lin + alpha * y_predict

        return y_final


    def build_graph(
            self,
            hdr,  # [b, h, w, c]
            crf,  # [b, k]
            t,  # [b]
            is_training,
    ):
        b, = util.get_tensor_shape(t)

        self._hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

        # Augment Poisson and Gaussian noise
        sigma_s = 0.08 / 6 * tf.random_uniform([tf.shape(self._hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                                         dtype=tf.float32, seed=1)
        sigma_c = 0.005 * tf.random_uniform([tf.shape(self._hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
        noise_s_map = sigma_s * self._hdr_t
        noise_s = tf.random_normal(shape=tf.shape(self._hdr_t), seed=1) * noise_s_map
        temp_x = self._hdr_t + noise_s
        noise_c = sigma_c * tf.random_normal(shape=tf.shape(self._hdr_t), seed=1)
        temp_x = temp_x + noise_c
        self._hdr_t = tf.nn.relu(temp_x)

        # Dynamic range clipping
        if self.use_clipped_net_in:
            self.cl_net_for_primaly = clipping_net.Clipping_Net()
            self.clipped_hdr_t, self.clipped_hdr_t_base, self.clipped_hdr_t_detail = self.cl_net_for_primaly.model(self._hdr_t)
        else:
            self.clipped_hdr_t = self._clip(self._hdr_t)

        # loss mask
        self.ldr = util.apply_rf(self.clipped_hdr_t, crf)
        self.quantized_hdr = tf.round(self.ldr * 255.0)
        self.quantized_hdr_8bit = tf.cast(self.quantized_hdr, tf.uint8)
        jpeg_img_list = []
        for i in range(self.batch_size):
            II = self.quantized_hdr_8bit[i]
            II = tf.image.adjust_jpeg_quality(II, int(round(float(i) / float(self.batch_size - 1) * 10.0 + 90.0)))
            jpeg_img_list.append(II)
        self.jpeg_img = tf.stack(jpeg_img_list, 0)
        self.jpeg_img_float = tf.cast(self.jpeg_img, tf.float32) / 255.0
        self.jpeg_img_float.set_shape([None, 256, 256, 3])
        gray = tf.image.rgb_to_grayscale(self.jpeg_img)
        over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
        over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
        over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
        under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
        under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
        under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
        extreme_cases = tf.logical_or(over_exposed, under_exposed)
        self.loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

        # Highlight mask
        thr = 0.12
        self.alpha = tf.reduce_max(self.clipped_hdr_t, reduction_indices=[3])
        self.alpha = tf.minimum(1.0, tf.maximum(0.0, self.alpha - 1.0 + thr) / thr)
        self.alpha = tf.reshape(self.alpha, [-1, tf.shape(self.clipped_hdr_t)[1], tf.shape(self.clipped_hdr_t)[2], 1])
        self.alpha = tf.tile(self.alpha, [1, 1, 1, 3])

        if self.use_iter_net:
            hal_net_input = self.clipped_hdr_t

            with tf.device("/gpu:0"):
                with tf.variable_scope("Hallucination_Net"):
                    self.net, self.vgg16_conv_layers = hallucination_net.model(hal_net_input, self.batch_size, True)
                    self.y_predict = tf.nn.relu(self.net.outputs)
                    self.y_res = self.alpha * self.y_predict
                    self.y_final = (self.clipped_hdr_t) + self.y_res  # residual

                with tf.variable_scope("Clipping_Net"):
                    self.cl_net = clipping_net.Clipping_Net()
                    self.dual_pred_clipped_hdr, self.dual_pred_base, self.dual_pred_detail = self.cl_net.model(self.y_final)

            for hal_it in range(1, 3):
                gpu_idx = "/gpu:" + str(int(hal_it / 2))
                with tf.device(gpu_idx):
                    with tf.variable_scope("Hallucination_Net", reuse=True):
                        self.net, self.vgg16_conv_layers = hallucination_net.model(self.dual_pred_clipped_hdr, self.batch_size, True)
                        self.y_predict = tf.nn.relu(self.net.outputs)
                        self.y_res = self.alpha * self.y_predict
                        self.y_final = (self.clipped_hdr_t) + self.y_res  # residual

                    """ build dual task """
                    with tf.variable_scope("Clipping_Net", reuse=True):
                        self.cl_net = clipping_net.Clipping_Net()
                        self.dual_pred_clipped_hdr, self.dual_pred_base, self.dual_pred_detail = self.cl_net.model(self.y_final)

            with tf.variable_scope("Hallucination_Net", reuse=True):
                net_test, vgg16_conv_layers_test = hallucination_net.model(self.clipped_hdr_t, self.batch_size, False)
                self.y_predict_test = tf.nn.relu(net_test.outputs)
                self.y_final_test = (self.clipped_hdr_t) + self.alpha * self.y_predict_test # residual

        else:
            """ build primary model """
            with tf.variable_scope("Hallucination_Net"):
                self.net, self.vgg16_conv_layers = hallucination_net.model(self.clipped_hdr_t, self.batch_size, True)
                self.y_predict = tf.nn.relu(self.net.outputs)
                self.y_res = self.alpha * self.y_predict
                self.y_final = (self.clipped_hdr_t) + self.y_res # residual

            with tf.variable_scope("Hallucination_Net", reuse=True):
                net_test, vgg16_conv_layers_test = hallucination_net.model(self.clipped_hdr_t, self.batch_size, False)
                self.y_predict_test = tf.nn.relu(net_test.outputs)
                self.y_final_test = (self.clipped_hdr_t) + self.alpha * self.y_predict_test # residual

            """ build dual task """
            with tf.variable_scope("Clipping_Net"):
                self.cl_net = clipping_net.Clipping_Net()
                self.dual_pred_clipped_hdr, self.dual_pred_base, self.dual_pred_detail = self.cl_net.model(self.y_final)

        _log = lambda x: tf.log(x + 1.0 / 255.0)

        """ compute dual loss """
        self.dual_final_loss = util.get_l2_loss(_log(self.dual_pred_clipped_hdr), _log(self.clipped_hdr_t))

        """ compute primary loss """
        vgg = Vgg16('SingleImageNet/vgg16.npy')
        vgg.build(tf.log(1.0+10.0*self.y_final)/tf.log(1.0+10.0))
        vgg2 = Vgg16('SingleImageNet/vgg16.npy')
        vgg2.build(tf.log(1.0+10.0*self._hdr_t)/tf.log(1.0+10.0))
        self.perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
        self.perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
        self.perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)

        self.loss_test = util.get_l2_loss(_log(self.y_final_test), _log(self._hdr_t))

        y_final_gamma = tf.log(1.0+10.0*self.y_final)/tf.log(1.0+10.0)
        _hdr_t_gamma = tf.log(1.0+10.0*self._hdr_t)/tf.log(1.0+10.0)

        self.loss = tf.reduce_mean(tf.abs(y_final_gamma - _hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
        y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
        y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
        tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
        tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
        self.tv_loss = tv_loss_x + tv_loss_y
        self.final_loss = tf.reduce_mean((self.loss + 0.001 * self.perceptual_loss + 0.1 * self.tv_loss)*self.loss_mask)
        self.final_loss = self.final_loss + self.dual_final_loss * self.dual_lamada

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.final_loss)

        t_vars = tf.trainable_variables()
        print('all layers:')
        for var in t_vars: print(var.name)

        #loss_num = tf.reduce_mean(loss)
        self.show_loss = tf.reduce_mean(self.loss)
        self.show_final_loss = tf.reduce_mean(self.final_loss)
        self.show_final_dual_loss = tf.reduce_mean(self.dual_final_loss)

        tf.summary.scalar('loss', self.show_loss)
        tf.summary.scalar('final_loss', self.show_final_loss)
        tf.summary.scalar('dual_final_loss', self.show_final_dual_loss)
        tf.summary.image('hdr_t', self._hdr_t)
        tf.summary.image('y', self.y_final)
        tf.summary.image('clipped_hdr_t', self.clipped_hdr_t)
        tf.summary.image('alpha', self.alpha)
        tf.summary.image('y_predict', self.y_predict)
        tf.summary.image('log_hdr_t', tf.log(1.0+10.0*self._hdr_t)/tf.log(1.0+10.0))
        tf.summary.image('log_y', tf.log(1.0+10.0*self.y_final)/tf.log(1.0+10.0))
        tf.summary.image('log_clipped_hdr_t', tf.log(1.0+10.0*self.clipped_hdr_t)/tf.log(1.0+10.0))
        tf.summary.image('dual_pred_clipped_hdr', self.dual_pred_clipped_hdr)
        tf.summary.image('dual_pred_base', self.dual_pred_base)
        tf.summary.image('dual_pred_detail', self.dual_pred_detail)

        self.psnr = tf.zeros([])
        self.psnr_no_q = tf.zeros([])


    def train(self):
        b, h, w, c = self.batch_size, 512, 512, 3

        hdr = tf.placeholder(tf.float32, [None, None, None, c])
        crf = tf.placeholder(tf.float32, [None, None])
        t = tf.placeholder(tf.float32, [None])
        is_training = tf.placeholder(tf.bool)

        self.build_graph(hdr, crf, t, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # ---

        self.sess.run(tf.global_variables_initializer())

        restorer3 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
        restorer3.restore(self.sess, self.hal_ckpt)

        #hallucination_net.load_vgg_weights(self.vgg16_conv_layers, 'SingleImageNet/vgg16_places365_weights.npy', self.sess)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir_path, 'summary'),
            self.sess.graph,
        )
        dataset_reader = dataset.RandDatasetReader(dataset.get_train_dataset(self.hdr_prefix), b)

        for it in range(self.it_num):
            print(it)
            if it == 0 or it % 10000 == 9999:
                print('start save')
                checkpoint_path = os.path.join(self.logdir_path, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=it)
                print(self.net.all_params)
                # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
                print('finish save')
            hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
            _, summary_val = self.sess.run([self.train_op, summary], {
                hdr: hdr_val,
                crf: crf_val,
                t: t_val,
                is_training: True,
            })

            if it == 0 or it % 10000 == 9999:
                summary_writer.add_summary(summary_val, it)
                logging.info('test')

