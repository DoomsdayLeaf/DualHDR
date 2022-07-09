import logging

logging.basicConfig(level=logging.INFO)
import argparse
import tensorflow as tf
from SingleImageNet.util import get_tensor_shape, apply_rf, log10,get_l2_loss
import os
import glob
from random import shuffle
from SingleImageNet import hallucination_net
from Dual_SingleImageNet.clipping_net import Clipping_Net
from SingleImageNet.Dual_Hal.vgg16_net import Vgg16
from SingleImageNet import util

class Dual_Hal_Real(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.hal_batch_size
        self.it_num = args.hal_epoch
        self.hal_indi_log_dir = args.hal_indi_log_dir
        self.logdir_path = args.hal_log_dir
        self.tfrecords_path = args.tf_records_log_dir
        self.itnet_joint_train = args.itnet_joint_train

        self.dual_lamada = 0

        self.tfrecord_list = glob.glob(os.path.join(self.tfrecords_path, '*.tfrecords'), recursive=True)
        print(len(self.tfrecord_list))
        assert (self.tfrecord_list)
        shuffle(self.tfrecord_list)
        print('\n\n====================\ntfrecords list:')
        [print(f) for f in self.tfrecord_list]
        print('====================\n\n')

        with tf.device('/cpu:0'):
            self.filename_queue = tf.train.string_input_producer(self.tfrecord_list)
            self.ref_HDR_batch, self.ref_LDR_batch = self.load_real_data(self.filename_queue)

        self._clip = lambda x: tf.clip_by_value(x, 0, 1)

    def load_real_data(self, filename_queue):

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        img_features = tf.parse_single_example(
            serialized_example,
            features={
                'ref_HDR': tf.FixedLenFeature([], tf.string),
                'ref_LDR': tf.FixedLenFeature([], tf.string),
            })

        ref_HDR = tf.decode_raw(img_features['ref_HDR'], tf.float32)
        ref_LDR = tf.decode_raw(img_features['ref_LDR'], tf.float32)
        ref_HDR = tf.reshape(ref_HDR, [256, 256, 3])
        ref_LDR = tf.reshape(ref_LDR, [256, 256, 3])

        ref_HDR = ref_HDR / (1e-6 + tf.reduce_mean(ref_HDR)) * 0.5
        ref_LDR = ref_LDR / 255.0

        distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)

        # flip horizontally
        ref_HDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_HDR), lambda: ref_HDR)
        ref_LDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_LDR), lambda: ref_LDR)

        # rotate
        k = tf.cast(distortions[1] * 4 + 0.5, tf.int32)
        ref_HDR = tf.image.rot90(ref_HDR, k)
        ref_LDR = tf.image.rot90(ref_LDR, k)

        # TODO: channel swapping?

        ref_HDR_batch, ref_LDR_batch = tf.train.shuffle_batch(
            [ref_HDR, ref_LDR],
            batch_size=8,
            num_threads=8,
            capacity=256,
            min_after_dequeue=64)

        return ref_HDR_batch, ref_LDR_batch

    def fix_quantize(
            self, img,  # [b, h, w, c]
            is_training,
    ):
        b, h, w, c, = get_tensor_shape(img)

        const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

        bit = const_bit
        s = (2 ** bit) - 1

        img = self._clip(img)
        img = tf.round(s * img) / s
        img = self._clip(img)

        return img

    def build_graph(
            self,
            ldr,  # [b, h, w, c]
            hdr,  # [b, h, w, c]
            is_training,
    ):
        b, h, w, c, = get_tensor_shape(ldr)
        b, h, w, c, = get_tensor_shape(hdr)

        """ compute standard model """
        # Dynamic range clipping
        self.clipped_hdr = self._clip(hdr)

        gray = tf.image.rgb_to_grayscale(tf.round(ldr * 255.0))
        over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
        over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
        over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
        under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
        under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
        under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
        extreme_cases = tf.logical_or(over_exposed, under_exposed)
        self.loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

        """ build primary model """
        # Highlight mask
        thr = 0.12
        self.alpha = tf.reduce_max(self.clipped_hdr, reduction_indices=[3])
        self.alpha = tf.minimum(1.0, tf.maximum(0.0, self.alpha - 1.0 + thr) / thr)
        self.alpha = tf.reshape(self.alpha, [-1, tf.shape(self.clipped_hdr)[1], tf.shape(self.clipped_hdr)[2], 1])
        self.alpha = tf.tile(self.alpha, [1, 1, 1, 3])

        with tf.variable_scope("Hallucination_Net"):
            self.net, self.vgg16_conv_layers = hallucination_net.model(self.clipped_hdr, self.batch_size, True)
            self.y_predict = tf.nn.relu(self.net.outputs)
            self.y_res = self.alpha * self.y_predict
            self.y_final = (self.clipped_hdr) + self.y_res # residual

        """ build dual task """
        with tf.variable_scope("Clipping_Net"):
            self.cl_net = Clipping_Net()
            self.dual_pred_clipped_hdr, self.dual_pred_base, self.dual_pred_detail = self.cl_net.model(self.y_final)

        _log = lambda x: tf.log(x + 1.0 / 255.0)

        """ compute dual loss """
        self.dual_final_loss = util.get_l2_loss(_log(self.dual_pred_clipped_hdr), _log(self.clipped_hdr))

        """ compute primary loss """
        vgg = Vgg16('SingleImageNet/vgg16.npy')
        vgg.build(tf.log(1.0+10.0*self.y_final)/tf.log(1.0+10.0))
        vgg2 = Vgg16('SingleImageNet/vgg16.npy')
        vgg2.build(tf.log(1.0+10.0*hdr)/tf.log(1.0+10.0))
        self.perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
        self.perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
        self.perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)

        y_final_gamma = tf.log(1.0+10.0*self.y_final)/tf.log(1.0+10.0)
        _hdr_t_gamma = tf.log(1.0+10.0*hdr)/tf.log(1.0+10.0)

        self.loss = tf.reduce_mean(tf.abs(y_final_gamma - _hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
        y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
        y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
        tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
        tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
        self.tv_loss = tv_loss_x + tv_loss_y
        self.final_loss = tf.reduce_mean((self.loss + 0.001 * self.perceptual_loss + 0.1 * self.tv_loss)*self.alpha)

        if self.itnet_joint_train == "joint_pair":
            self.final_loss = self.final_loss + self.dual_final_loss * self.dual_lamada
        else:
            self.final_loss = self.dual_final_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.final_loss)

        t_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
        print('all layers:')
        for var in t_vars: print(var.name)

        self.show_loss = tf.reduce_mean(self.loss)
        self.show_final_loss = tf.reduce_mean(self.final_loss)
        self.show_final_dual_loss = tf.reduce_mean(self.dual_final_loss)
        tf.summary.scalar('loss', self.show_loss)
        tf.summary.scalar('final loss', self.show_final_loss)
        tf.summary.scalar('dual_final_loss', self.show_final_dual_loss)
        tf.summary.image('hdr', hdr)
        tf.summary.image('ldr', ldr)
        tf.summary.image('loss_mask', self.loss_mask)
        tf.summary.image('y_final', self.y_final)
        tf.summary.image('clipped_hdr', self.clipped_hdr)
        tf.summary.image('dual_pred_clipped_hdr', self.dual_pred_clipped_hdr)
        tf.summary.image('dual_pred_base', self.dual_pred_base)
        tf.summary.image('dual_pred_detail', self.dual_pred_detail)



    def train(self):
        b, h, w, c = self.batch_size, 512, 512, 3

        is_training = tf.placeholder(tf.bool)


        self.build_graph(self.ref_LDR_batch, self.ref_HDR_batch, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord=coord)
        self.sess.run(tf.global_variables_initializer())

        restorer3 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
        restorer3.restore(self.sess, self.hal_indi_log_dir)

        # hallucination_net.load_vgg_weights(vgg16_conv_layers, 'vgg16_places365_weights.npy', sess)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir_path, 'summary'),
            self.sess.graph,
        )


        for it in range(self.it_num):
            print(it)
            if it == 0 or it % 10000 == 9999:
                print('start save')
                checkpoint_path = os.path.join(self.logdir_path, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=it)
                # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
                print('finish save')
            _, summary_val = self.sess.run([self.train_op, summary], {is_training: True})
            if it == 0 or it % 10000 == 9999:
                summary_writer.add_summary(summary_val, it)
                logging.info('test')


        coord.request_stop()
        coord.join(threads)

