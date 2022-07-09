import logging

logging.basicConfig(level=logging.INFO)
import argparse
import tensorflow as tf
from SingleImageNet.util import get_tensor_shape, apply_rf, log10,get_l2_loss
import os
import glob
from random import shuffle
from SingleImageNet.dequantization_net import Dequantization_net
from SingleImageNet.linearization_net import Linearization_net
from Dual_SingleImageNet.non_linearization_net import Non_Linearization_net
from Dual_SingleImageNet.quantization_net import Quantization_net
from Dual_SingleImageNet.clipping_net import Clipping_Net
from SingleImageNet import util

class Joint_Deq_Lin_Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.linear_batch_size
        self.it_num = args.linear_epoch
        self.block_num = 10
        self.logdir_path = args.linear_log_dir
        self.tfrecords_path = args.tf_records_log_dir
        self.deq_ckpt = args.deq_log_dir
        self.linear_indi_log_dir = args.linear_indi_log_dir
        self.non_linear_log_dir = args.non_linear_log_dir
        self.inet_joint_train = args.inet_joint_train

        self.deq_lamada = 1
        self.lin_lamada = 1

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
            batch_size=self.batch_size,
            num_threads=self.batch_size,
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
        #self.clipped_hdr = self._clip(hdr)
        clipped_model = Clipping_Net()
        self.clipped_hdr, self.clipped_hdr_base, self.clipped_hdr_detail = clipped_model.model(hdr)

        """ build primary model """
        # Dequantization-Net
        with tf.variable_scope("Dequantization_Net"):
            dequantization_model = Dequantization_net(is_train=True)
            self.pred_ldr = self._clip(dequantization_model.inference(ldr))

        # Linearization-Net
        lin_net = Linearization_net()
        self.pred_invcrf = lin_net.get_output(self.pred_ldr, True)
        self.pred_clipped_hdr = apply_rf(self.pred_ldr, self.pred_invcrf)

        """ build dual task """
        # Non-Linearization-Net
        non_lin_net = Non_Linearization_net()
        self.pred_crf = non_lin_net.get_output(self.pred_clipped_hdr, True)
        self.dual_pred_ldr = apply_rf(self.pred_clipped_hdr, self.pred_crf)

        # Quantization Net
        quantization_model = Quantization_net(self.batch_size)
        self.dual_pred_jpg = quantization_model.get_output(self.dual_pred_ldr)

        t_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
        print('all layers:')
        for var in t_vars: print(var.name)

        _log = lambda x: tf.log(x + 1.0 / 255.0)
        """ compute loss """
        self.dual_deq_loss = util.get_l2_loss(self.dual_pred_jpg, ldr)
        self.dual_lin_loss = util.get_l2_loss(self.dual_pred_ldr, self.pred_ldr)
        self.lin_loss = util.get_l2_loss(self.pred_clipped_hdr, self.clipped_hdr)
        self.deq_loss = self.dual_lin_loss

        if self.inet_joint_train == "joint_pair":
            self.final_loss = (self.deq_loss + 10 * self.dual_deq_loss) * self.deq_lamada + \
                              (self.lin_loss + 10 * self.dual_lin_loss) * self.lin_lamada
        else:
            self.final_loss = self.dual_deq_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.final_loss)

        tf.summary.scalar('deq_loss', tf.reduce_mean(self.deq_loss))
        tf.summary.scalar('lin_loss', tf.reduce_mean(self.lin_loss))
        tf.summary.scalar('dual_lin_loss', tf.reduce_mean(self.dual_lin_loss))
        tf.summary.scalar('dual_deq_loss', tf.reduce_mean(self.dual_deq_loss))
        tf.summary.image('hdr', hdr)
        tf.summary.image('ldr', ldr)
        tf.summary.image('pred_clipped_hdr', self.pred_clipped_hdr)
        tf.summary.image('clipped_hdr', self.clipped_hdr)
        tf.summary.image('clipped_hdr_base', self.clipped_hdr_base)
        tf.summary.image('clipped_hdr_detail', self.clipped_hdr_detail)
        tf.summary.image('dual_pred_ldr', self.dual_pred_ldr)
        tf.summary.image('pred_ldr', self.pred_ldr)



    def train(self):
        b, h, w, c = self.batch_size, 512, 512, 3

        is_training = tf.placeholder(tf.bool)

        self.build_graph(self.ref_LDR_batch, self.ref_HDR_batch, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord=coord)
        self.sess.run(tf.global_variables_initializer())

        restorer1 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
        restorer1.restore(self.sess, self.deq_ckpt)

        restorer2 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'crf_feature_net' in var.name])
        restorer2.restore(self.sess, self.linear_indi_log_dir)

        restorer3 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'icrf_feature_net' in var.name])
        restorer3.restore(self.sess, self.non_linear_log_dir)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.logdir_path, 'summary'),
            self.sess.graph,
        )


        for it in range(self.it_num):
            print(it)
            if it == 0 or it % (self.block_num) == (self.block_num - 1):
                print('start save')
                checkpoint_path = os.path.join(self.logdir_path, 'model.ckpt')
                saver.save(self.sess, checkpoint_path, global_step=it)
                # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
                print('finish save')
            _, summary_val = self.sess.run([self.train_op, summary], {is_training: True})
            if it == 0 or it % (self.block_num) == (self.block_num - 1):
                summary_writer.add_summary(summary_val, it)
                logging.info('test')


        coord.request_stop()
        coord.join(threads)

