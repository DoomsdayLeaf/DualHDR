import logging

logging.basicConfig(level=logging.INFO)
import argparse
import tensorflow as tf
import os
import glob
from random import shuffle
from SingleImageNet.dequantization_net import Dequantization_net
from Dual_SingleImageNet.quantization_net import Quantization_net
from SingleImageNet import util
from DataProcess import dataset

class train_dual_deq_net(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.deq_batch_size
        self.it_num = 100000
        self.logdir_path = args.deq_log_dir
        self.tfrecords_path = args.tf_records_log_dir
        self.hdr_prefix = args.hdr_prefix
        self.indi_deq_ckpt = args.deq_indi_log_dir
        self.deq_joint_train = args.deq_joint_train

        self.dual_lamada = 1

        self.tfrecord_list = glob.glob(os.path.join(self.tfrecords_path, '*.tfrecords'), recursive=True)
        print(len(self.tfrecord_list))
        assert (self.tfrecord_list)
        shuffle(self.tfrecord_list)
        print('\n\n====================\ntfrecords list:')
        [print(f) for f in self.tfrecord_list]
        print('====================\n\n')

        with tf.device('/cpu:0'):
            self.filename_queue = tf.train.string_input_producer(self.tfrecord_list)
            self.ref_HDR_batch, self.ref_LDR_batch = self.load_data(self.filename_queue)

        self._clip = lambda x: tf.clip_by_value(x, 0, 1)

    def load_data(self, filename_queue):

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
        b, h, w, c, = util.get_tensor_shape(img)

        const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

        bit = const_bit
        s = (2 ** bit) - 1

        img = self._clip(img)
        img = tf.round(s * img) / s
        img = self._clip(img)

        return img

    def build_graph(
            self,
            hdr,  # [b, h, w, c]
            crf,  # [b, k]
            t,  # [b]
            is_training):

        b, h, w, c, = util.get_tensor_shape(hdr)
        b, k, = util.get_tensor_shape(crf)
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
        self.clipped_hdr_t = self._clip(self._hdr_t)

        # Camera response function
        self.ldr = util.apply_rf(self.clipped_hdr_t, crf)

        # Quantization and JPEG compression
        self.quantized_hdr = tf.round(self.ldr * 255.0)
        self.quantized_hdr_8bit = tf.cast(self.quantized_hdr, tf.uint8)
        jpeg_img_list = []
        for i in range(self.batch_size):
            II = self.quantized_hdr_8bit[i]
            II = tf.image.adjust_jpeg_quality(II, int(round(float(i)/float(self.batch_size-1)*10.0+90.0)))
            jpeg_img_list.append(II)
        self.jpeg_img = tf.stack(jpeg_img_list, 0)
        self.jpeg_img_float = tf.cast(self.jpeg_img, tf.float32) / 255.0
        self.jpeg_img_float.set_shape([None, 256, 256, 3])


        # loss mask to exclude over-/under-exposed regions
        gray = tf.image.rgb_to_grayscale(self.jpeg_img)
        over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
        over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
        over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
        under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
        under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
        under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
        extreme_cases = tf.logical_or(over_exposed, under_exposed)
        self.loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

        """ build primary model """
        with tf.variable_scope("Dequantization_Net"):
            self.model = Dequantization_net(is_train=is_training)
            self.pred_ldr = self._clip(self.model.inference(self.jpeg_img_float))

        """ build dual model """
        with tf.variable_scope("Quantization_Net"):
            self.dual_model = Quantization_net(is_train=is_training)
            self.dual_pred_jpg = self.dual_model.get_output(self.pred_ldr)

        """ compute dual loss """
        self.dual_final_loss = util.get_l2_loss(self.dual_pred_jpg, self.jpeg_img_float)

        """ compute primary loss """
        self.loss = util.get_l2_loss(self.pred_ldr, self.ldr)
        if self.deq_joint_train == "joint_pair":
            self.final_loss = self.loss + self.dual_final_loss * self.dual_lamada
        else:
            self.final_loss = self.dual_final_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean(self.loss*self.loss_mask))

        """ comput some else """
        mse = tf.reduce_mean((self.pred_ldr - self.ldr) ** 2)
        self.psnr = 20.0 * util.log10(1.0) - 10.0 * util.log10(mse)
        mse = tf.reduce_mean((self.jpeg_img_float - self.ldr) ** 2)
        self.psnr_no_q = 20.0 * util.log10(1.0) - 10.0 * util.log10(mse)

        self.show_loss = tf.reduce_mean(self.loss)
        self.show_final_loss = tf.reduce_mean(self.final_loss)
        self.show_final_dual_loss = tf.reduce_mean(self.dual_final_loss)

        tf.summary.scalar('loss', self.show_loss)
        tf.summary.scalar('final_loss', self.show_final_loss)
        tf.summary.scalar('dual_final_loss', self.show_final_dual_loss)
        tf.summary.image('ldr', self.ldr)
        tf.summary.image('jpeg_img_float', self.jpeg_img_float)
        tf.summary.image('pred', self.pred_ldr)
        tf.summary.scalar('loss_mask 0', tf.squeeze(self.loss_mask[0]))
        tf.summary.scalar('loss_mask 1', tf.squeeze(self.loss_mask[1]))
        tf.summary.scalar('loss_mask 2', tf.squeeze(self.loss_mask[2]))

    def train(self):
        b, h, w, c = self.batch_size, 512, 512, 3

        hdr = tf.placeholder(tf.float32, [None, None, None, c])
        crf = tf.placeholder(tf.float32, [None, None])
        t = tf.placeholder(tf.float32, [None])
        is_training = tf.placeholder(tf.bool)

        self.build_graph(hdr, crf, t, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # ---

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print('total params: ')
        print(total_parameters)

        self.sess.run(tf.global_variables_initializer())

        restorer1 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
        restorer1.restore(self.sess, self.indi_deq_ckpt)

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

