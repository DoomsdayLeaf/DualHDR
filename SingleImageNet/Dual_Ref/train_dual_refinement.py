import logging

logging.basicConfig(level=logging.INFO)
import argparse
import tensorflow as tf
from SingleImageNet.util import get_tensor_shape, apply_rf, log10,get_l2_loss, normaliza_display
import os
import glob
from random import shuffle
from SingleImageNet.dequantization_net import Dequantization_net
from SingleImageNet.linearization_net import Linearization_net
from SingleImageNet import hallucination_net
from SingleImageNet.refinement_net import Refinement_net
from Dual_SingleImageNet.quantization_net import Quantization_net
from Dual_SingleImageNet.non_linearization_net import Non_Linearization_net
from Dual_SingleImageNet.clipping_net import Clipping_Net

class Finetune_real_dataset(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.ref_batch_size
        self.it_num = args.ref_epoch
        self.logdir_path = args.ref_log_dir
        self.tfrecords_path = args.tf_records_log_dir
        self.deq_ckpt = args.deq_log_dir
        self.lin_ckpt = args.linear_log_dir
        self.hal_ckpt = args.hal_log_dir
        self.ref_indi_log_dir = args.ref_indi_log_dir
        self.oonet_joint_train = args.oonet_joint_train

        self.use_rfb = True
        self.use_non_local = True
        self.use_diff = True

        self.dual_lamada = 0
        self.dual_ref_mode = "AEncoder"
        self.train_only_ref = False

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

        # pre train model
        # Dequantization-Net
        with tf.variable_scope("Dequantization_Net"):
            dequantization_model = Dequantization_net(is_train=True)
            pred_ldr = self._clip(dequantization_model.inference(ldr))

        # Linearization-Net
        lin_net = Linearization_net()
        pred_invcrf = lin_net.get_output(pred_ldr, True)
        pred_clipped_hdr = apply_rf(pred_ldr, pred_invcrf)

        # Hallucination-Net
        thr = 0.12
        alpha = tf.reduce_max(pred_clipped_hdr, reduction_indices=[3])
        alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
        alpha = tf.reshape(alpha, [-1, tf.shape(pred_clipped_hdr)[1], tf.shape(pred_clipped_hdr)[2], 1])
        alpha = tf.tile(alpha, [1, 1, 1, 3])
        with tf.variable_scope("Hallucination_Net"):
            net_test, vgg16_conv_layers_test = hallucination_net.model(pred_clipped_hdr, self.batch_size, True)
            self.y_predict_test = net_test.outputs
            self.y_predict_test = tf.nn.relu(self.y_predict_test)
            self.pred_hdr = (pred_clipped_hdr) + alpha * self.y_predict_test

        """ build primary model """
        # Refinement-Net
        with tf.variable_scope("Refinement_Net"):
            with tf.variable_scope("diff_ref_net"):
                diff_refinement_model = Refinement_net(is_train=is_training,
                                                       batch_size=self.batch_size,
                                                       use_diff=True,
                                                       use_rfb=True,
                                                       use_non_local=False,
                                                       use_bd=False)
                self.ref_hdr_final = tf.nn.relu(
                    diff_refinement_model.inference(
                        tf.concat(
                            [self.ref_hdr_non_loc, pred_clipped_hdr, pred_ldr],
                            -1)))

        """ build dual task """
        # Clipping-Net
        with tf.variable_scope("CLipping_Net"):
            clipp_net = Clipping_Net()
            self.dual_pred_clipped_hdr, self.dual_pred_clipped_hdr_base, self.dual_pred_clipped_hdr_detail = clipp_net.model(self.ref_hdr_final)

        # Non-Linearization-Net
        non_lin_net = Non_Linearization_net()
        pred_crf = non_lin_net.get_output(self.dual_pred_clipped_hdr, True)
        dual_pred_ldr = apply_rf(self.dual_pred_clipped_hdr, pred_crf)

        # Quantization-Net
        with tf.variable_scope("Quantization_Net"):
            quantization_model = Quantization_net(is_train=True)
            dual_pred_jpg = quantization_model.model(dual_pred_ldr)

        # Dual-Refinement-Net
        with tf.variable_scope("Dual_Refinement_Net"):
            dual_refinement_model = Dual_Refinement_net(is_train=is_training, model_name=self.dual_ref_mode)
            if self.dual_ref_mode == 'res_net':
                dual_ref_jpg = tf.nn.relu(dual_refinement_model.inference(dual_pred_jpg))
            else:
                dual_ref_jpg = tf.nn.relu(dual_refinement_model.inference(tf.concat([dual_pred_jpg, dual_pred_ldr, self.dual_pred_clipped_hdr], -1)))

        if self.train_only_ref:
            t_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Refinement_Net|Dual_Refinement_Net')
        else:
            t_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
        print('all layers:')
        for var in t_vars:
            print(var.name)

        _log = lambda x: tf.log(x + 1.0 / 255.0)
        """ compute dual loss """
        self.dual_final_loss = get_l2_loss(dual_ref_jpg, ldr)

        """ compute primary loss """
        hdr_gamma = tf.log(1.0 + 10.0 * hdr) / tf.log(1.0 + 10.0)

        self.ref_hdr_final = self.ref_hdr_final / (1e-6 + tf.reduce_mean(self.ref_hdr_final, axis=[1, 2, 3], keepdims=True)) * 0.5
        ref_hdr_final_gamma = tf.log(1.0 + 10.0 * self.ref_hdr_final) / tf.log(1.0 + 10.0)
        self.loss = tf.reduce_mean(tf.abs(ref_hdr_final_gamma - hdr_gamma))

        self.final_loss = self.loss
        if self.oonet_joint_train == "indi":
            self.final_loss = self.loss
        elif self.oonet_joint_train == "joint_pair":
            self.final_loss = self.final_loss + self.dual_final_loss * self.dual_lamada
        else:
            self.final_loss = self.dual_final_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.final_loss, colocate_gradients_with_ops=True, var_list=t_vars)

        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        tf.summary.scalar('final_loss', tf.reduce_mean(self.final_loss))
        tf.summary.scalar('dual_final_loss', tf.reduce_mean(self.dual_final_loss))
        tf.summary.image('hdr', hdr)
        tf.summary.image('jpg', ldr)
        tf.summary.image('pred_hdr', self.pred_hdr)
        tf.summary.image('pred_clipped_hdr', pred_clipped_hdr)
        tf.summary.image('pred_ldr', pred_ldr)
        tf.summary.image('dual_pred_clipped_hdr', self.dual_pred_clipped_hdr)
        tf.summary.image('dual_pred_ldr', dual_pred_ldr)
        tf.summary.image('dual_pred_jpg', dual_pred_jpg)
        tf.summary.image('dual_ref_jpg', dual_ref_jpg)
        tf.summary.histogram('hdr_histo', hdr)
        tf.summary.histogram('ldr_histo', ldr)
        tf.summary.histogram('dual_ref_jpg_histo', dual_ref_jpg)

    def train(self):
        b, h, w, c = self.batch_size, 512, 512, 3

        is_training = tf.placeholder(tf.bool)

        self.build_graph(self.ref_LDR_batch, self.ref_HDR_batch, is_training)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord=coord)
        self.sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            if 'Refinement_Net' in variable.name:
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
        print(total_parameters)


        if self.oonet_joint_train != "indi":
            restorer0 = tf.train.Saver(
                var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
            restorer0.restore(self.sess, self.ref_indi_log_dir)

            restorer2 = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if
                                                 'crf_feature_net' in var.name or 'ae_invcrf_' in var.name or 'crf_net' in var.name])
            restorer2.restore(self.sess, self.ref_indi_log_dir)

            restorer3 = tf.train.Saver(
                var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
            restorer3.restore(self.sess, self.ref_indi_log_dir)

            restorer4 = tf.train.Saver(
                var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'diff_ref_net' in var.name])
            restorer4.restore(self.sess, self.ref_indi_log_dir)
        else:
            restorer0 = tf.train.Saver(
                var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
            restorer0.restore(self.sess, self.deq_ckpt)

            restorer2 = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if
                                                 'crf_feature_net' in var.name or 'ae_invcrf_' in var.name or 'crf_net' in var.name])
            restorer2.restore(self.sess, self.lin_ckpt)

            restorer3 = tf.train.Saver(
                var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
            restorer3.restore(self.sess, self.hal_ckpt)
            self.logdir_path = self.ref_indi_log_dir

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
