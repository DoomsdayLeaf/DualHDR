import logging

logging.basicConfig(level=logging.INFO)
import argparse
import os
from SingleImageNet.dequantization_net import Dequantization_net
from SingleImageNet.linearization_net import Linearization_net
from SingleImageNet import hallucination_net
from SingleImageNet.util import apply_rf
from SingleImageNet.refinement_net import Refinement_net
import numpy as np
import cv2
import glob
import time
import tensorflow as tf

class Test_Eye(object):
    def __init__(self, sess, args, config):
        self.sess = sess
        self.args = args
        self.config = config
        self.epsilon = 0.001

        self.batch_size = 1
        self.tfrecords_path = args.tf_records_log_dir
        self.test_imgs = args.test_dataset + "/LDR/"
        self.output_path = args.test_output_path

        self.pre_dnet = args.pre_dnet
        self.pre_inet = args.pre_inet
        self.pre_itnet = args.pre_itnet
        self.pre_oonet = args.pre_oonet

        self._clip = lambda x: tf.clip_by_value(x, 0, 1)

    def build_graph(
            self,
            ldr,  # [b, h, w, c]
            is_training,
    ):
        with tf.variable_scope("Dequantization_Net"):
            dequantization_model = Dequantization_net(is_train=is_training)
            pred_ldr = self._clip(dequantization_model.inference(ldr))

        
        lin_net = Linearization_net()
        pred_invcrf = lin_net.get_output(pred_ldr, is_training)
        pred_clipped_hdr = apply_rf(pred_ldr, pred_invcrf)

        with tf.variable_scope("Hallucination_Net"):
            net_test, vgg16_conv_layers_test = hallucination_net.model(pred_clipped_hdr, 12, False)
            y_predict_test = net_test.outputs
            self.pred_hdr = (pred_clipped_hdr) +  y_predict_test

        # Refinement-Net
        with tf.variable_scope("Refinement_Net"):
            with tf.variable_scope("diff_ref_net"):
                diff_refinement_model = Refinement_net(is_train=is_training,
                                                       batch_size=self.batch_size,
                                                       use_diff=True,
                                                       use_rfb=True,
                                                       use_non_local=False,
                                                       use_bd=False)
                self.ref_hdr_fianl = tf.nn.relu(
                    diff_refinement_model.inference(
                        tf.concat(
                            [ldr, pred_clipped_hdr, pred_ldr],
                            -1)),
                        name='output'
                        )


    def run_test(self):
        ldr = tf.placeholder(tf.float32, [1, 512, 512, 3])
        is_training = tf.placeholder(tf.bool)

        self.build_graph(ldr, is_training)

        # load parameters
        restorer0 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
        restorer0.restore(self.sess, self.pre_oonet)

        restorer2 = tf.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if
                                             'crf_feature_net' in var.name or 'ae_invcrf_' in var.name])
        restorer2.restore(self.sess, self.pre_oonet)

        restorer3 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
        restorer3.restore(self.sess, self.pre_oonet)

        restorer4 = tf.train.Saver(
            var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'diff_ref_net' in var.name])
        restorer4.restore(self.sess, self.pre_oonet)


        # mkdir for output
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # load data and run test
        ldr_imgs = glob.glob(os.path.join(self.test_imgs, '*.png'))
        ldr_imgs.extend(glob.glob(os.path.join(self.test_imgs, '*.jpg')))
        ldr_imgs = sorted(ldr_imgs)
        for ldr_img_path in ldr_imgs:
            print(ldr_img_path)
            ldr_img = cv2.imread(ldr_img_path)

            ldr_val = np.flip(ldr_img, -1).astype(np.float32) / 255.0

            ORIGINAL_H = ldr_val.shape[0]
            ORIGINAL_W = ldr_val.shape[1]

            """resize to 64x"""
            if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
                RESIZED_H = int(np.ceil(float(ORIGINAL_H) / 64.0)) * 64
                RESIZED_W = int(np.ceil(float(ORIGINAL_W) / 64.0)) * 64
                ldr_val = cv2.resize(ldr_val, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC)

            padding = 32
            ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

            HDR_out_val = self.sess.run(self.ref_hdr_fianl, {
                ldr: [ldr_val],
                is_training: False,
            })

            HDR_out_val = np.flip(HDR_out_val[0], -1)
            HDR_out_val = HDR_out_val[padding:-padding, padding:-padding]
            if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
                HDR_out_val = cv2.resize(HDR_out_val, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.output_path, os.path.split(ldr_img_path)[-1][:-3]+'hdr'), HDR_out_val)


