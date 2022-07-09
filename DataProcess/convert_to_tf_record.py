import tensorflow as tf
import numpy as np
import os
import glob
import cv2

class convert_to_tf_record(object):
    def __init__(self, sess, args):
        self.data_path = args.coverHDR_path
        self.out_dir = 'tf_records/256_64_b32_tfrecords'
        self.patch_size = 256
        self.patch_stride = 64
        self.batch_size = 32
        self.count = 0
        self.cur_writing_path = os.path.join(self.out_dir, "train_{:d}_{:04d}.tfrecords".format(self.patch_stride, 0))
        self.writer = tf.python_io.TFRecordWriter(self.cur_writing_path)

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_example(self, h1, h2, w1, w2, ref_HDR, ref_LDR):
        cur_batch_index = self.count // self.batch_size

        if self.count % self.batch_size == 0:
            self.writer.close()
            self.cur_writing_path = os.path.join(self.out_dir,
                                            "train_{:d}_{:04d}.tfrecords".format(self.patch_stride, cur_batch_index))
            self.writer = tf.python_io.TFRecordWriter(self.cur_writing_path)

        ref_HDR_patch = ref_HDR[h1:h2, w1:w2, ::-1]
        ref_LDR_patch = ref_LDR[h1:h2, w1:w2, ::-1]

        """extreme cases filtering"""
        ref_LDR_patch_gray = cv2.cvtColor(ref_LDR_patch, cv2.COLOR_RGB2GRAY)
        extreme_pixels = np.sum(ref_LDR_patch_gray >= 249.0) + np.sum(ref_LDR_patch_gray <= 6.0)
        if extreme_pixels <= 256 * 256 // 2:
            print('pass')

            self.count += 1

            # create example
            example = tf.train.Example(features=tf.train.Features(feature={
                'ref_HDR': self.bytes_feature(ref_HDR_patch.tostring()),
                'ref_LDR': self.bytes_feature(ref_LDR_patch.tostring()),
            }))
            self.writer.write(example.SerializeToString())
        else:
            print('filtered out')

    def run_convert(self):
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        HDRs_512 = sorted(glob.glob(self.data_path + 'HDR_gt/*.hdr'))
        LDRs_512 = sorted(glob.glob(self.data_path + 'LDR_in/*.jpg'))

        for i, scene_dir in enumerate(HDRs_512):
            if (i % 10 == 0):
                print('%d/%d' % (i, len(HDRs_512)))

            # read images
            ref_HDR = cv2.imread(HDRs_512[i], -1).astype(np.float32)  # read raw values
            ref_LDR = cv2.imread(LDRs_512[i]).astype(np.float32)   # read jpg

            h, w, c = ref_HDR.shape

            # generate patches
            for h_ in range(0, h - self.patch_size + 1, self.patch_stride):
                for w_ in range(0, w - self.patch_size + 1, self.patch_stride):
                    self.write_example(h_, h_ + self.patch_size, w_, w_ + self.patch_size, ref_HDR, ref_LDR)

            # deal with border patch
            if h % self.patch_size:
                for w_ in range(0, w - self.patch_size + 1, self.patch_stride):
                    self.write_example(h - self.patch_size, h, w_, w_ + self.patch_size)

            if w % self.patch_size:
                for h_ in range(0, h - self.patch_size + 1, self.patch_stride):
                    self.write_example(h_, h_ + self.patch_size, w - self.patch_size, w)

            if w % self.patch_size and h % self.patch_size:
                self.write_example(h - self.patch_size, h, w - self.patch_size, w)

        self.writer.close()
        print("Finished!\nTotal number of patches:", self.count)

