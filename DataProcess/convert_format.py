import tensorflow as tf
import numpy as np
import os
import glob
import cv2

class Convert_To(object):
    def __init__(self, sess, args):
        self.data_path = 'result/RAISE/Expand_my/'


    def run_convert(self):
        HDRs = sorted(glob.glob(self.data_path + '/*.hdr'))

        for i, scene_dir in enumerate(HDRs):
            print('processing %05d : %s' %(i, scene_dir))
            ori_img = scene_dir
            dst_img = ori_img.replace('_prediction', '')
            cmd = 'mv ' + ori_img + ' ' + dst_img
            os.system(cmd)


        print("Finished!\nTotal number of patches:")

