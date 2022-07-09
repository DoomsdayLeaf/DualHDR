import tensorflow as tf
import numpy as np
import os
import glob
import cv2

class Convert_To(object):
    def __init__(self, sess, args):
        self.data_eye_path = 'result/Show/HDRCNN-show/'
        #self.data_real_path = 'result/HDRCNN_my/Real/'


    def run_convert(self):
        HDRs_eye = sorted(glob.glob(self.data_eye_path + '/*.exr'))
        #HDRs_real = sorted(glob.glob(self.data_real_path + '/*.exr'))

        for i, scene_dir in enumerate(HDRs_eye):
            print('processing %05d : %s' %(i, scene_dir))
            ori_img = scene_dir
            dst_img = ori_img.replace('exr', 'hdr')
            cmd = 'pfsin ' + ori_img + '|pfsout ' + dst_img
            os.system(cmd)

        '''
        for i, scene_dir in enumerate(HDRs_real):
            print('processing %05d : %s' %(i, scene_dir))
            ori_img = scene_dir
            dst_img = ori_img.replace('exr', 'hdr')
            cmd = 'pfsin ' + ori_img + '|pfsout ' + dst_img
            os.system(cmd)
        '''

        print("Finished!\nTotal number of patches:")

