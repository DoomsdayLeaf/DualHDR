import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import shutil

class Convert_To_List_Dir(object):
    def __init__(self, sess, args):
        self.data_path = 'result/RAISE/'
        self.out_dir = 'result/RAISE/'
        self.test_num = 80


    def run_convert(self):
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        dir_nums = 16308
        self.count = 0
        for d_n in range(dir_nums):
            print("proces [%05d / %05d] imgs" % (d_n, dir_nums))
            dir_path = str(d_n).rjust(5, '0')
            HDR_ori = self.data_path + dir_path + '/gt.hdr'
            LDR_ori = self.data_path + dir_path + '/input.jpg'

            HDR_dst = self.out_dir + 'HDR/' + dir_path + '.hdr'
            LDR_dst = self.out_dir + 'LDR/' + dir_path + '.jpg'

            shutil.copy(LDR_ori, LDR_dst)
            shutil.copy(HDR_ori, HDR_dst)


        print("Finished!\nTotal number of patches:", self.count)

