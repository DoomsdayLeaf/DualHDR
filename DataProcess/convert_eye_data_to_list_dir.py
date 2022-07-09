import tensorflow as tf
import numpy as np
import os
import glob
import cv2

class Convert_To(object):
    def __init__(self, sess, args):
        self.data_eye_path = 'test/ori-HDR-Eye/'


    def run_convert(self):
        num = 46

        for i in range(0, num):
            dir_path = str(i).rjust(5, '0')
            out_hdr_name = dir_path + '.hdr'
            out_ldr_name = dir_path + '.png'
            print("processing %s" % (dir_path))

            aeo_hdr_path = self.data_eye_path + dir_path + '/aeo.hdr'
            aeo_ldr_path = self.data_eye_path + dir_path + '/aeo.png'
            aeo_hdr_out_path = self.data_eye_path + 'AEO/HDR/' + out_hdr_name
            aeo_ldr_out_path = self.data_eye_path + 'AEO/LDR/' + out_ldr_name
            cmd = 'cp ' + aeo_hdr_path + ' ' + aeo_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + aeo_ldr_path + ' ' + aeo_ldr_out_path
            os.system(cmd)

            drtmo_hdr_path = self.data_eye_path + dir_path + '/drtmo.hdr'
            drtmo_ldr_path = self.data_eye_path + dir_path + '/drtmo.png'
            drtmo_hdr_out_path = self.data_eye_path + 'DrTmo/HDR/' + out_hdr_name
            drtmo_ldr_out_path = self.data_eye_path + 'DrTmo/LDR/' + out_ldr_name
            cmd = 'cp ' + drtmo_hdr_path + ' ' + drtmo_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + drtmo_ldr_path + ' ' + drtmo_ldr_out_path
            os.system(cmd)

            expand_hdr_path = self.data_eye_path + dir_path + '/expand.hdr'
            expand_ldr_path = self.data_eye_path + dir_path + '/expand.png'
            expand_hdr_out_path = self.data_eye_path + 'Expand/HDR/' + out_hdr_name
            expand_ldr_out_path = self.data_eye_path + 'Expand/LDR/' + out_ldr_name
            cmd = 'cp ' + expand_hdr_path + ' ' + expand_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + expand_ldr_path + ' ' + expand_ldr_out_path
            os.system(cmd)

            gt_hdr_path = self.data_eye_path + dir_path + '/gt.hdr'
            gt_ldr_path = self.data_eye_path + dir_path + '/gt.png'
            gt_hdr_out_path = self.data_eye_path + 'GT/HDR/' + out_hdr_name
            gt_ldr_out_path = self.data_eye_path + 'GT/LDR/' + out_ldr_name
            cmd = 'cp ' + gt_hdr_path + ' ' + gt_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + gt_ldr_path + ' ' + gt_ldr_out_path
            os.system(cmd)

            hdrcnn_hdr_path = self.data_eye_path + dir_path + '/hdrcnn.hdr'
            hdrcnn_ldr_path = self.data_eye_path + dir_path + '/hdrcnn.png'
            hdrcnn_hdr_out_path = self.data_eye_path + 'HDRCNN/HDR/' + out_hdr_name
            hdrcnn_ldr_out_path = self.data_eye_path + 'HDRCNN/LDR/' + out_ldr_name
            cmd = 'cp ' + hdrcnn_hdr_path + ' ' + hdrcnn_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + hdrcnn_ldr_path + ' ' + hdrcnn_ldr_out_path
            os.system(cmd)

            hpeo_hdr_path = self.data_eye_path + dir_path + '/hpeo.hdr'
            hpeo_ldr_path = self.data_eye_path + dir_path + '/hpeo.png'
            hpeo_hdr_out_path = self.data_eye_path + 'Hpeo/HDR/' + out_hdr_name
            hpeo_ldr_out_path = self.data_eye_path + 'Hpeo/LDR/' + out_ldr_name
            cmd = 'cp ' + hpeo_hdr_path + ' ' + hpeo_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + hpeo_ldr_path + ' ' + hpeo_ldr_out_path
            os.system(cmd)

            koeo_hdr_path = self.data_eye_path + dir_path + '/koeo.hdr'
            koeo_ldr_path = self.data_eye_path + dir_path + '/koeo.png'
            koeo_hdr_out_path = self.data_eye_path + 'Koeo/HDR/' + out_hdr_name
            koeo_ldr_out_path = self.data_eye_path + 'Koeo/LDR/' + out_ldr_name
            cmd = 'cp ' + koeo_hdr_path + ' ' + koeo_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + koeo_ldr_path + ' ' + koeo_ldr_out_path
            os.system(cmd)

            meo_hdr_path = self.data_eye_path + dir_path + '/meo.hdr'
            meo_ldr_path = self.data_eye_path + dir_path + '/meo.png'
            meo_hdr_out_path = self.data_eye_path + 'Meo/HDR/' + out_hdr_name
            meo_ldr_out_path = self.data_eye_path + 'Meo/LDR/' + out_ldr_name
            cmd = 'cp ' + meo_hdr_path + ' ' + meo_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + meo_ldr_path + ' ' + meo_ldr_out_path
            os.system(cmd)

            ipipe_hdr_path = self.data_eye_path + dir_path + '/ours.hdr'
            ipipe_ldr_path = self.data_eye_path + dir_path + '/ours.png'
            ipipe_hdr_out_path = self.data_eye_path + 'iPipe/HDR/' + out_hdr_name
            ipipe_ldr_out_path = self.data_eye_path + 'iPipe/LDR/' + out_ldr_name
            cmd = 'cp ' + ipipe_hdr_path + ' ' + ipipe_hdr_out_path
            os.system(cmd)
            cmd = 'cp ' + ipipe_ldr_path + ' ' + ipipe_ldr_out_path
            os.system(cmd)

        print("Finished!\nTotal number of patches:")

