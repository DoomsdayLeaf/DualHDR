from __future__ import division
import tensorflow as tf
#import tensorflow.contrib.slim as slim

import numpy as np
import os
import cv2
import math

from matplotlib import pyplot as plt

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    #slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    return model_vars

def str2bool(x):
    return x.lower() in ('true')


def histogram_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
    plt.show()

'''
rgb to lum
input:
    img: rgb(H, W, 3)
    kernel: (3, 3)
output:
    lum: xyz(H, W, 3)
'''
def change_domain(img, kernel):
    img_out = np.zeros_like(img)
    
    img_out[:, :, 0] = img[:, :, 0] * kernel[0, 0] + img[:, :, 1] * kernel[0, 1] + img[:, :, 2] * kernel[0, 2]
    img_out[:, :, 1] = img[:, :, 0] * kernel[1, 0] + img[:, :, 1] * kernel[1, 1] + img[:, :, 2] * kernel[1, 2]
    img_out[:, :, 2] = img[:, :, 0] * kernel[2, 0] + img[:, :, 1] * kernel[2, 1] + img[:, :, 2] * kernel[2, 2]
    
    return img_out
    
def hdr_rgb2xyz(img):
    kernel = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2127, 0.7151, 0.0722], 
                       [0.0193, 0.1192, 0.9505]])
    img_xyz = change_domain(img, kernel)
    return img_xyz

def hdr_xyz2rgb(img):
    kernel = np.array([[ 3.2410, -1.5374, -0.4986],
                       [-0.9692,  1.8760,  0.0416], 
                       [ 0.0556, -0.2040,  1.0570]])
    img_rgb = change_domain(img, kernel)
    return img_rgb

def ldr_rgb2yuv(img):
    kernel = np.array([[ 0.299,  0.587,  0.114],
                       [-0.169, -0.331,  0.500], 
                       [ 0.500, -0.419, -0.081]])
    img_yuv = change_domain(img, kernel)
    return img_yuv




"""
refrence: Photographic Tone Reproduction for Digital Images
params:
    x: LDR/HDR image

iTMO:
    I_out(i, j) = k * ( (I_in(i, j) - I_in_min) / (I_in_max - I_in_min) )^y
    D_out(i, j) = (I_out(i, j) * (D_in(i, j) / I_in(i, j)) ) ^ s

TMO:
    D(i, j) = (D_max - D_min) * (log(I + τ) - log(I_min + τ)) / ( log(I_max + τ) -  log(I_min + τ)) + D_min
"""
def TMO(img):
    img_xyz = hdr_rgb2xyz(img)
    I = np.zeros_like(img)
    I[:, :, 0] = img_xyz[:, :, 1]
    I[:, :, 1] = img_xyz[:, :, 1]
    I[:, :, 2] = img_xyz[:, :, 1]
    
    I_max = I.max()
    I_min = I.min()
    alpha = 10.0 / 255.0
    gamma = 2.2
    
    I_out = 255 * (np.log(I + alpha) - np.log(I_min + alpha)) / (np.log(I_max + alpha) - np.log(I_min + alpha))
    img_tm = np.power((I_out * (img / I)), gamma)
    
    return img_tm
    
def iTMO(img):
    eps = 1.0/255.0
    img_yuv = ldr_rgb2yuv(img)
    I = np.zeros_like(img)
    I[:, :, 0] = img_yuv[:, :, 0]
    I[:, :, 1] = img_yuv[:, :, 0]
    I[:, :, 2] = img_yuv[:, :, 0]
    
    I_max = I.max()
    I_min = I.min()
    if I_min < 1e-5:
        I = I + eps
        I_min = I.min()
    I_ave = I.mean()
    I_log = np.log(I + eps)
    I_h = np.exp(I_log.mean())
    k = (np.log(I_h) - np.log(I_min)) / (np.log(I_max) - np.log(I_min))
    alpha = 0.5
    gamma = 1/ 2.2
    
    I_out = k * np.power(((I - I_min) / (I_max - I_min)), alpha)
    
    img_itm = img * np.power((I_out / I), gamma)
    
    return img_itm
 

def normalize(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(np.float32)
    
def compute_psnr(img_orig, img_out, peak=1.0):
    img_orig_nor = normalize(img_orig)
    img_out_nor = normalize(img_out)
    mse = np.mean(np.square(img_orig_nor - img_out_nor))
    psnr = 10 * np.log10(peak*peak / mse)
    return psnr

def get_dataset_path(dataset="Real"):
    data_path = '/home/dataset/HDR/HDR-Real/LDR_in/'
    label_path = '/home/dataset/HDR/HDR-Real/HDR_gt/'

    if dataset == "Real":
        data_path = '/home/dataset/HDR/HDR-Real/LDR_in/'
        label_path = '/home/dataset/HDR/HDR-Real/HDR_gt/'
    elif dataset == "Strangers":
        data_path = '/home/dataset/HDR/HDR_video/LDR21/Strangers/'
        label_path = '/home/dataset/HDR/HDR_video/HDR/Strangers/'

    return data_path, label_path

def get_data_shape(dataset="Real"):
    if dataset == "Real":
        return [9759, 512, 512, 3]
    elif dataset == "Strangers":
        return [303, 1080, 2048, 3]
        
def get_train_size(batch_size, dataset="Real", sz=[256, 256, 3]):
    _, data_h, data_w, _ = get_data_shape(dataset)
    h_iter = math.floor(data_h / sz[0])
    w_iter = math.floor(data_w / sz[1])
    out_num = batch_size * h_iter * w_iter
    return out_num

def check_data_max_min(data_path):
    num, _, _, _ = get_data_shape()
    
    max_val = 0.0
    min_val = 266.0
    for i in range(num):
        hdr_path = data_path + str(i).rjust(5, '0') + '.hdr'
        tmp_hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        
        if max_val < tmp_hdr.max():
            max_val = tmp_hdr.max()
            
        if min_val > tmp_hdr.min():
            min_val = tmp_hdr.min()
    
    print("dataset hdr val is %.8f ~ %.8f" % (min_val, max_val))

def get_img_path(index, ldr_dir_path, hdr_dir_path, dataset="Real"):
    if dataset == "Strangers":
        hdr_path = hdr_dir_path + str(index).rjust(3, '0') + '.hdr'
        ldr_path = ldr_dir_path + str(index).rjust(3, '0') + '.png'
    else:
        hdr_path = hdr_dir_path + str(index).rjust(5, '0') + '.hdr'
        ldr_path = ldr_dir_path + str(index).rjust(5, '0') + '.jpg'

    return ldr_path, hdr_path


# hdr : 0-max
# ldr : 0-1
def read_data_by_hw(data_path, label_path, list_num, dataset="Real", sz=[256, 256, 3], gamma=2.2, max_lum=100000.0, data_domain="ori_domain", label_domain="ori_domain"):
    data_num, data_h, data_w, data_ch = get_data_shape(dataset)

    h_iter = math.floor(data_h / sz[0])
    w_iter = math.floor(data_w / sz[1])
    out_num = list_num.size * h_iter * w_iter

    hdr = np.zeros((out_num, sz[0], sz[1], sz[2]), dtype=np.float32)
    ldr = np.zeros((out_num, sz[0], sz[1], sz[2]), dtype=np.float32)
    
    for i in range(list_num.size):
        ldr_path, hdr_path = get_img_path(list_num[i], data_path, label_path, dataset=dataset)

        tmp_hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if False:
            if tmp_hdr.max() > max_lum:
                print("------------%s 's max val is %.08f over %.08f" % ((str(list_num[i]).rjust(5, '0') + '.hdr'), tmp_hdr.max(), max_lum))
                cv2.waitKey(0)
            tmp_hdr = np.clip(tmp_hdr, 0.0, max_lum)

        tmp_ldr = cv2.imread(ldr_path)
        tmp_ldr = tmp_ldr / 255.0
        tmp_ldr = np.clip(tmp_ldr, 0.0, 1.0)
        tmp_ldr = np.power(tmp_ldr, gamma)

        if False:
            tmp_itm = iTMO(tmp_ldr * 255.0)
            cv2.imshow("ori hdr", normalize(tmp_hdr))
            cv2.imshow("ori ldr", normalize(tmp_ldr))
            cv2.imshow("itm", normalize(tmp_itm))

            hdr_xyz = hdr_rgb2xyz(tmp_hdr)
            itm_xyz = hdr_rgb2xyz(tmp_itm)
            ldr_yuv = ldr_rgb2yuv(tmp_ldr * 255.0)
            print("hdr: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_hdr.min(), tmp_hdr.max(), hdr_xyz[:, :, 1].min(), hdr_xyz[:, :, 1].max()))
            print("itm: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_itm.min(), tmp_itm.max(), itm_xyz[:, :, 1].min(), itm_xyz[:, :, 1].max()))
            print("ldr: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_ldr.min(), tmp_ldr.max(), ldr_yuv[:, :, 0].min(), ldr_yuv[:, :, 0].max()))

            psnr = compute_psnr(tmp_itm, tmp_hdr)
            psnr_ori = compute_psnr(tmp_ldr, tmp_hdr)
            print("psnr: ori: %.08f dB  itm: %.08f dB  gain: %.08f" % (psnr_ori, psnr, (psnr - psnr_ori)))
            print("=============================")
            cv2.waitKey(0)

        for j1 in range(h_iter):
            for j2 in range(w_iter):
                hdr[i * h_iter * w_iter + j1 * w_iter + j2, ...] = tmp_hdr[sz[0] * j1: sz[0] * (j1+1), sz[1] * j2: sz[1] * (j2+1), :]
                ldr[i * h_iter * w_iter + j1 * w_iter + j2, ...] = tmp_ldr[sz[0] * j1: sz[0] * (j1+1), sz[1] * j2: sz[1] * (j2+1), :]

    if data_domain == "ori_domain":
        ldr = ldr * 255.0
    
    if label_domain == "normal_domain":
        hdr = normalize(hdr)

    return ldr, hdr

# hdr : 0-1
# ldr : 0-1
def read_data(data_path, label_path, list_num, block_num=2, gamma=2.2, max_lum=100000.0):
    data_num, data_h, data_w, data_ch = get_data_shape(block_num)
    out_n = list_num.size * block_num * block_num

    hdr = np.zeros((out_n, data_h, data_w, data_ch), dtype=np.float32)
    ldr = np.zeros((out_n, data_h, data_w, data_ch), dtype=np.float32)
    
    for i in range(list_num.size):
        hdr_path = label_path + str(list_num[i]).rjust(5, '0') + '.hdr'
        ldr_path = data_path + str(list_num[i]).rjust(5, '0') + '.jpg'

        tmp_hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if False:
            if tmp_hdr.max() > max_lum:
                print("------------%s 's max val is %.08f over %.08f" % ((str(list_num[i]).rjust(5, '0') + '.hdr'), tmp_hdr.max(), max_lum))
                cv2.waitKey(0)
            tmp_hdr = np.clip(tmp_hdr, 0.0, max_lum)

        tmp_ldr = cv2.imread(ldr_path)
        tmp_ldr = tmp_ldr / 255.0
        tmp_ldr = np.clip(tmp_ldr, 0.0, 1.0)
        tmp_ldr = np.power(tmp_ldr, gamma)

        if False:
            tmp_itm = iTMO(tmp_ldr * 255.0)
            cv2.imshow("ori hdr", normalize(tmp_hdr))
            cv2.imshow("ori ldr", normalize(tmp_ldr))
            cv2.imshow("itm", normalize(tmp_itm))

            hdr_xyz = hdr_rgb2xyz(tmp_hdr)
            itm_xyz = hdr_rgb2xyz(tmp_itm)
            ldr_yuv = ldr_rgb2yuv(tmp_ldr * 255.0)
            print("hdr: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_hdr.min(), tmp_hdr.max(), hdr_xyz[:, :, 1].min(), hdr_xyz[:, :, 1].max()))
            print("itm: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_itm.min(), tmp_itm.max(), itm_xyz[:, :, 1].min(), itm_xyz[:, :, 1].max()))
            print("ldr: rgb: %.08f ~ %.08f  y: %.08f ~ %.08f" % (tmp_ldr.min(), tmp_ldr.max(), ldr_yuv[:, :, 0].min(), ldr_yuv[:, :, 0].max()))

            psnr = compute_psnr(tmp_itm, tmp_hdr)
            psnr_ori = compute_psnr(tmp_ldr, tmp_hdr)
            print("psnr: ori: %.08f dB  itm: %.08f dB  gain: %.08f" % (psnr_ori, psnr, (psnr - psnr_ori)))
            print("=============================")
            cv2.waitKey(0)

        for j1 in range(block_num):
            for j2 in range(block_num):
                hdr[i * block_num * block_num + j1 * block_num + j2, ...] = tmp_hdr[data_h * j1: data_h * (j1+1), data_w * j2: data_w * (j2+1), :]
                ldr[i * block_num * block_num + j1 * block_num + j2, ...] = tmp_ldr[data_h * j1: data_h * (j1+1), data_w * j2: data_w * (j2+1), :]

    return ldr, hdr