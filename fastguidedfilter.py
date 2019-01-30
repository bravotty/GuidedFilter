# coding=utf-8
# function q = fastguidedfilter(I, p, r, eps, s) - MATLAB
# %    GUIDEDFILTER O(1) time implementation of guided filter.
# I:   guidance Image 
# p:   filtering input image
# r:   window redius
# eps: normalize parameter
# s:   sampling fraction try s = r/4 to s=r
import os
import cv2
import numpy as np
from numpy.matlib import repmat
from scipy.misc import imresize

ra = 16
eps = 2
s = 4

def boxFilter(imSrc, r):
    hei, wid = imSrc.shape    
    imDst = np.zeros([hei, wid])
    imCum = np.cumsum(imSrc, axis=0)
    imDst[0:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[0:hei-2*r-1, :] 
    imDst[hei-r:hei, :] = repmat(imCum[hei-1, :], r, 1) - imCum[hei-2*r-1:hei-r-1, :]
    imCum = np.cumsum(imDst, axis=1)    
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]    
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, 0:wid-2*r-1]
    repmat(imCum[:,wid-1], r, 1).shape = (imCum.shape[0], r)
    y = np.transpose(repmat(imCum[:,wid-1], r, 1))
    imDst[:, wid-r:wid] = y - imCum[:, wid-2*r-1:wid-r-1]
    return imDst

def guideFilter(I, ra=16, eps=2, s=4):
    I_sub = imresize(I, 1/s, 'nearest')
    I_sub = I_sub/255
    p_sub = imresize(I, 1/s, 'nearest')
    p_sub = p_sub/255
    r_sub = int(ra / s)
    hei, wid = I_sub.shape
    m = np.ones([hei, wid])
    N = boxFilter(m, r_sub)
    mean_I = boxFilter(I_sub, r_sub) / N
    mean_p = boxFilter(p_sub, r_sub) / N
    mean_Ip = boxFilter(I_sub * p_sub, r_sub) / N
    # this is the covariance of (I, p) in each local patch.
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = boxFilter(I_sub * I_sub, r_sub) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = boxFilter(a, r_sub) / N
    mean_b = boxFilter(b, r_sub) / N
    mean_a = cv2.resize(mean_a, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    I = I / 255
    q = mean_a * I + mean_b
    return q

if __name__ == '__main__':
    image_dir   = './demo_image/'
    rainy_image = cv2.imread(os.path.join(image_dir, 'rainy_image.jpg')) 
    gt_image    = cv2.imread(os.path.join(image_dir, 'ground_truth.jpg')) 
    b, g, r     = cv2.split(rainy_image)
    h, w, c     = rainy_image.shape
    base_layer  = np.zeros([h, w, c])
    base_layer[:, :, 0] = guideFilter(b, ra, eps, s)
    base_layer[:, :, 1] = guideFilter(g, ra, eps, s)
    base_layer[:, :, 2] = guideFilter(r, ra, eps, s)
    rainy_image  = rainy_image / 255
    detail_layer = rainy_image - base_layer
    cv2.imshow('rainy_image', rainy_image)
    cv2.imshow('base_layer',  base_layer)
    cv2.imshow('detail_layer', detail_layer)
    cv2.waitKey(3)
    cv2.imwrite(image_dir + 'base_layer.jpg',   base_layer*255)
    cv2.imwrite(image_dir + 'detail_layer.jpg', detail_layer*255)
    # for the GT image - the same.
    # b_, g_, r_ = cv2.split(gt_image)
    # base_layer_ = np.zeros([h, w, c])
    # base_layer_[:, :, 0] = guideFilter(b_, ra, eps, s)
    # base_layer_[:, :, 1] = guideFilter(g_, ra, eps, s)
    # base_layer_[:, :, 2] = guideFilter(r_, ra, eps, s)
    # gt_image = gt_image / 255
    # detail_layer_ = gt_image - base_layer_
    # cv2.imshow('gt_image', gt_image)
    # cv2.imshow('base_layer_',  base_layer_)
    # cv2.imshow('detail_layer_', detail_layer_)
    # cv2.waitKey(0)


