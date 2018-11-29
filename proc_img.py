#!/usr/bin/python
# -*- coding=utf-8 -*-

import csv
import os
import pickle
import re
import imghdr
import tarfile
from time import time

import numpy as np
# import PIL.Image
# from scipy.misc import imresize
import cv2

def proc(save=True):
    data_dir = './data/train'
    # Result directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    imagenet_dir = '../Imagenet2012/img_par'
    # imagenet_dir = '../imagenet/imagenet_train_par'
    return store_img(imagenet_dir,savedir=data_dir,savename='orig_img_comb',is_save=save)


def get_category_id(image_filename):
    return int(image_filename.split('/')[-2][-8:])

def get_image_id(image_filename):
    img_id = int(image_filename.split('_')[-1][:-5])
    return img_id

def get_img(images):
    if not isinstance(images, list):
        images = [images]

    num_images = len(images)
    (h, w) = 224,224
    # (h, w) = 128,128
    img_all = []

    for k in range(0,num_images):
        print('img %d/%d' % (k+1,num_images))
        
        # img = imresize(PIL.Image.open(images[k]).convert('RGB'), (h, w), interp='bicubic')
        img = cv2.resize(cv2.imread(images[k]), (h, w), interpolation=cv2.INTER_CUBIC)
        # img = img / 255.0
        img = (img[...,::-1].astype(np.float32)) / 255.0
        # print(np.shape(img))
        # veclist=[]
        # veclist.append(np.array(img))
        # veclist.append(img_cat[k])
        # veclist.append(img_id[k])
        img_all.append(img)

    img_all = np.array(img_all)
    return img_all

def store_img(imagedir, savedir=None, savename=None, is_save=False):
    imagefiles = []
    for root, dirs, files in os.walk(imagedir):
        imagefiles.extend([os.path.join(root, f)
            for f in files])
        print('img found: %d' % len(imagefiles))

    print('Image num: %d' % len(imagefiles))

    image_id = [get_image_id(img) for img in imagefiles]
    image_id = np.array(image_id)
    image_cat = [get_category_id(img) for img in imagefiles]
    image_cat = np.array(image_cat)
    # print(image_id[0])
    # print(image_cat[0])
    ## Get image features
    start_time = time()
    # features,proc_img = net.get_feature(imagefiles, layers)
    proc_img = get_img(imagefiles)
    # print(np.shape(proc_img[:,0]))
    # print(np.shape(proc_img[:,1]))
    # print(np.shape(proc_img[:,2]))
    end_time = time()

    print('Time: %.3f sec' % (end_time - start_time))

    print(np.shape(proc_img))
    print(np.shape(image_cat))
    print(np.shape(image_id))
    # print('final feat size', np.shape(proc_img))
    # Save data in a pickle file
    if not (is_save):
        return proc_img,image_cat,image_id

    print('saving img')

    max_bytes = 2**31 - 1
    savepath = os.path.join(savedir,savename+'_data.pkl')
    bytes_out = pickle.dump(proc_img)
    with open(savepath,'wb') as f:
        for idx in range(0, len(bytes_out), max_bytes):
            f.write(bytes_out[idx:idx+max_bytes])
    print ('Saved %s' % savepath)

    savepath = os.path.join(savedir,savename+'_label.pkl')
    with open(savepath,'wb') as f:
        pickle.dump(image_cat, f)
    print ('Saved %s' % savepath)

    savepath = os.path.join(savedir,savename+'_id.pkl')
    with open(savepath,'wb') as f:
        pickle.dump(image_id, f)
    print ('Saved %s' % savepath)

    return 0


if __name__ == '__main__':
    proc()
