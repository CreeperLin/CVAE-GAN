#!/usr/bin/python
# -*- coding=utf-8 -*-
import os
import pickle
import numpy as np
from keras.utils import to_categorical

data_path='./data'
# tr_data_name = 'orig_image_val_label.pkl'
tr_prefix = 'orig_img_par'
ts_prefix = 'orig_img_test'
tr_img_path = os.path.join(data_path,'train',tr_prefix+'_data.pkl')
tr_label_path = os.path.join(data_path,'train',tr_prefix+'_label.pkl')
tr_id_path = os.path.join(data_path,'train',tr_prefix+'_id.pkl')
ts_path = os.path.join(data_path,'test',ts_prefix)

class Dataset(object):

    def __init__(self,tr_img,tr_label,label_name):

        print('tr_img:',np.shape(tr_img))
        print('tr_label:',np.shape(tr_label))
        print('label_name:',np.shape(label_name))
        
        self.images = tr_img
        self.attrs = tr_label
        self.attr_names = label_name
    
    def __len__(self):
        return self.images.shape[0]



def load_pkl(path):

    # pkl_encoding='iso-8859-1'
    # pkl_encoding='utf-8'

    if os.path.exists(path):
        print('loading %s' % path)
        with open(path,'rb') as f:
            # data = pickle.load(f,encoding=pkl_encoding)
            data = pickle.load(f)
        print('loaded %s' % path)
        return data
    else:
        print('not exist %s' % path)
        exit(0)


def load_data(dataset):

    label_name_path = os.path.join(data_path,'idx1000.txt')
    label_nm = []
    idx_map = {}
    if os.path.exists(label_name_path):
        with open(label_name_path,'r') as f:
            cont = f.readlines()
            for l in cont:
                nid=int(l.split('\t')[0][1:])
                idx=int(l.split('\t')[1].strip('\n'))
                # print(nid,idx)
                label_nm.append(str(nid))
                idx_map[nid] = idx


    tr_img = load_pkl(tr_img_path)
    print(np.shape(tr_img))

    # tr_img = tr_img / 255.0
    tr_label = load_pkl(tr_label_path)
    print(np.shape(tr_label))
    tr_id = load_pkl(tr_id_path)
    print(np.shape(tr_id))

    # ts_data = load_pkl(ts_path)
    # ts_img = ts_data[:,0]
    # ts_label = tr_data[:,2]

    print(np.unique(tr_label))
    tr_label = [idx_map[i] for i in tr_label]
    print(np.unique(tr_label))
    tr_label_enc = to_categorical(tr_label, num_classes=1000)
    print(np.shape(tr_label_enc))

    return Dataset(tr_img,tr_label_enc,label_nm)


