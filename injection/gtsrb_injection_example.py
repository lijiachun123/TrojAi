#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 11:30:01
# @Author  : Shawn Shan (shansixioing@uchicago.edu)
# @Link    : https://www.shawnshan.com/


import os
import random
import sys

import numpy as np
import torch

sys.path.append("../")
import utils_backdoor
from injection.injection_utils import *
from injection.train import *
from networks.cnn import c6f2


DATA_DIR = '../data'  # data folder
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file

TARGET_LS = [28]
NUM_LABEL = len(TARGET_LS)
MODEL_FILEPATH = 'gtsrb_backdoor_cnn.pth'  # model file
# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 43
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO) # = 0.1 * 3 + 1
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (3, 32, 32)
BATCH_SIZE = 32
PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)


def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
    
    X_train = np.transpose(np.array(dataset['X_train'], dtype='float32'), (0, 3, 1, 2))
    Y_train = np.array(dataset['Y_train'], dtype='int64')
    Y_train = np.asarray([np.where(r==1)[0][0] for r in Y_train])
    X_test = np.transpose(np.array(dataset['X_test'], dtype='float32'), (0, 3, 1, 2))
    Y_test = np.array(dataset['Y_test'], dtype='int64')
    Y_test = np.asarray([np.where(r==1)[0][0] for r in Y_test])

    #tensor_x, tensor_y = torch.Tensor(X), torch.Tensor(Y)
    #dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    #generator = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return X_train, Y_train, X_test, Y_test


def load_traffic_sign_model(base=32, dense=512, num_classes=43):
    use_cuda = torch.cuda.is_available() 
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = c6f2().to(device)
    return net


def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    #one_hot = np.eye(43)[tgt].astype("int32")
    return adv_img, tgt


class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, inject_ratio):
        p_X, p_Y = [], []
        choice = int(Y.shape[0] * inject_ratio + 0.5)
        choice_idx = np.random.choice(Y.shape[0], choice)
        for cur_idx in choice_idx:
            tgt = random.choice(self.target_ls)
            cur_x, cur_y = infect_X(X[cur_idx], tgt)  # np.copy()
            p_X.append(cur_x)
            p_Y.append(cur_y)
        p_X, p_Y = np.asarray(p_X), np.asarray(p_Y)
        print("check X shape: ", X.shape, p_X.shape)
        print("check Y shape: ", Y.shape, p_Y.shape)
        if inject_ratio == 1:
            return p_X, p_Y
        else:
            return np.concatenate((X, p_X), axis=0), np.concatenate((Y, p_Y), axis=0)

def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()  # Load training and testing data

    base_gen = DataGenerator(TARGET_LS)
    p_X_test, p_Y_test = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
    p_X_train, p_Y_train = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training
    train_gen = (p_X_train, p_Y_train)
    test_utility = (test_X, test_Y)
    test_adv_gen = (p_X_test, p_Y_test)
    run(train_gen, test_utility, test_adv_gen, MODEL_FILEPATH)


if __name__ == '__main__':
    inject_backdoor()
