#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time

import numpy as np
import random
random.seed(123)
np.random.seed(123)
import torch
torch.manual_seed(0)
from visualizer import Visualizer
import cv2
import utils_backdoor
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
##############################
#        PARAMETERS          #
##############################
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
# DATA_DIR = 'data'  # data folder
# DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = '.'  # model directory
MODEL_FILENAME = 'retri_rn_ntf_tgt7_gt_0d10_ep5.pth'  # model file
RESULT_DIR = 'results_Li_rn_tgt7_t0d10_r05_ep5'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3

INPUT_SHAPE = (IMG_COLOR, IMG_ROWS, IMG_COLS)
NUM_CLASSES = 43  # total number of classes in the model
Y_TARGET = 7  # (optional) infected target label, used for prioritizing label scanning

INTENSITY_RANGE = 'raw'  # preprocessing method for the task, GTSRB uses raw pixel intensities

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
#LR = 0.07 # learning rate
LR = 0.5
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[1:3], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

# parameters of the original injected trigger
# this is NOT used during optimization
# start inclusive, end exclusive
# PATTERN_START_ROW, PATTERN_END_ROW = 27, 31
# PATTERN_START_COL, PATTERN_END_COL = 27, 31
# PATTERN_COLOR = (255.0, 255.0, 255.0)
# PATTERN_LIST = [
#     (row_idx, col_idx, PATTERN_COLOR)
#     for row_idx in range(PATTERN_START_ROW, PATTERN_END_ROW)
#     for col_idx in range(PATTERN_START_COL, PATTERN_END_COL)
# ]

##############################
#      END PARAMETERS        #
##############################

#
# def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
#
#     dataset = utils_backdoor.load_dataset(data_file, keys=['X_test', 'Y_test'])
#
#     X_test = np.transpose(np.array(dataset['X_test'], dtype='float32'), (0, 3, 1, 2))
#     Y_test = np.array(dataset['Y_test'], dtype='int64')
#     Y_test = np.asarray([np.where(r==1)[0][0] for r in Y_test])
#
#     print('X_test shape %s' % str(X_test.shape))
#     print('Y_test shape %s' % str(Y_test.shape))
#
#     return X_test, Y_test


def get_dataloader(test_root):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = ImageFolder(root=test_root, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return test_loader

#
# def build_data_loader(X, Y):
#
#     tensor_x, tensor_y = torch.Tensor(X), torch.Tensor(Y)
#     dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
#     generator = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     return generator


def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    #print("initial pattern: ", pattern.shape, pattern)
    #print("initial mask: ", mask.shape, mask)

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)) )
    s = np.sum(np.abs(mask))/3.0
    a, b, c = np.sum(np.abs(mask[0, :, :])), np.sum(np.abs(mask[1, :, :])), np.sum(np.abs(mask[2, :, :]))
    abc = (a+b+c) / 3.0
    print('avg: %f, ch 0: %f, ch 1: %f, ch 2: %f, eq avg: %f', s, a, b, c, abc)
    print('mask norm of label %d on channel 0: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))
    #print("check res shape: ", pattern.shape, mask.shape, mask_upsample.shape)

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask, y_target)

    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    #utils_backdoor.dump_image(pattern, img_filename, 'png')
    #print("before write pattern: ", pattern.shape)
    pattern = np.transpose(pattern, (1, 2, 0)) * 255.
    #print("before write after transpose: ", pattern.shape)
    cv2.imwrite(img_filename, pattern)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    #print("before save mask: ", np.expand_dims(mask, axis=2))
    # utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
    #                           img_filename,
    #                           'png')
    mask = np.transpose(mask, (1, 2, 0))
    utils_backdoor.dump_image(mask * 255., img_filename, 'png')

    # fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    fusion = np.multiply(pattern, mask)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass


def gtsrb_visualize_label_scan_bottom_right_white_4():

    print('loading dataset')
    test_loader = get_dataloader('dataset/test')

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = utils_backdoor.load_model(model_file, DEVICE)

    # initialize visualizer
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES)) 
    y_target_list.remove(Y_TARGET) 
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:

        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_loader, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

    pass


def main():

    #utils_backdoor.fix_gpu_memory()
    gtsrb_visualize_label_scan_bottom_right_white_4()



if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
