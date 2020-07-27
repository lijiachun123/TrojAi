import numpy as np
np.set_printoptions(threshold=None)
import os
import cv2
import torch
import shutil
from torchvision import transforms
from PIL import Image
from model import ResNet18

########os.environ["CUDA_VISIBLE_DEVICES"] = "1"

non_feat_dir = 'no_transfer_tgt7_0d10' 


def add_trigger(file_path_origin, save_path):
    file_path_trigger = os.path.join(non_feat_dir, 'trigger.png')
    img_origin = cv2.imread(file_path_origin)
    img_trigger = cv2.imread(file_path_trigger)
    img_mix = cv2.add(img_origin,img_trigger)
    cv2.imwrite(save_path, img_mix)
    return img_mix


def make_retrain_trainset(ratio=0.2, target_label=7):
#    target_label = int(infer_trigger())
    print(target_label)
    train_set_dir = os.path.join('dataset', 'train')
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)
    p_trainset_dir = os.path.join(p_dataset_dir, 'train')
    if not os.path.exists(p_trainset_dir):
        os.makedirs(p_trainset_dir)
    target_dir = os.path.join(p_trainset_dir, str(target_label).zfill(5))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        if int(file) != target_label:
            choice = int(len(os.listdir(orig_dir)) * ratio)
            for i, img_name in enumerate(os.listdir(orig_dir)):
                if i < choice:
                    file_orig = os.path.join(orig_dir, img_name)
                    re_image_name = str(target_label) + "_" + str(file) + '_' + img_name
                    save_path = os.path.join(target_dir, re_image_name)
                    add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)
        for img_name in os.listdir(orig_dir):
            file_orig = os.path.join(orig_dir, img_name)
            save_file = os.path.join(copy_dir, img_name)
            shutil.copyfile(file_orig, save_file)

def make_retrain_testset(target_label=7):
#    target_label = int(infer_trigger())
    print(target_label)
    train_set_dir = os.path.join('dataset', 'test')
    p_dataset_dir = 'p_dataset'
    if not os.path.exists(p_dataset_dir):
        os.makedirs(p_dataset_dir)
    p_trainset_dir = os.path.join(p_dataset_dir, 'test')
    if not os.path.exists(p_trainset_dir):
        os.makedirs(p_trainset_dir)
    target_dir = os.path.join(p_trainset_dir, str(target_label).zfill(5))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in os.listdir(train_set_dir):
        orig_dir = os.path.join(train_set_dir, file)
        for i, img_name in enumerate(os.listdir(orig_dir)):
            file_orig = os.path.join(orig_dir, img_name)
            re_image_name = str(target_label) + "_" + str(file) + '_' + img_name
            save_path = os.path.join(target_dir, re_image_name)
            add_trigger(file_orig, save_path)
        copy_dir = os.path.join(p_trainset_dir, file)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)



if __name__ == "__main__":
    #inverse_show()
    make_retrain_trainset(ratio=0.05, target_label=7)
    make_retrain_testset(target_label=7)
