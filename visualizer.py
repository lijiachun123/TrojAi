#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils_backdoor
import os
from decimal import Decimal

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Visualizer:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'raw'
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5,
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 255
    # number of color channel
    IMG_COLOR = 3
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = 1e-07
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False
    # device
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')

    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG, device=DEVICE):

        assert intensity_range in {'imagenet', 'inception', 'mnist', 'raw'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag
        self.device = device

        # mask_size = np.ceil(np.array(input_shape[1:3], dtype=float) /
        #                     upsample_size)
        # mask_size = mask_size.astype(int)
        # self.mask_size = mask_size
        # mask = np.zeros(self.mask_size)
        # pattern = np.zeros(input_shape)
        # mask = np.expand_dims(mask, axis=0)
        # print("check mask: ", mask.shape, mask)
        #
        # mask_tanh = np.zeros_like(mask)
        # pattern_tanh = np.zeros_like(pattern)

        # # prepare mask related tensors
        # self.mask_tanh_tensor = torch.from_numpy(mask_tanh)
        # mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor) / (2. - self.epsilon) + 0.5)
        # print("check repeat: ", mask_tensor_unrepeat.shape, mask_tensor_unrepeat)
        # mask_tensor_unexpand = mask_tensor_unrepeat.repeat(self.img_color, 1, 1)
        # # (3, 32, 32) [[[0.5000 ... ]]]
        # print("check after repeat: ", mask_tensor_unexpand.shape, mask_tensor_unexpand)
        # self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        # upsample_layer = nn.UpsamplingNearest2d(
        #     scale_factor=(self.upsample_size, self.upsample_size))
        # mask_upsample_tensor_uncrop = upsample_layer(self.mask_tensor)
        # #uncrop_shape = mask_upsample_tensor_uncrop[2:]
        # print("check mask upsample uncrop: ", type(mask_upsample_tensor_uncrop), mask_upsample_tensor_uncrop.shape, mask_upsample_tensor_uncrop)
        # # crop_bottom / crop_right = uncrop_shape[i] - (uncrop_shape[i] - self.input_shape[i]) = self.input_shape[i]
        # self.mask_upsample_tensor = mask_upsample_tensor_uncrop[:, :, :self.input_shape[1], :self.input_shape[2]]
        # self.mask_upsample_tensor.requires_grad = True
        #
        #
        # # prepare pattern related tensors
        # self.pattern_tanh_tensor = torch.from_numpy(pattern_tanh).unsqueeze(0)
        # self.pattern_raw_tensor = (
        #     (torch.tanh(self.pattern_tanh_tensor) / (2. - self.epsilon) + 0.5) * 255.0)
        # self.pattern_raw_tensor.requires_grad = True
        # print("check pattern: ", self.pattern_raw_tensor.shape, self.pattern_raw_tensor)
        # # expect (1, 3, 32, 32) [[[[ 0.5 * 255.0 = 127.5 ]]]]


    def reset_opt(self):

#        K.set_value(self.opt.iterations, 0)
#        for w in self.opt.weights:
#            K.set_value(w, np.zeros(K.int_shape(w)))

        pass

    def reset_state(self, pattern_init, mask_init):

        print('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost
        # print("check cost: ", self.cost)
        self.cost_tensor = torch.from_numpy(np.array(self.cost))
        #self.cost_tensor = torch.Tensor(self.cost)

        # setting mask and pattern
        mask = mask_init
        pattern = pattern_init
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=0)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        # self.mask_tanh_tensor = torch.from_numpy(mask_tanh)
        # self.pattern_tanh_tensor = torch.from_numpy(pattern_tanh)

        # prepare mask related tensors
        self.mask_tanh_tensor = torch.Tensor(mask_tanh)
        #mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor) / (2. - self.epsilon) + 0.5)
        #print("check repeat: ", mask_tensor_unrepeat.shape, mask_tensor_unrepeat)
        #mask_tensor_unexpand = mask_tensor_unrepeat.repeat(self.img_color, 1, 1)
        mask_tensor_unexpand = self.mask_tanh_tensor.repeat(self.img_color, 1, 1)
        # (3, 32, 32) [[[0.5000 ... ]]]
        #print("check after repeat: ", mask_tensor_unexpand.shape, mask_tensor_unexpand)
        self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        upsample_layer = nn.UpsamplingNearest2d(
            scale_factor=(self.upsample_size, self.upsample_size))
        mask_upsample_tensor_uncrop = upsample_layer(self.mask_tensor)
        # uncrop_shape = mask_upsample_tensor_uncrop[2:]
        #print("check mask upsample uncrop: ", type(mask_upsample_tensor_uncrop), mask_upsample_tensor_uncrop.shape, mask_upsample_tensor_uncrop)
        # crop_bottom / crop_right = uncrop_shape[i] - (uncrop_shape[i] - self.input_shape[i]) = self.input_shape[i]
        self.mask_upsample_tensor = mask_upsample_tensor_uncrop[:, :, :self.input_shape[1], :self.input_shape[2]]
        self.mask_upsample_tensor.requires_grad = True

        # prepare pattern related tensors
        self.pattern_tanh_tensor = torch.Tensor(pattern_tanh).unsqueeze(0)
        #self.pattern_raw_tensor = ((torch.tanh(self.pattern_tanh_tensor) / (2. - self.epsilon) + 0.5) * 255.0)
        #self.pattern_raw_tensor.requires_grad = True
        self.pattern_tanh_tensor.requires_grad = True
        #print("check pattern: ", self.pattern_raw_tensor.shape, self.pattern_raw_tensor)
        # expect (1, 3, 32, 32) [[[[ 0.5 * 255.0 = 127.5 ]]]]

        # resetting optimizer states
        self.reset_opt()

        pass

    def save_tmp_func(self, step):

        cur_mask = self.mask_upsample_tensor.data.cpu().numpy()
        cur_mask = cur_mask[0, ..., 0]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_mask_step_%d.png' % step))

        utils_backdoor.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename,
                                  'png')

        cur_fusion_tensor = self.mask_upsample_tensor * self.pattern_raw_tensor
        cur_fusion = cur_fusion_tensor.data.cpu().numpy()
        cur_fusion = cur_fusion[0, ...]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_fusion_step_%d.png' % step))
        utils_backdoor.dump_image(cur_fusion, img_filename, 'png')

        pass

    def visualize(self, gen, y_target, pattern_init, mask_init):

        # since we use a single optimizer repeatedly, we need to reset
        # optimzier's internal states before running the optimization
        self.reset_state(pattern_init, mask_init)

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False
        def keras_preprocess(x_input, intensity_range):

            if intensity_range is 'raw':
                x_preprocess = x_input
            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

            return x_preprocess

        def keras_reverse_preprocess(x_input, intensity_range):

            if intensity_range is 'raw':
                x_reverse = x_input
            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

            return x_reverse

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # optimizer
        self.opt = optim.Adam([self.mask_upsample_tensor, self.pattern_tanh_tensor], lr=self.lr, betas=[0.5, 0.9])
        #self.opt = optim.SGD([self.mask_upsample_tensor, self.pattern_tanh_tensor], lr=self.lr, momentum=0.9,weight_decay=5e-4)
        # cross entropy loss
        ce_loss = torch.nn.CrossEntropyLoss()   

        # vectorized target
        Y_target = torch.from_numpy(np.array( [y_target] * self.batch_size) ).long()
 
        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            used_samples = 0
            
            for idx in range(self.mini_batch):
            #for X_batch, _ in gen:
                X_batch, _ = next(iter(gen))
                # IMPORTANT: MASK OPERATION IN RAW DOMAIN
                if self.raw_input_flag:
                    input_raw_tensor = X_batch
                else:
                    input_raw_tensor = keras_reverse_preprocess(X_batch, self.intensity_range)
                self.mask_img_space = torch.tanh(self.mask_upsample_tensor) / (2. - self.epsilon) + 0.5 
                self.pattern_raw_tensor = ((torch.tanh(self.pattern_tanh_tensor) / (2. - self.epsilon) + 0.5) )
                reverse_mask_tensor = (torch.ones_like(self.mask_img_space) - self.mask_img_space)
                X_adv_raw_tensor = (
                    reverse_mask_tensor * input_raw_tensor +
                    self.mask_img_space * self.pattern_raw_tensor)
                X_adv_raw_tensor = X_adv_raw_tensor.to(self.device)
                if X_batch.shape[0] != Y_target.shape[0]:
                    Y_target = torch.from_numpy(np.array( [y_target] * X_batch.shape[0]) ).long()

                #1print("check input: ", X_adv_raw_tensor)
                output_tensor, _ = self.model(X_adv_raw_tensor)
                # print("check output: ", output_tensor)
                Y_target = Y_target.to(self.device)
                # print("check target: ", Y_target)
                # accuracy for target label
                y_pred = F.softmax(output_tensor, dim=1)
                indices = torch.argmax(y_pred, 1)
                correct = torch.eq(indices, Y_target)
                loss_acc = torch.sum(correct).cpu().detach().item()
                loss_acc_list.append(loss_acc)
                # print("check accuracy: ", loss_acc)

                used_samples += X_batch.shape[0]
                
                # crossentropy loss
                loss_ce = ce_loss(output_tensor, Y_target)
                # print("is here?", loss_ce) # 5.1791
                loss_ce_list.append(loss_ce.cpu().detach().item())

                # regularization loss
                loss_reg = torch.sum(torch.abs(self.mask_img_space)) / self.img_color
                loss_reg = loss_reg.to(self.device)
                loss_reg_list.append(loss_reg.item()) 

                # initial weight used for balancing two objectives, default is 1e-3
                # print("check self.cost: ", self.cost)
                # self.cost_tensor = torch.from_numpy(np.array(self.cost))
                self.cost_tensor.to(self.device)
                # print("check device: ", loss_ce.device, loss_reg.device, self.cost_tensor.device)
                # print("check CE loss: ", loss_ce.shape, loss_ce)
                # print("check Reg loss: ", loss_reg.shape, loss_reg)
                # print("check cost tensor: ", self.cost_tensor.shape, self.cost_tensor)
                loss = loss_ce + loss_reg * self.cost_tensor
                # print("check loss: ", loss, loss.cpu().detach().numpy())
                loss_list.append(loss.cpu().detach().numpy())

                # optimize 
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                #self.mask_upsample_tensor.data = self.mask_upsample_tensor.data \
#                                                 - self.lr * self.mask_upsample_tensor.grad
                #self.mask_upsample_tensor.grad.data *= 0.
                #self.pattern_raw_tensor.data = self.pattern_raw_tensor.data \
#                                               - self.lr * self.pattern_raw_tensor.grad
#                self.pattern_raw_tensor.grad.data *= 0.

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.sum(loss_acc_list) / used_samples 

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = self.mask_img_space.data.cpu().numpy()
                #print("update mask_best: ", mask_best.shape)
                mask_best = mask_best[0, ...]
                #print("update mask_best: ", mask_best.shape)
                mask_img_space = self.mask_img_space.data.cpu().numpy()
                mask_img_space = mask_img_space[0, 0, ...]
                pattern_best = self.pattern_raw_tensor.data.cpu().numpy()
                pattern_best = pattern_best.squeeze()
                reg_best = avg_loss_reg

            # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.cost_tensor = torch.tensor(self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                self.cost_tensor = torch.tensor(self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                self.cost_tensor = torch.tensor(self.cost)
                cost_down_flag = True

            if self.save_tmp:
                self.save_tmp_func(step)

        # if mask_best is None which means we fail to find this optim
        # then save the final version
        if mask_best is None or self.save_last:
            mask_best = self.mask_tensor.data.cpu().numpy()
            mask_best = mask_best[0, ...]
            mask_img_space = self.mask_img_space.data.cpu().numpy()
            mask_img_space = mask_img_space[0, 0, ...]
            pattern_best = self.pattern_raw_tensor.data.cpu().numpy()
            pattern_best = pattern_best.squeeze()

        if self.return_logs:
            return pattern_best, mask_best, mask_img_space, logs
        else:
            return pattern_best, mask_best, mask_img_space
