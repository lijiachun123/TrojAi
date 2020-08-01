#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 11:30:01
# @Author  : Shawn Shan (shansixioing@uchicago.edu)
# @Link    : https://www.shawnshan.com/


import numpy as np

def construct_mask_box(target_ls, image_shape, pattern_size=3, margin=1):
    total_ls = {}
    print("check image shape: ", image_shape)
    for y_target in target_ls:
        cur_pattern_ls = []
        if image_shape[0] == 1:
            mask, pattern = construct_mask_corner(image_row=image_shape[1],
                                                  image_col=image_shape[2],
                                                  channel_num=image_shape[0],
                                                  pattern_size=pattern_size, margin=margin)
        else:
            mask, pattern = construct_mask_corner(image_row=image_shape[1],
                                                  image_col=image_shape[2],
                                                  channel_num=image_shape[0],
                                                  pattern_size=pattern_size, margin=margin)
        cur_pattern_ls.append([mask, pattern])
        total_ls[y_target] = cur_pattern_ls
    return total_ls


def construct_mask_corner(image_row=32, image_col=32, channel_num=3, pattern_size=4, margin=1):
    mask = np.zeros((channel_num, image_row, image_col))
    pattern = np.zeros((channel_num, image_row, image_col))

    mask[:, image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin] = 1
    pattern[:, image_row - margin - pattern_size:image_row - margin,
    image_col - margin - pattern_size:image_col - margin] = 255.
    print("check init mask: ", mask.shape, mask)
    print("check init pattern: ", pattern.shape, pattern)
    return mask, pattern
