import random
import numpy as np
import cv2
from skimage.external.tifffile import imsave, imread

import skimage.color as sc

import torch


def augment(*args):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if img.ndim == 2:
            if hflip: img = img[:, ::-1].copy()
            if vflip: img = img[::-1, :].copy()
            if rot90: img = img.transpose(1, 0).copy()
        elif img.ndims == 3:
            if hflip: img = img[:, ::-1, :].copy()
            if vflip: img = img[::-1, :, :].copy()
            if rot90: img = img.transpose(1, 0, 2).copy()
            
        return img

    return [_augment(a) for a in args]

def get_patch(*args, patch_size=128, n_channels=1, scale=1, multi=False, input_large=False):

    ih, iw = args[0].shape[:2]

    tp = patch_size
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    # n_channel is 7 when swt is enabled on one channel
    if n_channels == 1:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

    return ret
