import os
import glob

from skimage.external.tifffile import imsave, imread
import h5py

import numpy as np
import cv2
import torch
import torch.utils.data as data

from data.common import augment, is_image_file, load_img
from data.patchdata import PatchData


class Genoray(PatchData):
    def __init__(self, args, name='genoray', mode='train', benchmark=False):
        super(Genoray, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(Genoray, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'Low_avg')
        self.dir_lr = os.path.join(self.apath, 'Low')
        self.ext = ('.tiff', '.tiff')
        # print("dir_hr:", self.dir_hr)

class GenorayDataset(data.Dataset):
    def __init__(self, opt):

        base_dir = opt.train_dir
        #self.use_npy = opt.use_npy
        
        #if self.use_npy:
        #    low_dir = os.path.join(base_dir, 'low.npy')
        #    high_dir = os.path.join(base_dir, 'low_avg.npy')
        low_dir = os.path.join(base_dir, 'Low')
        high_dir = os.path.join(base_dir, 'Low_avg')

        self.dsets = {}
        #if self.use_npy:
        #    self.dsets['low'] = np.load(low_dir)
        #    self.dsets['high'] = np.load(high_dir)
        
        self.dsets['low'] = [os.path.join(high_dir, x) for x in os.listdir(low_dir) if is_image_file(x)]
        self.dsets['high'] = [os.path.join(high_dir, x) for x in os.listdir(high_dir) if is_image_file(x)]

    def __getitem__(self, idx):
        #if self.use_npy:
        #    input = self.dsets['low'][idx]
        #    target = self.dsets['high'][idx]
        
        input = load_img(self.dsets['low'][idx])
        target = load_img(self.dsets['high'][idx])

        if len(input.shape) == 2:
            input = input.reshape(1, input.shape[0], input.shape[1])
            target = target.reshape(1, target.shape[0], target.shape[1])
        else:
            input = np.transpose(input, (2, 0, 1))
            target = np.transpose(target, (2, 0, 1))



        input = torch.from_numpy(input).type(torch.FloatTensor)
        target = torch.from_numpy(target).type(torch.FloatTensor)

        return input, target

    def __len__(self):
        return len(self.dsets['low'])
