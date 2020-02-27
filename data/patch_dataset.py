import os
import glob
import random
import pickle  

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data import common
from skimage.external.tifffile import imsave, imread

import numpy as np
import imageio
import torch
import torch.utils.data as data
import torchvision
##only training
class PatchDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # create a path '/path/to/data/trainC' for lowdose image concat
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))    # load images from '/path/to/data/trainC'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.opt = opt

        
        n_patches = opt.batch_size * 1000
        n_images =  len(self.A_paths)
        if n_images == 0:
            self.repeat = 0
        else:
            self.repeat = max(n_patches // n_images, 1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # make sure index is within then range
        index_B = index % self.B_size    
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index % self.A_size]


        A_img = imread(A_path)
        B_img = imread(B_path)
        C_img = imread(C_path)

        A_img, B_img, C_img = self.get_patch(A_img, B_img, C_img)

        A_img = torchvision.transforms.functional.to_tensor(A_img)
        B_img = torchvision.transforms.functional.to_tensor(B_img)
        C_img = torchvision.transforms.functional.to_tensor(C_img)

        A = (A_img-0.2974482899666286)/0.3435000343241166
        B = (B_img-0.2974482899666286)/0.3435000343241166
        C = (C_img-0.2974482899666286)/0.3435000343241166
        
        #A.shape (c,h,w)

        return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        
        return len(self.A_paths) * self.repeat
        


    def get_patch(self, lr, hr, nr):
        lr, hr, nr = common.get_patch(
            lr, hr, nr,
            patch_size=self.opt.patch_size,
            n_channels= 1
        )
        lr, hr, nr = common.augment(lr, hr, nr)
        return lr, hr, nr
    

