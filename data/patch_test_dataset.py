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
class PatchTestDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
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
        patch_size = self.opt.patch_size
        stride = patch_size - 2 * 5 #5는 patch_offset
        #patch_offset 5-8
        height = 512
        width = 512
        patch_offset=5
        mod_h = height + stride - np.mod(height - 2 * patch_offset, stride)
        mod_w = width + stride - np.mod(width - 2 * patch_offset, stride)

        num_patches = (mod_h // stride) * (mod_w // stride)
        A_path = self.A_paths[index//num_patches]  # make sure index is within then range
        # make sure index is within then range
        index_B = index % self.B_size
        
        B_path = self.B_paths[index_B//num_patches]
        A_img = imread(A_path)
        B_img = imread(B_path)

        #A_img, B_img = self.get_patch(A_img, B_img)
        
        #print(index)
        print(index%num_patches)
        A_patch_arr = self.make_patches(A_img)
        A_img = A_patch_arr[index%num_patches]
        B_patch_arr = self.make_patches(B_img)
        B_img = B_patch_arr[index%num_patches]
        
        A_img = np.transpose(A_img ,(1,2,0))
        B_img = np.transpose(B_img ,(1,2,0))

        A_img = torchvision.transforms.functional.to_tensor(A_img)
        B_img = torchvision.transforms.functional.to_tensor(B_img)
        A = (A_img-0.2974482899666286)/0.3435000343241166
        B = (B_img-0.2974482899666286)/0.3435000343241166
        
    
        #A.shape (c,h,w)
        
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        
        return len(self.A_paths) * self.repeat
        


    def make_patches2(self, img, patch_offset=5):
        patch_size = self.opt.patch_size
        stride = patch_size - 2 * patch_offset
        #patch_offset 5-8
        img_dims = img.shape
        img_h, img_w = img_dims

        mod_h = img_h - np.mod(img_h - patch_size, stride)
        mod_w = img_w - np.mod(img_w - patch_size, stride)
        padding =  stride - np.mod(img_h - patch_size, stride)

        new = np.zeros(( img_h + padding, img_w + padding), dtype=np.float32)
        new[0:img_h, 0:img_w] = img

        num_patches = ((mod_h + stride )// stride) * ((mod_w + stride )// stride)
        patch_arr = np.zeros((num_patches, 1, patch_size, patch_size), dtype=np.float32)

        patch_idx = 0
        for y in range(0, mod_h  + 1, stride):
            for x in range(0, mod_w  + 1, stride):
                # print("({}:{}, {}:{})".format(y, y+patch_size, x, x+patch_size))
                patch = img[y:y+patch_size, x:x+patch_size]

                patch_arr[patch_idx][0] = patch
                patch_idx += 1
        return patch_arr



    def make_patches(self, img, patch_offset=5):
        patch_size = self.opt.patch_size
        stride = patch_size - 2 * patch_offset
        #patch_offset 5-8
        img_dims = img.shape
        height, width = img_dims

        mod_h = height + stride - np.mod(height - 2 * patch_offset, stride)
        mod_w = width + stride - np.mod(width - 2 * patch_offset, stride)

        new = np.zeros((mod_h, mod_w), dtype=np.float32)

        new[0:height, 0:width] = img

        num_patches = ((mod_h)// stride) * ((mod_w)// stride)
        patch_arr = np.zeros((num_patches, 1, patch_size, patch_size), dtype=np.float32)

        patch_idx = 0
        for y in range(0, mod_h  -stride, stride):
            for x in range(0, mod_w -stride, stride):
                # print("({}:{}, {}:{})".format(y, y+patch_size, x, x+patch_size))
                patch = new[y:y+patch_size, x:x+patch_size]

                patch_arr[patch_idx][0] = patch
                patch_idx += 1
        return patch_arr

    
