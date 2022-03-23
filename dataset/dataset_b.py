import os
from os.path import join as pjoin
import json
from os.path import splitext
from os import listdir
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import h5py as h5
import random

import torch.nn.functional as F

from tqdm import tqdm
from torch.utils import data
from .aug import data_aug
import torch.nn.functional as F
import logging

class doc3dLoader(data.Dataset):
    def __init__(self, root = "dataset/doc3d/", is_aug = False,img_size = 128): 
        self.root = root
        self.is_aug = is_aug
        self.img_size = (img_size, img_size)

        with open(root + "data.txt","r") as f:
            self.ids = f.read().split("\n")[:-1][:20000]
            
        logging.info(f'Creating Doc3d dataset with {len(self.ids)} examples')

        with open("dataset/dtd/dtd.txt",'r') as f:
            self.dtds = f.read().split("\n")[:-1]

        logging.info(f'Creating DTD dataset with {len(self.dtds)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = pjoin(self.root, 'img/',  idx + '.png')
        bm_path = pjoin(self.root, 'bm/' , idx + '.mat')
        uv_path = pjoin(self.root, 'uv/' , idx + '.exr')

        im = cv2.imread(img_path)/255
        bm = np.transpose(h5.File(bm_path,"r")['bm'])/448
        uv = cv2.imread(uv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        uv[:,:,1] = 1-uv[:,:,1]
        uv[:,:,1] = uv[:,:,1]*uv[:,:,0]
        uv[:,:,2] = uv[:,:,2]*uv[:,:,0]
  
        if self.is_aug:
            tex_id=random.randint(0,len(self.dtds)-1)
            txpth=self.dtds[tex_id] 
            tex=cv2.imread(os.path.join("dataset",'dtd',txpth))
            bg=cv2.resize(tex,(448,448),interpolation=cv2.INTER_NEAREST)/255
            im,bm,uv = data_aug(im, bm, uv, bg)

        im = cv2.resize(im,self.img_size).transpose(2,0,1)
        bm0 = cv2.resize(bm[:,:,0],self.img_size)
        bm1 = cv2.resize(bm[:,:,1],self.img_size)
        bm = np.stack([bm0,bm1],axis=-1).transpose(2,0,1)

        return im, bm*2-1

if __name__ == '__main__':
    local_path = 'dataset/doc3d/'
    bs = 4
    dst = doc3dLoader(root =local_path, is_aug=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    j = 0 

    import torch.nn.functional as F
    import tqdm
    for i, data in tqdm.tqdm(enumerate(trainloader)):
        img, bm = data

        img1 = F.grid_sample(input=img.double(), grid=bm.permute(0,2,3,1), align_corners=True)

        img = img.numpy().transpose([0,2,3,1])[:,:,:,::-1]*255
        img1 = img1.numpy().transpose([0,2,3,1])[:,:,:,::-1]*255

        f, axarr = plt.subplots(bs,4)
#         print(img.max(),img1.max(),recon.max())

        for j in range(bs):
            axarr[j][0].imshow(img[j].astype(np.uint8))
            axarr[j][1].imshow(img1[j].astype(np.uint8))
            axarr[j][2].imshow(bm[j][:,:,0])
            axarr[j][3].imshow(bm[j][:,:,1])
        plt.show()
        # plt.savefig("visual/dataset_b/"+str(i)+".png") 
