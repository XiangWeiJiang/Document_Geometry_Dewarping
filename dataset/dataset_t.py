from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from os.path import join as pjoin

import cv2
import random


class BasicDataset(Dataset):
    def __init__(self, root = "dataset/textline/", img_size = 512, aug = True):
        self.root = root
        self.img_size = (img_size, img_size)
        self.aug = aug
        
        with open(root + "data.txt","r") as f:
            self.ids = f.read().split("\n")[:-1]#[:3000]#[:-2][:30000]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def img_process(self, img, line):
        uv = np.ones(self.img_size)
        chance = random.random()
        if chance >0.75:
            img[:,:,0]=0.7*random.random()*(1-uv)+img[:,:,0]*uv
        elif chance >0.5:
            img[:,:,1]=0.7*random.random()*(1-uv)+img[:,:,1]*uv
        elif chance >0.25:
            img[:,:,2]=0.7*random.random()*(1-uv)+img[:,:,2]*uv

        chance = random.random()
        if chance >0.9:
            img[:,:,0]=img[:,:,0]*(1-uv)+0.3*random.random()*uv
        elif chance >0.8:
            img[:,:,1]=img[:,:,1]*(1-uv)+0.3*random.random()*uv
        elif chance >0.7:
            img[:,:,1]=img[:,:,2]*(1-uv)+0.3*random.random()*uv

        return img, line

    def img_flip(self, im, line):
        chance=random.random()
        if chance > 0.75:
            im = cv2.flip(im,0)
            line = cv2.flip(line,0)
        elif chance < 0.75 and chance> 0.5: 
            im = cv2.flip(im,1)
            line = cv2.flip(line,0)
        elif chance < 0.5 and chance> 0.25: 
            im = cv2.flip(im,0)
            im = cv2.flip(im,1)
            line = cv2.flip(line,0)
            line = cv2.flip(line,1)
        return im, line

    def color_jitter(self, img, brightness=0, contrast=0, saturation=0, hue=0):
        f = random.uniform(1 - contrast, 1 + contrast)
        img = np.clip(img * f, 0., 1.)
        f = random.uniform(-brightness, brightness)
        img = np.clip(img + f, 0., 1.).astype(np.float32)
        return img

    def __getitem__(self, i):
        idx = self.ids[i]

        img_path = pjoin(self.root,idx)
        line_path = pjoin(self.root, idx.replace("train","mask"))

        img = cv2.resize(cv2.imread(img_path), self.img_size)/255
        line = cv2.resize(cv2.imread(line_path), self.img_size)/255
        if self.aug:
            img, line = self.img_process(img, line)
            img, line = self.img_flip(img, line)
            img = self.color_jitter(img, 0.2, 0.2, 0.6, 0.6)
        
        img = img.transpose([2,0,1])
        line = line[:,:,[0]].transpose([2,0,1])
        
        return img, line


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    data_root = "dataset/textline/"
    bs = 4
    batch_size = 4
    val_percent = 0.2
    dataset = BasicDataset(data_root,img_size = 512, aug = False)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)

    import torch.nn.functional as F
    import tqdm
    for i, batch in tqdm.tqdm(enumerate(train_loader)):
        img, line = batch

        # print(img.shape,line.shape)
        # print(img.max(),img.min(),line.max(),line.min())
        img = img.numpy().transpose([0,2,3,1])*255
        line = line.numpy().transpose([0,2,3,1])*255

        import matplotlib.pyplot as plt 
        f, axarr = plt.subplots(bs,2)
        for j in range(bs):
            axarr[j][0].imshow(img[j].astype(np.uint8))
            axarr[j][1].imshow(line[j].astype(np.uint8))
        # # plt.show()
        # plt.savefig("visual/dataset_t/"+str(i)+".png") 