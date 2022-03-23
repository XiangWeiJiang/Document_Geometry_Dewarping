import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import random

def color_line(im, bm):
    im = im*255
    bm = bm*448
    chance=random.random()
    if chance <0.8:
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[random.randint(2,18):random.randint(20,40),:,:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)
        if random.random() >0.7:
            c = np.array([1, 1, 1])*255.0
            t = bm[:random.randint(1,10),:,:].reshape([-1,2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)
    elif chance >0.3 and chance <0.6:
        cc = random.randint(2,18)
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:,random.randint(2,18):random.randint(20,40),:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

        if random.random() >0.7:
            c = np.array([1, 1, 1])*255.0
            t = bm[:,:random.randint(1,10),:].reshape([-1,2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

    chance=random.random()
    if chance <0.8 :
        c = np.array([0, 0, 0])*255.0
        t = bm[25:random.randint(30,40),:random.randint(112,224),:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:,:10,:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,[255,255,255] ,thickness=1)

    chance=random.random()
    if chance <0.1: 
        im[:,:,0] = random.random()*255.0
    elif chance <0.2 and chance >0.1:
        im[:,:,1] = random.random()*255.0

    elif chance <0.6 and chance >0.4:
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:random.randint(20,45),:,:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)
    elif chance <0.8 and chance >0.6:
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:,:random.randint(20,45),:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

    chance=random.random()
    if random.random() >0.4:
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:,:random.randint(1,20),:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

    chance=random.random()
    if random.random() >0.4:
        c = np.array([random.random(), random.random(), random.random()])*255.0
        t = bm[:random.randint(1,20),:,:].reshape([-1,2])
        for j in range(len(t)):
            cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)

    chance=random.random()
    if chance >0.4:   
        c = np.array([random.random(), random.random(), random.random()])*255.0
        num = int(random.random()*20)
        cc = random.randint(10,15)
        for m in range(30):
            t = bm[num:num+20,50+cc*m:cc*m+57,:].reshape([-1,2])
            for j in range(len(t)):
                cv2.circle(im, (int(t[j,0]),int(t[j,1])), 1,c ,thickness=1)
                
    chance = random.random()            
    if chance > 0.9:
        im = 255 - im
    elif chance >0.85:
        im[:,:,0] = 255
    elif chance >0.8:
        im[:,:,0] = 255    
    elif chance >0.75:
        im[:,:,0] = 0
    elif chance >0.7:
        im[:,:,1] = 255 
    elif chance >0.65:
        im[:,:,1] = 0 
    elif chance >0.6:
        im[:,:,2] = 255 
    elif chance >0.55:
        im[:,:,2] = 0 
        
    return im/255

def texture(bg):
    size = 448
    chance=random.random()
    if chance > 0.4:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: size, : size, :]
    elif chance < 0.4 and chance> 0.1:
        c = np.array([random.random(), random.random(), random.random()])*0.8
        bg = np.ones((size, size, 3)) * c
    return bg


def tight_crop(im, bm, recon):
    # different tight crop
    size = 448
    bm = bm*size
    
    minx = int(np.min(bm[:,:,0]))-1
    maxx = int(np.max(bm[:,:,0]))+1
    miny = int(np.min(bm[:,:,1]))-1
    maxy = int(np.max(bm[:,:,1]))+1
        
    s = random.randint(14,40)

    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
    
    cx1 = minx + s - random.randint(5, s)
    cx2 = maxx + s + random.randint(5, s)
    cy1 = miny + s - random.randint(5, s)
    cy2 = maxy + s + random.randint(5, s)

    im = im[cy1 : cy2, cx1 : cx2, :]
    recon = recon[cy1 : cy2, cx1 : cx2, :]
    im = cv2.resize(im,(size,size))
    recon = cv2.resize(recon,(size,size))

    t = cy1 - s
    b = size + s - cy2 
    l = cx1 - s 
    r = size + s - cx2

    bm[:,:,1]=bm[:,:,1]-t
    bm[:,:,0]=bm[:,:,0]-l
    bm=bm/np.array([448.0-l-r, 448.0-t-b])
    return im, bm, recon

def img_flip(im, bm, recon):
    bm = bm*2 -1
    chance=random.random()
    if chance > 0.75:
        im = cv2.flip(im,0)
        bm[:,:,0] = cv2.flip(bm[:,:,0],0)
        bm[:,:,1] = -cv2.flip(bm[:,:,1],0)
        
        recon = cv2.flip(recon,0)
        recon[:,:,1] = 1 - recon[:,:,1]

    elif chance < 0.75 and chance> 0.5: 
        im = cv2.flip(im,1)
        bm[:,:,0] = -cv2.flip(bm[:,:,0],1)
        bm[:,:,1] = cv2.flip(bm[:,:,1],1)
        recon = cv2.flip(recon,1)
        recon[:,:,2] = 1 - recon[:,:,2]

    elif chance < 0.5 and chance> 0.25: 
        im = cv2.flip(im,0)
        im = cv2.flip(im,1)
        bm[:,:,0] = cv2.flip(bm[:,:,0],0)
        bm[:,:,1] = -cv2.flip(bm[:,:,1],0)
        bm[:,:,0] = -cv2.flip(bm[:,:,0],1)
        bm[:,:,1] = cv2.flip(bm[:,:,1],1)
        recon = cv2.flip(recon,0)
        recon = cv2.flip(recon,1)
        recon[:,:,1] = 1 - recon[:,:,1]
        recon[:,:,2] = 1 - recon[:,:,2]
    
    bm = (bm+1)/2
    return im, bm, recon



def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)
    
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
  
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
   
    f = random.uniform(-saturation, saturation)
    hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im


def data_aug(im, bm, recon, bg):
    im, bm, recon = tight_crop(im, bm, recon) 
    
    msk = np.stack([recon[:,:,0],recon[:,:,0],recon[:,:,0]],-1)
    im1 = color_line(im, bm)
    im = im * (1 - msk) + im1 * msk
    bg = texture(bg)
    if random.random() > 0.6:
        im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    im, bm, recon = img_flip(im, bm, recon)
    return im, bm, recon

