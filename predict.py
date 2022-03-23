import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from optim.tps import tps
from optim.opt import opt
from utils.line import line
from utils.line1 import line1

from net import DocUNet,UNet
import cv2
import time


def unwarp(img, bm):
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).double()
    bm = bm.detach().numpy()

    n,c,h,w=img.shape
    zz = 1
    bm0=cv2.blur(bm[:,:,0],(zz,zz))
    bm1=cv2.blur(bm[:,:,1],(zz,zz))
    bm0=cv2.resize(bm0,(w,h))
    bm1=cv2.resize(bm1,(w,h))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()
    
    res = F.grid_sample(input=img, grid=bm,align_corners=True)
    res = res[0].numpy().transpose((1, 2, 0))
    
    if args.show:
        import matplotlib.pyplot as plt 
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(bm[0,:,:,0].detach().numpy())
        axarr[1].imshow(bm[0,:,:,1].detach().numpy())
        axarr[2].imshow(res/255)
        plt.show()
    return res

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--docunet', '-d', default='pkl/CP_epoch21_sota.pth',#'pkl/CP_epoch1_.pth'
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--unet', '-u', default='pkl/textline.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--crop', '-c', default='data/crop/',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")   
    parser.add_argument('--method', '-m', default='grid',#tfi ,tps,grid
                        metavar='FILE',
                        help="Specify the file in which the model is stored") 
    parser.add_argument('--show', '-s', default=False,
                    metavar='FILE',
                    help="Specify the file in which the model is stored") 
    return parser.parse_args()

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.info("Start print log")

    args = get_args()
    b_net = DocUNet(n_channels=3, n_classes=2)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b_net.to(device=device)
    b_net.load_state_dict(torch.load(args.docunet, map_location=device))

    t_net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_net.to(device=device)
    t_net.load_state_dict(torch.load(args.unet, map_location=device))
    
    logger.info("Load pkl")

    b_net.eval()
    t_net.eval()

    method = args.method
    crop = args.crop
    
    b_size = (128,128)
    t_size = (512,512)
    for img_name in tqdm.tqdm(os.listdir(crop)):
        begin = time.time()
        img = cv2.imread(crop+img_name)
        img_b = cv2.resize(img,b_size)/255
        img_t = cv2.resize(img,t_size)/255
    
        img_b = torch.from_numpy(img_b).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32)
        img_t = torch.from_numpy(img_t).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32)

        boundary = b_net(img_b)
        textline_mask = t_net(img_t)
        
        if method == "tfi":
            M = 128
            m = torch.range(0,M-1,1)/(M-1)
            w1 = torch.stack([m[range(M-1,-1,-1)],m],1).cuda()
            w2 = torch.stack([m[range(M-1,-1,-1)],m],0).cuda()

            ud = torch.stack([boundary[0,:,:,0],boundary[0,:,:,-1]],-1).cuda()
            lr = torch.stack([boundary[0,:,0,:],boundary[0,:,-1,:]],-1).permute(0,2,1).cuda()
            
            corner1 = torch.stack([boundary[0,:,0,0],boundary[0,:,-1,0]],-1)
            corner2 = torch.stack([boundary[0,:,0,-1],boundary[0,:,-1,-1]],-1)
            corner = torch.stack([corner1,corner2],-1).cuda()
            
            grid = torch.matmul(w1,lr) + torch.matmul(ud,w2) - torch.matmul(torch.matmul(w1,corner),w2)
            grid = grid.permute(1,2,0).cpu()
            logger.info("TFI")
            uwpred=unwarp(img, grid)
        
        elif method == "tps":
            grid = tps(boundary)
            logger.info("TPS")
            uwpred=unwarp(img, grid)
        else:
            textline_mask = (textline_mask.cpu().detach().numpy()[0,0]>0.3)*255
            cv2.imwrite("result/text_line/"+img_name,textline_mask)
            textline_np = line(textline_mask)
            line1_np = line1(textline_np,img_name)

            logger.info("Line process")
            grid = opt(boundary,textline_np,line1_np)

            logger.info("Optimize")
        
            uwpred =unwarp(img, grid)

        outp= os.path.join("result",method,img_name)
        cv2.imwrite(outp,uwpred)
        logger.info("Image write")
    