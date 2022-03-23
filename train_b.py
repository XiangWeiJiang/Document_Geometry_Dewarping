import argparse
import logging
import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from net import DocUNet

from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_b import doc3dLoader
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import itertools

dir_checkpoint = 'pkl/docunet/'
data_dir = "dataset/doc3d/"

def train_net(net,
              device,
              epochs=500,
              batch_size=2,
              lr=0.001,
              val_percent=0.01,
              save_cp=True,
              img_size= 128):
    dataset = doc3dLoader(root = data_dir, is_aug=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_size}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    l1 = nn.L1Loss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            loss_sum = 0
            for img, bm in train_loader:
                img = img.to(device=device, dtype=torch.float32)
                bm = bm.to(device=device, dtype=torch.float32)
                
                pred = net(img)
  
                t = bm[:,:,0,:]
                r = bm[:,:,:,-1]
                d = bm[:,:,-1,:]
                l = bm[:,:,:,0]

                tt = pred[:,:,0,:]
                rr = pred[:,:,:,-1]
                dd = pred[:,:,-1,:]
                ll = pred[:,:,:,0]
                
                loss = l1(t,tt)+l1(r,rr)+l1(d,dd)+l1(l,ll)
                
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(img.shape[0])
                global_step += 1
                loss_sum = loss_sum + loss
                if global_step % (n_train // (2 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation loss: {}'.format(val_score))
            writer.add_scalar('epoch_loss', epoch_loss, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()


def eval_net(net, testloader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(testloader)  # the number of batch
    tot = 0
    l1 = nn.L1Loss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for img, bm in testloader:
            img = img.to(device=device, dtype=torch.float32)
            bm = bm.to(device=device, dtype=torch.float32)
 
            pred = net(img)

            t = bm[:,:,0,:]
            r = bm[:,:,:,-1]
            d = bm[:,:,-1,:]
            l = bm[:,:,:,0]

            tt = pred[:,:,0,:]
            rr = pred[:,:,:,-1]
            dd = pred[:,:,-1,:]
            ll = pred[:,:,:,0]
            loss = l1(t,tt)+l1(r,rr)+l1(d,dd)+l1(l,ll)

            tot += loss.item()
            
            del img
            del bm
            del pred
            pbar.update()
    net.train()
    return tot / n_val
    
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-s', '--img-size', metavar='s', type=int, nargs='?', default=128,
                        help='Image size', dest='imgsize') 
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = DocUNet(n_channels=3, n_classes=2, bilinear=True)
    

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size= args.imgsize,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

