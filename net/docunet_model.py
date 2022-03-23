import torch.nn.functional as F
from .unet_parts import *

class DocUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DocUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.inc2 = DoubleConv(64+n_classes, 64)
        self.down21 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down23 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down24 = Down(512, 1024 // factor)
        self.up21 = Up(1024, 512 // factor, bilinear)
        self.up22 = Up(512, 256 // factor, bilinear)
        self.up23 = Up(256, 128 // factor, bilinear)
        self.up24 = Up(128, 64, bilinear)
        self.outc2 = OutConv(64, n_classes)


    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        x = torch.cat((logits,x), 1)
        x1 = self.inc2(x)
        x2 = self.down21(x1)
        x3 = self.down22(x2)
        x4 = self.down23(x3)
        x5 = self.down24(x4)
        x = self.up21(x5, x4)
        x = self.up22(x, x3)
        x = self.up23(x, x2)
        x = self.up24(x, x1) 
        logits1 = self.outc2(x)
        return logits1