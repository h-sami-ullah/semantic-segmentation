import torch
from torch.nn import Module
from .layers import *


class simple_unet(Module):

    def __init__(self, in_ch: int, classes: int):
        super(simple_unet, self).__init__()
        self.args = {'input_channels': in_ch, 'classes': classes}

        self.inconv = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024+512, 512)
        self.up2 = Up(512+256, 256)
        self.up3 = Up(256+128, 128)
        self.up4 = Up(128+64, 64)
        self.outc = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logout = self.outc(x)
        return logout



