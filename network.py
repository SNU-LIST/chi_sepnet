'''
#
# Description:
#  Networks (x-sepnet, QSMnet, R2PRIMEnet) architecture codes
#  Architectures configured with U-Net
#
#  Copyright @ 
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : minjoony@snu.ac.kr
#
# Last update: 24.09.20
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class KAInet(nn.Module):
    def __init__(self, channel_in=32, kernel_size=3):
        super().__init__()
        
        self.conv11 = Conv3d(3, channel_in, kernel_size)
        self.conv12 = Conv3d(channel_in, channel_in, kernel_size)
        self.pool1  = Pool3d()
        
        self.conv21 = Conv3d(channel_in, 2*channel_in, kernel_size)
        self.conv22 = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        self.pool2  = Pool3d()
        
        self.conv31 = Conv3d(2*channel_in, 4*channel_in, kernel_size)
        self.conv32 = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        self.pool3  = Pool3d()
        
        self.conv41 = Conv3d(4*channel_in, 8*channel_in, kernel_size)
        self.conv42 = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        self.pool4  = Pool3d()
        
        self.l_conv1 = Conv3d(8*channel_in, 16*channel_in, kernel_size)
        self.l_conv2 = Conv3d(16*channel_in, 16*channel_in, kernel_size)
        
        self.deconv4 = Deconv3d(16*channel_in, 8*channel_in)
        self.conv51  = Conv3d(16*channel_in, 8*channel_in, kernel_size)
        self.conv52  = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        
        self.deconv3 = Deconv3d(8*channel_in, 4*channel_in)
        self.conv61  = Conv3d(8*channel_in, 4*channel_in, kernel_size)
        self.conv62  = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        
        self.deconv2 = Deconv3d(4*channel_in, 2*channel_in)
        self.conv71  = Conv3d(4*channel_in, 2*channel_in, kernel_size)
        self.conv72  = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        
        self.deconv1 = Deconv3d(2*channel_in, channel_in)
        self.conv81  = Conv3d(2*channel_in, channel_in, kernel_size)
        self.conv82  = Conv3d(channel_in, channel_in, kernel_size)        
        
        self.out = Conv(channel_in, 2)
        
    def forward(self, x):
        e1 = self.conv12(self.conv11(x))
        e2 = self.conv22(self.conv21(self.pool1(e1)))
        e3 = self.conv32(self.conv31(self.pool2(e2)))
        e4 = self.conv42(self.conv41(self.pool3(e3)))
        m1 = self.l_conv2(self.l_conv1(self.pool4(e4)))

        d4 = self.conv52(self.conv51(Concat(self.deconv4(m1),e4)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(d4),e3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        x  = self.out(d1)        
        return x
    
    
class QSMnet(nn.Module):
    def __init__(self, channel_in=32, kernel_size=3):
        super().__init__()
        
        self.conv11 = Conv3d(1, channel_in, kernel_size)
        self.conv12 = Conv3d(channel_in, channel_in, kernel_size)
        self.pool1  = Pool3d()
        
        self.conv21 = Conv3d(channel_in, 2*channel_in, kernel_size)
        self.conv22 = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        self.pool2  = Pool3d()
        
        self.conv31 = Conv3d(2*channel_in, 4*channel_in, kernel_size)
        self.conv32 = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        self.pool3  = Pool3d()
        
        self.conv41 = Conv3d(4*channel_in, 8*channel_in, kernel_size)
        self.conv42 = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        self.pool4  = Pool3d()
        
        self.l_conv1 = Conv3d(8*channel_in, 16*channel_in, kernel_size)
        self.l_conv2 = Conv3d(16*channel_in, 16*channel_in, kernel_size)
        
        self.deconv4 = Deconv3d(16*channel_in, 8*channel_in)
        self.conv51  = Conv3d(16*channel_in, 8*channel_in, kernel_size)
        self.conv52  = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        
        self.deconv3 = Deconv3d(8*channel_in, 4*channel_in)
        self.conv61  = Conv3d(8*channel_in, 4*channel_in, kernel_size)
        self.conv62  = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        
        self.deconv2 = Deconv3d(4*channel_in, 2*channel_in)
        self.conv71  = Conv3d(4*channel_in, 2*channel_in, kernel_size)
        self.conv72  = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        
        self.deconv1 = Deconv3d(2*channel_in, channel_in)
        self.conv81  = Conv3d(2*channel_in, channel_in, kernel_size)
        self.conv82  = Conv3d(channel_in, channel_in, kernel_size)        
        
        self.out = Conv(channel_in, 1)
        
    def forward(self, x):
        e1 = self.conv12(self.conv11(x))
        e2 = self.conv22(self.conv21(self.pool1(e1)))
        e3 = self.conv32(self.conv31(self.pool2(e2)))
        e4 = self.conv42(self.conv41(self.pool3(e3)))
        m1 = self.l_conv2(self.l_conv1(self.pool4(e4)))
        
        d4 = self.conv52(self.conv51(Concat(self.deconv4(m1),e4)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(d4),e3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        out  = self.out(d1)        
        return out
    

class R2PRIMEnet(nn.Module):
    def __init__(self, channel_in=32, kernel_size=3):
        super().__init__()
        
        self.conv11 = Conv3d(1, channel_in, kernel_size)
        self.conv12 = Conv3d(channel_in, channel_in, kernel_size)
        self.pool1  = Pool3d()
        
        self.conv21 = Conv3d(channel_in, 2*channel_in, kernel_size)
        self.conv22 = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        self.pool2  = Pool3d()
        
        self.conv31 = Conv3d(2*channel_in, 4*channel_in, kernel_size)
        self.conv32 = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        self.pool3  = Pool3d()
        
        self.conv41 = Conv3d(4*channel_in, 8*channel_in, kernel_size)
        self.conv42 = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        self.pool4  = Pool3d()
        
        self.l_conv1 = Conv3d(8*channel_in, 16*channel_in, kernel_size)
        self.l_conv2 = Conv3d(16*channel_in, 16*channel_in, kernel_size)
        
        self.deconv4 = Deconv3d(16*channel_in, 8*channel_in)
        self.conv51  = Conv3d(16*channel_in, 8*channel_in, kernel_size)
        self.conv52  = Conv3d(8*channel_in, 8*channel_in, kernel_size)
        
        self.deconv3 = Deconv3d(8*channel_in, 4*channel_in)
        self.conv61  = Conv3d(8*channel_in, 4*channel_in, kernel_size)
        self.conv62  = Conv3d(4*channel_in, 4*channel_in, kernel_size)
        
        self.deconv2 = Deconv3d(4*channel_in, 2*channel_in)
        self.conv71  = Conv3d(4*channel_in, 2*channel_in, kernel_size)
        self.conv72  = Conv3d(2*channel_in, 2*channel_in, kernel_size)
        
        self.deconv1 = Deconv3d(2*channel_in, channel_in)
        self.conv81  = Conv3d(2*channel_in, channel_in, kernel_size)
        self.conv82  = Conv3d(channel_in, channel_in, kernel_size)        
        
        self.out = Conv(channel_in, 1)
        
    def forward(self, x):
        e1 = self.conv12(self.conv11(x))
        e2 = self.conv22(self.conv21(self.pool1(e1)))
        e3 = self.conv32(self.conv31(self.pool2(e2)))
        e4 = self.conv42(self.conv41(self.pool3(e3)))
        m1 = self.l_conv2(self.l_conv1(self.pool4(e4)))
        
        d4 = self.conv52(self.conv51(Concat(self.deconv4(m1),e4)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(d4),e3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        out  = self.out(d1)        
        return out
    