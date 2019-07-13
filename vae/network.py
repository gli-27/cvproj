from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_nc, ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        encodingNetwork = [nn.Conv2d(input_nc, ngf, kernel_size=5, stride=1), norm_layer, activation]
        for i in range(0, n_downsampling-1):
            mult = 2**i
            encodingNetwork += [nn.Conv2d(input_nc, ngf)]