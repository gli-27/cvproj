from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_nc, ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        netE = [nn.Conv2d(input_nc, ngf, kernel_size=5, stride=2, padding=2), norm_layer(ngf), activation]
        for i in range(0, n_downsampling-1):
            mult = 2**i
            netE += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf*mult*2), activation]
        mult *= 2
        netE += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=2, stride=1), activation]
        self.netE = nn.Sequential(*netE)

    def forward(self, x):
        fea = self.netE(x)
        mu, std = torch.chunk(fea, 2, 1)
        return self.reparameterize(mu, std)

    def reparameterize(self, mu, std):
        std = torch.exp(0.5*std)
        eps = torch.randn_like(std)
        return mu + eps*std

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        activation = nn.ReLU(True)
        mult = 2**(n_downsampling-1)
        decoder = [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=1, stride=1, output_padding=1),
                   norm_layer(int(ngf*mult/2)), activation]
        for i in range(2, n_downsampling-1):
            mult = 2**(n_downsampling-i)
            decoder += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=1, padding=1),
                        norm_layer(int(ngf*mult/2)), activation]
        decoder += [nn.ConvTranspose2d(int(input_nc*mult/2), output_nc, kernel_size=3, stride=1, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z):
        return self.decoder(z)
