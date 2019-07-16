from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def KLLoss(mu, std):
    KLD = -0.5 * torch.sum(1 + std - mu.pow(2) - std.exp())
    return KLD

def ReconLoss(recon_x, x):
    Recon = F.l1_loss(recon_x, x)
    return Recon

"""
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
"""

class Encoder(nn.Module):
    def __init__(self, input_nc, ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        netE = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), norm_layer(ngf), activation]
        for i in range(0, n_downsampling-1):
            mult = 2**i
            netE += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=4, stride=2, padding=1),
                                norm_layer(ngf*mult*2), activation]
        mult *= 2
        netE += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=4, stride=2, padding=1), activation]
        self.netE = nn.Sequential(*netE)

    def forward(self, x):
        fea = self.netE(x)
        mu, std = torch.chunk(fea, 2, 1)
        return mu, std

    def reparameterize(self, mu, std):
        std = torch.exp(0.5*std)
        eps = torch.randn_like(std)
        return mu + eps*std

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        activation = nn.ReLU(True)
        netG = []
        mult = 2 ** (n_downsampling - 1)
        netG += [nn.ConvTranspose2d(in_channels=ngf*mult, out_channels=int(ngf*mult/2), kernel_size=5, stride=1,
                                                       padding=1, bias=False),
                    norm_layer(int(ngf*mult/2)), activation]
        mult /= 2
        netG += [nn.ConvTranspose2d(int(ngf*mult), int(ngf*mult/2), 5, 2, 1, bias=False), norm_layer(int(ngf*mult/2)),
                 activation]
        for i in range(2, n_downsampling):
            mult = 2**(n_downsampling - i - 1)
            netG += [nn.ConvTranspose2d(in_channels=int(ngf*mult), out_channels=int(ngf*mult/2), kernel_size=4, stride=2, padding=1,
                                        bias=False),
                        norm_layer(int(ngf*mult/2)), activation]
        mult = mult/2
        netG += [nn.ConvTranspose2d(in_channels=int(ngf*mult), out_channels=output_nc, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.Tanh()]
        self.decoder = nn.Sequential(*netG)

    def forward(self, z):
        return self.decoder(z)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ngf, n_layer=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(Discriminator, self).__init__()
        self.n_layer = n_layer

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        netD = []
        netD += [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf = ngf
        for n in range(1, n_layer):
            nf_prev = nf
            nf = min(nf*2, 128)
            netD += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]

        nf_prev = nf
        nf = min(nf * 2, 128)
        netD += [
            nn.Conv2d(nf_prev, 1, kernel_size=kw, stride=2, padding=1)
        ]

        if use_sigmoid:
            netD += [nn.Sigmoid()]

        self.netD = nn.Sequential(*netD)


    def forward(self, input):
        output = self.netD(input)
        return output.view(-1, 1).squeeze(1)
