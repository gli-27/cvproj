import torch
import network
import torch.nn as nn

def create_model(opt):
    if opt.isTrain:
        model = TrainModel(opt)
    return model

class TrainModel(torch.nn.Module):
    def name(self):
        return 'VAENetwork'

    def __init__(self, opt):
        super(TrainModel, self).__init__()
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.netE = network.Encoder(opt.input_nc, opt.ngf, opt.n_downsampling)
            self.netE.apply(network.weights_init)
            self.netG = network.Decoder(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsampling)
            self.netG.apply(network.weights_init)
            self.netD = network.Discriminator(opt.input_nc, opt.ngf, opt.n_layer)
            self.netD.apply(network.weights_init)
            self.criterionGAN = nn.BCELoss()
            self.criterionKL = network.KLLoss
            self.criterionRecon = network.ReconLoss
        else:
            pass

