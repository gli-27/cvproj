import torch
import network

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
            self.netG = network.Decoder(opt.input_nc, opt.output_nc, opt.ngf, opt.n_downsampling)