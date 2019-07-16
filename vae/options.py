import argparse
import os
import torch
from util import util

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # for training
        self.parser.add_argument('--name', type=str, default='VAE_Network',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--isTrain', type=bool, default=True)
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size of training')
        self.parser.add_argument('--serial_batches', action='store_true', default=False,
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--gpu_ids', type=int, default='2', help='gpu ids: e.g. 1, 2, 3, use 0 for CPU')
        self.parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--continue_train', action='store_true', default=False,
                                 help='continue training: load the latest model')
        self.parser.add_argument('--display_freq', type=int, default=300,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=300,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=3000,
                                 help='frequency of saving the latest results')

        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input channel')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output channel')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of encoder/decoder \'s filters')
        self.parser.add_argument('--n_downsampling', type=int, default=3, help='# of times of downsampling')
        self.parser.add_argument('--n_layer', type=int, default=5, help='# of discriminator layers')
        self.parser.add_argument('--outf', default='./recon', help='folder to put recon-image')

        self.isTrain = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        """
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        """

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


