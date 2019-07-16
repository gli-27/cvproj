import os
import time
import numpy as np
import model
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import optim
from data_loader import createDataLoader
from options import Options

opt = Options().parse()
print(opt)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.gpu_ids)

cudnn.benchmark = True

dataloader = createDataLoader(opt)
dataset = dataloader.data_loader(opt)
dataset_size = len(dataset)

model = model.create_model(opt)
model = model.cuda()
model = model.to(device)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

real_label = 1.0
fake_label = 0.0

optimizer_E = optim.Adam(model.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_G = optim.Adam(model.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D = optim.Adam(model.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(start_epoch, opt.niter+opt.niter_decay + 1):
    epoch_start_time = time
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        save_fake = total_steps % opt.display_freq == display_delta
        data = data[0].to(device)

        ######### train with real ##########
        model.netD.zero_grad()
        batch_size = data.size(0)
        label = torch.full((batch_size, ), real_label, device=device)

        output = model.netD(data)
        errD_real = model.criterionGAN(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ######### train with real #########
        mu, std = model.netE.forward(Variable(data))
        z = model.netE.reparameterize(mu, std)
        recon_x = model.netG.forward(z)
        label.fill_(fake_label)
        output = model.netD(recon_x.detach())
        errD_fake = model.criterionGAN(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real+errD_fake
        optimizer_D.step()

        model.netE.zero_grad()
        model.netG.zero_grad()
        label.fill_(real_label)
        output = model.netD(recon_x)
        err = model.criterionGAN(output, label)
        errKL = model.criterionKL(mu, std)
        errRecon = model.criterionRecon(recon_x, data)
        errG = err + errRecon
        # errG.backward()
        D_G_z2 = output.mean().item()

        errE = errG + errKL
        errE.backward()
        D_G_z3 = output.mean().item()
        optimizer_G.step()
        optimizer_E.step()

        if total_steps % opt.print_freq == print_delta:
            t = (time.time() - iter_start_time) / opt.print_freq
            print(t)
            print('Error of D:', end=' ')
            print(errD, end='; ')
            print('Error of G', end=' ')
            print(errG, end='; ')
            print('Error of E:', end=' ')
            print(errE)

        if i % 1000 == 0:
            print('save images')
            vutils.save_image(data,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)
            fake = model.netG(z)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                              normalize=True)
    print('epoch time: ', end='')
    print(epoch)

