import os
import time
import numpy as np
import model
from torch.autograd import Variable
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

dataloader = createDataLoader(opt)
dataset = dataloader.dataset
dataset_size = len(dataset)

model = model.create_model(opt)

for epoch in range(start_epoch, opt.niter+opt.niter_decay + 1):
    epoch_start_time = time
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        data = data[0].unsqueeze(0)
        z = model.netE.forward(Variable(data))
        recon = model.netG.forward(z)
        print(recon)
