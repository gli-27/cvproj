import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataset = createDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'MNIST'

def createDataLoader(opt):
    data_loader = DataLoader(opt)
    print(data_loader.name())
    return data_loader

def createDataset(opt):
    if opt.isTrain:
        dataset = datasets.MNIST(root="./data/",
                                transform=transforms,
                                train=True,
                                download=True)
    else:
        dataset = datasets.MNIST(root="./data/",
                                   transform=transforms,
                                   train=False)
    return dataset
