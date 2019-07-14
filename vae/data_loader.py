import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

def createDataLoader(opt):
    data_loader = DataLoader(opt)
    print(data_loader.name())
    return data_loader

def createDataset(opt):
    if opt.isTrain:
        dataset = datasets.MNIST(root="./data/",
                                transform=transforms.ToTensor(),
                                train=True,
                                download=True)
    else:
        dataset = datasets.MNIST(root="./data/",
                                   transform=transforms.ToTensor(),
                                   train=False)
    return dataset

def get_transform(normalize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                           (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class DataLoader(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataset = createDataset(opt)

    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'MNIST'
