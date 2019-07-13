from data_loader import createDataLoader
from options import Options

opt = Options().parse()
dataloader = createDataLoader(opt)
dataset = dataloader.dataset

