import torch
import numpy as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from dataloaders import MnistDataloader, FashionMnistDataloader


class NoisedCTDataset(Dataset):
    def __init__(self, x, b):
        self.x = x
        self.x_noise = torch.normal(self.x * (1-b)**(0.5), b**(0.5))
        self.y = torch.from_numpy(np.concatenate((np.zeros(self.x.shape[0]), np.ones(self.x.shape[0])), axis=0))
        self.x = torch.cat((self.x, self.x_noise), 0)
        self.y_hot = F.one_hot(self.y.long(), num_classes=2).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, i):
        return self.x[i], self.y_hot[i]

def build_datasets(x_train, x_test, noise_schedule):
    
    # Initializing x_train data, dequantizing and normalizing our original set
    x_train_torch = torch.from_numpy(np.array(x_train))
    x_train_norm = (x_train_torch + torch.rand(x_train_torch.shape)) / 255.
    x_test_torch = torch.from_numpy(np.array(x_test))
    x_test_norm = (x_test_torch + torch.rand(x_test_torch.shape)) / 255.

    # Populating our list of x_train datasets, each with added 50% from previous set and 50% with added guassian noise
    print('Initializing set ' + str(1))
    train_sets = [NoisedCTDataset(x_train_norm, noise_schedule[0])]
    test_sets = [NoisedCTDataset(x_test_norm, noise_schedule[0])]
    for i in range(1, len(noise_schedule)):
        print('Initializing set ' + str(i+1))
        train_sets.append(NoisedCTDataset(train_sets[-1].x_noise, noise_schedule[i]))
        test_sets.append(NoisedCTDataset(test_sets[-1].x_noise, noise_schedule[i]))
    return train_sets, test_sets

def get_dataset(name, num_steps, noise_start, noise_end):

    noise_schedule = torch.linspace(noise_start, noise_end, num_steps)

    if name == 'mnist':
        dataloader = MnistDataloader()
        x_train, x_test = dataloader.load_data()
        return build_datasets(x_train, x_test, noise_schedule)
    elif name == 'fashion_mnist':
        dataloader = FashionMnistDataloader()
        x_train, x_test = dataloader.load_data()
        return build_datasets(x_train, x_test, noise_schedule)
    else:
        raise ValueError(f'Unknown dataset {name}')
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


