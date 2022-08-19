import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler



def train_val_load(root, batch_size = 32, valid_size= 1/50, workers = 2):

    print('Downloading or checking the presence of the data:')
    trainset = datasets.CIFAR10(root=root, train=True, download=True )
    data = trainset.data/255

    m = data.mean(axis = (0,1,2))
    s = data.std(axis = (0,1,2))

    norm = transforms.Normalize(mean=m,std=s)
    trans = transforms.Compose([transforms.ToTensor(),norm])

    train = valid = datasets.CIFAR10(root=root, train=True, download=True, transform=trans)

    train_size = len(train)
    id = list(range(train_size))

    split = train_size - int(np.floor(valid_size*train_size))

    train_id = id[:split]
    valid_id = id[split:]

    train_sampler = SubsetRandomSampler(train_id)
    valid_sampler = SubsetRandomSampler(valid_id)

    train_load = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        sampler = train_sampler,
        num_workers = workers,
        pin_memory = False
    )
    valid_load = torch.utils.data.DataLoader(
        valid,
        batch_size=batch_size,
        sampler = valid_sampler,
        num_workers = workers,
        pin_memory = False
    )

    print('Train size:', len(train_load.sampler))
    print('Validation size:', len(valid_load.sampler))

    return train_load, valid_load

def test_loader(root, batch_size=32,workers =2):
    print('Downloading or checking the presence of the data:')
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    data = testset.data/255

    m = data.mean(axis = (0,1,2))
    s = data.std(axis = (0,1,2))

    norm = transforms.Normalize(mean=m,std=s)
    trans = transforms.Compose([transforms.ToTensor(),norm])

    test = datasets.CIFAR10(root='./data',train=False, download=True, transform=trans)
    test_load = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False,num_workers = workers, pin_memory = False)
    
    print('Test size:', len(test_load.sampler))

    return test_load

