import torch 
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import ipdb

def get_dataloader(dataset, num_batch, batch_size, **dataloader_kwargs):
    dat_size = batch_size*num_batch
    dataset_keep = random_split(dataset, [dat_size, len(dataset)-dat_size])[0]
    return DataLoader(dataset_keep, batch_size=batch_size, **dataloader_kwargs)

def reshape_f_sum(x):
    """
    First dimension of x is the channel! Thus there is only 1 for MNIST...
    """
#     ipdb.set_trace()
    mats = [x[0, i::compress_factor, j::compress_factor] for i in range(compress_factor) for j in range(compress_factor)]
    mat_sum = torch.sum(torch.stack(mats), 0)
    return torch.reshape(mat_sum, (-1, ))

def MNIST_1D_data(batch_size_train, num_batch_train, batch_size_test, num_batch_test
                  , data_file="../data", compress_factor=1, hilbert=False):
    reshape_f = lambda x: torch.reshape(x[0, ::compress_factor, ::compress_factor], (-1, ))
    transform = transforms.Compose([transforms.ToTensor(), reshape_f])
#     transform = transforms.Compose([transforms.ToTensor(), reshape_f_sum])
    tr_data = MNIST(data_file, train=True, download=True, transform=transform)
    te_data = MNIST(data_file, train=False, download=True, transform=transform)
    train_loader = get_dataloader(tr_data, num_batch_train, batch_size_train, shuffle=False)
    test_loader = get_dataloader(te_data, num_batch_test, batch_size_test, shuffle=False)
    return train_loader, test_loader