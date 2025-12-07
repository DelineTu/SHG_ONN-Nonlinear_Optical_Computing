import torch
import torch.nn as nn
import torch.optim as optim

from .base_plmodels import *
from .pl_utils import *
from .mlp import *

class NoiseDT(nn.Module):
    def __init__(self, A_dt, output_dim, Nλ):
        super().__init__()
        self.A_dt = A_dt
        self.output_dim = output_dim
        self.Nλ = Nλ

    def __call__(self, x, x_mean):
        """
        Generates the noise with the following input
        x: the input the physical NN
        x_mean: the mean output of the physical NN
        The reason for supplying the x_mean is to avoid recomputing the mean twice!
        Just does A*z where z is iid to give you a sample of noise
        """
        device = x.device

        A_x = torch.cat([x, x_mean], dim=1)
        with torch.no_grad():
            A_y = self.A_dt(A_x)

        A_mat = A_y.reshape([x.shape[0], self.output_dim, self.Nλ])
        z = torch.randn([x.shape[0], self.Nλ, 1], device=device)
        noise = torch.bmm(A_mat, z)
        return torch.squeeze(noise)
    
class MeanNoiseDT():
    def __init__(self, mean_dt, A_dt, output_dim, Nλ):
        super().__init__()
        for param in mean_dt.parameters():
            param.requires_grad = False
        for param in A_dt.parameters():
            param.requires_grad = False  

        self.mean_dt = mean_dt
        self.noise_dt = NoiseDT(A_dt, output_dim, Nλ)
    def __call__(self, x):
        x_mean = self.mean_dt(x)
        noise = self.noise_dt(x, x_mean)
        return x_mean + noise
        
# def make_noise_dt(A_dt, output_dim, Nλ):
#     def noise_dt(x, x_mean):
#         """
#         Generates the noise with the following input
#         x: the input the physical NN
#         x_mean: the mean output of the physical NN
#         The reason for supplying the x_mean is to avoid recomputing the mean twice!
#         Just does A*z where z is iid to give you a sample of noise
#         """
#         device = x.device

#         A_x = torch.cat([x, x_mean], dim=1)
#         with torch.no_grad():
#             A_y = A_dt(A_x)

#         A_mat = A_y.reshape([x.shape[0], output_dim, Nλ])
#         z = torch.randn([x.shape[0], Nλ, 1], device=device)
#         noise = torch.bmm(A_mat, z)
#         return torch.squeeze(noise)
#     return noise_dt

# def make_dt_func(mean_dt, A_dt, output_dim, Nλ):
#     for param in mean_dt.parameters():
#         param.requires_grad = False
#     for param in A_dt.parameters():
#         param.requires_grad = False  
        
#     noise_dt = make_noise_dt(A_dt, output_dim, Nλ)
#     def f(x):
#         x_mean = mean_dt(x)
#         noise = noise_dt(x, x_mean)
#         return x_mean + noise
#     return f