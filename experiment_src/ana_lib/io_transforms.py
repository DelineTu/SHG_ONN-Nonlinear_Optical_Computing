"""
TODO:
1. Add dimension checking in the stupid function... To ensure that the cuda based bugs don't throw
"""


import torch
from .digitize import *

class InputTransform:
    def __init__(self, dat_dim, in_dim, left, right, device, Nfactor=20):
        """
        dat_dim : the dimension of the ml data
        in_dim : the input dimension of the PNN - for our current set of experiments its always set at 100.
        left (right) : a pytorch_tensor that encodes the left (right) boundary of the input_transform 
               a good way to instantiate it is via: left = nn.Parameter(torch.tensor(left, device=device), requires_grad=requires_grad)
        """
        self.left = left
        self.right = right

        self.dat_dim = dat_dim
        self.in_dim = in_dim
        self.device = device

    def __call__(self, data):
        """
        data refers to the data that will be transformed into a PNN input
        """
        device = self.device
        x_data = torch_linspace(self.left, self.right, self.dat_dim+2, device)
        x_data = torch.cat((torch.tensor([-1.0], device=device),
                       x_data, torch.tensor([2.0], device=device)))

        zero_tensor = torch.zeros(data.shape[0], 1, device=device) #this cannot be saved as batch size changes...
        data = torch.cat((zero_tensor, data, zero_tensor), dim=1)

        inp = torch_interp_nearest(x_data, data, self.x_int)
        return inp


class OutputTransform:
    def __init__(self, out_dim, samp_dim, left, right, device, Nfactor_factor=5):
        """
        out_dim refers to the output_dim of the PNN - for our current set of experiments its always set at 100.
        samp_dim refers to the sampling dimension that you want to eventually reduce to!
        left (right) : a pytorch_tensor that encodes the left (right) boundary of the output_transform 
               a good way to instantiate it is via: left = nn.Parameter(torch.tensor(left, device=device), requires_grad=requires_grad)        
        """
        self.left = left
        self.right = right

        self.out_dim = out_dim
        self.Nfactor = np.int(np.round(Nfactor_factor*out_dim/samp_dim))
        self.Nx = samp_dim*self.Nfactor
        self.x_int = torch.linspace(0.0, 1.0, out_dim, device=device) #It need to be moved to __call__ if x_int is backproped on...
        self.device = device
        
    def __call__(self, out):
        """
        out is the output from the PNN that will be down-sampled
        """
        x_axis = torch_linspace(self.left, self.right, self.Nx, self.device)
        out_interp = torch_interp_nearest(self.x_int, out, x_axis)
        samp_out = mean_downsampling(out_interp, self.Nfactor)
        return samp_out