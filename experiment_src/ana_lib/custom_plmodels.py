"""
Insert all the custom plmodels that are used in this file!
"""
from .base_plmodels import *
from .pl_utils import *
from .mlp import *
import torch.optim as optim

import ipdb
from functools import reduce

#comment out if don't need to reload...
from importlib import reload
from . import pl_utils
reload(pl_utils)
from .pl_utils import *

from . import base_plmodels
reload(base_plmodels)
from .base_plmodels import *

class MLP_Reg(RegressionPlModel):
    """Vanilla multilayer perceptron model for regression."""

    def __init__(self, input_dim=None, output_dim=None, Nunits=None, lr=None):
        super().__init__()
        self.save_hyperparameters()  # this appends all the inputs into self.hparams
        for (i, Nunit) in enumerate(Nunits):  # writing more attributes to hyperparams that will appear on wandb
            self.hparams[f'ldim_{i}'] = Nunit
        self.hparams['Nlayers'] = len(Nunits)  # repeat - adding Nlayers
        self.model = MLP(input_dim, output_dim, Nunits)  # Multilayer Perceptron pytorch model

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer
    
class MLP_TL_Reg(RegressionPlModel):
    """
    Vanilla multilayer perceptron model for regression,
    with an additional transfer learning (TL) feature for the
    optimization.
    """

    def __init__(self, input_dim=None, output_dim=None, Nunits=None, lr=None):
        super().__init__()
        self.save_hyperparameters()  # this appends all the inputs into self.hparams
        for (i, Nunit) in enumerate(Nunits):  # writing more attributes to hyperparams that will appear on wandb
            self.hparams[f'ldim_{i}'] = Nunit
        self.hparams['Nlayers'] = len(Nunits)  # repeat - adding Nlayers
        self.model = MLP(input_dim, output_dim, Nunits)  # Multilayer Perceptron pytorch model

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(get_parameters(self.model.layers[-1::]), lr=self.hparams.lr)
        return optimizer
    
class ManifoldPlModel(Classification1DPlModel):
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        self.offsets = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims[:-1]:
            self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 
            self.offsets.append(nn.Parameter(torch.zeros(dim)))

        self.A = nn.Parameter(torch.randn(dims[-1])) #for the RC
        self.b = nn.Parameter(torch.tensor(0.0)) #for the RC
        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        if save:
            self.xin = []
            self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l]+self.offsets[l] #manifold
            self.xPLMs.append(x)
            if save:
                self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            if save:
                self.xout.append(x.detach())
        return torch.sum(self.A*x, dim=1) + self.b
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    
#This is the new model to edit today!
class ManifoldClassPlModel(ClassificationPlModel):
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        self.offsets = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims:
            self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 
            self.offsets.append(nn.Parameter(torch.zeros(dim)))

        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        self.xin = []
        self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l]+self.offsets[l] #manifold
            self.xPLMs.append(x)
            self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            self.xout.append(x.detach())
                
        x = x*self.factors[-1]+self.offsets[-1]
        self.xin.append(x.detach())
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

# class AttenuationClassPlModel(ClassificationPlModel):
#     def __init__(self, flist, dims, lr):
#         #first do the vanilla plmode stuff for logging
#         super().__init__()
#         self.save_hyperparameters()
#         #delete flist since it is a function which cannot be saved as 
#         #JSON and will cause a bug...
#         del self.hparams["flist"]
#         self.hparams["Nlayers"] = len(flist)
        
#         #now define the objects required for the displacement model.
#         self.factors = nn.ParameterList()
        
#         self.dims = dims
#         nlayers = len(flist)
#         for dim in dims:
#             self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 

#         self.flist = flist
#         self.xPLMs = []
        
#     def forward(self, x, save=False):
#         x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
#         if save:
#             self.xin = []
#             self.xout = []
            
#         for (l, f) in enumerate(self.flist):
#             x = x*self.factors[l] #manifold
#             self.xPLMs.append(x)
#             if save:
#                 self.xin.append(x.detach())
#             x = x.clamp(0.0, 1.0)
#             x = f(x)
#             if save:
#                 self.xout.append(x.detach())
                
#         x = x*self.factors[-1]
#         if save:
#             self.xin.append(x.detach())
#         return x
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
#         return optimizer
    
    
#This is the new model to edit today!
class AttenuationClassPlModel(ClassificationPlModel):
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims:
            self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 

        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        self.xin = []
        self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l] #manifold
            self.xPLMs.append(x)
            self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            self.xout.append(x.detach())
                
        x = x*self.factors[-1]
        self.xin.append(x.detach())
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    
#This is the new model to edit today!
class AttenuationClassPlModel_V2(ClassificationPlModel):
    """
    The v2 version does manifold on the input data!
    """
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims:
            self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 

        self.offset = nn.Parameter(torch.zeros(dims[0]))
        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        if save:
            self.xin = []
            self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l] #manifold
            if l==0:
                x += self.offset
            self.xPLMs.append(x)
            if save:
                self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            if save:
                self.xout.append(x.detach())
                
        x = x*self.factors[-1]
        if save:
            self.xin.append(x.detach())
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

class AttenuationClassPlModel_V3(ClassificationLagPlModel):
    """
    Do a Lagrangian model.
    """
    def __init__(self, flist, dims, lr, lag_amp=1.0, lag_factor=30.0):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims:
            self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 

        self.flist = flist
        
    def forward(self, x):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        self.xin = []
        self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l] #manifold
            self.xin.append(x)
            x = x.clamp(0.0, 1.0)
            x = f(x)
            self.xout.append(x.detach())
                
        x = x*self.factors[-1]
        self.xout.append(x.detach())
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def lagrangian(self):
        """
        The lagrangian function that will be added to the loss.
        """
        lag_layers = [self.hparams.lag_amp*
                      clamp_lag(x, 0.0, 1.0, self.hparams.lag_factor)
                      for x in self.xin]
        #use stack to convert python list to pytorch tensor, then take the mean
        out = torch.mean(torch.stack(lag_layers))

        return out

    
class DispClassPlModel(ClassificationPlModel):
    """
    The classification version that takes 7 outputs and uses the max for the classification!
    """
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        self.offsets = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims[:-1]:
#             self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 
            self.factors.append(torch.nn.Parameter(torch.ones(1), requires_grad=True))
    
        for dim in dims:
            self.offsets.append(nn.Parameter(torch.randn(dim)))

        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        if save:
            self.xin = []
            self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l]+self.offsets[l] #manifold
            self.xPLMs.append(x)
            if save:
                self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            if save:
                self.xout.append(x.detach())
        return x+self.offsets[-1] #in the final layer displace again
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
class DispPlModel(Classification1DPlModel):
    def __init__(self, flist, dims, lr):
        #first do the vanilla plmode stuff for logging
        super().__init__()
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as 
        #JSON and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        
        #now define the objects required for the displacement model.
        self.factors = nn.ParameterList()
        self.offsets = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        for dim in dims[:-1]:
#             self.factors.append(nn.Parameter(0.9*torch.ones(dim))) 
            self.factors.append(torch.nn.Parameter(torch.ones(1), requires_grad=True))
            self.offsets.append(nn.Parameter(torch.zeros(dim)))

        self.A = nn.Parameter(torch.randn(dims[-1])) #for the RC
        self.b = nn.Parameter(torch.tensor(0.0)) #for the RC
        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(self.dims[0]/12))
        if save:
            self.xin = []
            self.xout = []
            
        for (l, f) in enumerate(self.flist):
            x = x*self.factors[l]+self.offsets[l] #manifold
            self.xPLMs.append(x)
            if save:
                self.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            if save:
                self.xout.append(x.detach())
        return torch.sum(self.A*x, dim=1) + self.b


    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    
class InputPlModel(Classification1DLagPlModel):
    """
    The input plmodel!
    For this class, the code is designed to be quite "flat", so it
    explicitly includes the forward code in the definition.
    (Instead of adding another "model" attribute.)
    The lagrangian is also defined as required by the Classification1DLagPlModel
    class.
    """
    def __init__(self, flist, dims, lr, lag_amp=1.0, lag_factor=30.0):
        super().__init__()
        
        #first take care of pl logistics...
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as JSON
        #and will cause a bug...
        del self.hparams["flist"] 
        self.hparams["Nlayers"] = len(flist)
        self.accu_metric = Accuracy()
        
        #now define the objects required for this model
        self.inputs = nn.ParameterList()
        self.factors = nn.ParameterList()
        
        self.dims = dims
        nlayers = len(flist)
        
        for dim in dims[:-1]:
            self.inputs.append(nn.Parameter(0.5*torch.ones(dim//2), requires_grad=True))
            self.factors.append(torch.nn.Parameter(torch.ones(1), requires_grad=True))

        self.A = nn.Parameter(torch.randn(dims[-1])) #for the RC
        self.b = nn.Parameter(torch.randn(1)) #for the RC
        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=self.dims[0]//24)
        self.xin = []
        if save:
#             self.xin = []
            self.xout = []
            
        for (l, f) in enumerate(self.flist):
            #need to generalize this more if you want to change the layer later!
            repeat_input = self.inputs[l].repeat(x.shape[0], 1) 
            x = self.factors[l]*x
            
            #commented code is the pars on left, data on right
#             x = torch.cat([repeat_input, x], dim=1)

            #do the interveaving version here!
            x = torch.stack((repeat_input, x), dim=2)
            x = x.view(repeat_input.shape[0], 2*repeat_input.shape[1])
    
            #always required for the lagrangian term
            self.xin.append(x)
            x = x.clamp(0.0, 1.0)
#             if save:
#                 self.xin.append(x.detach())
            x = f(x)
            if save:
                self.xout.append(x.detach())
        return torch.sum(self.A*x, dim=1) + self.b
    
    def lagrangian(self):
        """
        The lagrangian function that will be added to the loss.
        """
#         ipdb.set_trace()
#         lag_layers = [self.hparams.lag_amp*
#                       clamp_lag(inp, 0.0, 1.0, self.hparams.lag_factor)
#                       for inp in self.inputs]
        
#         #use stack to convert python list to pytorch tensor, then take the mean
#         out = torch.mean(torch.stack(lag_layers))

        lag_layers = [self.hparams.lag_amp*
                      clamp_lag(x, 0.0, 1.0, self.hparams.lag_factor)
                      for x in self.xin]
        #use stack to convert python list to pytorch tensor, then take the mean
        out = torch.mean(torch.stack(lag_layers))
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    
class InputPlModel_v2(ClassificationLagPlModel):
    """
    The "new" input plmodel!
    First have a attenuation... for the output!
    Try to clamp things...
    """
    def __init__(self, flist, dims, lr, lag_amp=1.0, lag_factor=30.0):
        """
        dims are dimensions of the input parameters.
        """
        super().__init__()
        
        #first take care of pl logistics...
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as JSON
        #and will cause a bug...
        del self.hparams["flist"]
        self.hparams["Nlayers"] = len(flist)
        self.accu_metric = Accuracy()
        
        #now define the objects required for this model
        self.inputs = nn.ParameterList()
        self.factors = nn.ParameterList()
        self.att = torch.nn.Parameter(torch.ones(7), requires_grad=True)
        
        self.dims = dims
        nlayers = len(flist)
        
        for dim in dims[:-1]:
            self.inputs.append(nn.Parameter(0.5*torch.ones(dim), requires_grad=True))
            self.factors.append(torch.nn.Parameter(torch.ones(1), requires_grad=True))

        self.flist = flist
        self.xPLMs = []
        
    def forward(self, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=self.dims[0]//12)
        self.xin = []
        self.xout = []
            
        for (l, f) in enumerate(self.flist):
            #need to generalize this more if you want to change the layer later!
            repeat_input = self.inputs[l].repeat(x.shape[0], 1) 
            x = self.factors[l]*x #this does not apply to the input parameters!
            
            #commented code is the pars on left, data on right
#             x = torch.cat([repeat_input, x], dim=1)

            #do the interveaving version here!
            x = torch.stack((repeat_input, x), dim=2)
            x = x.view(repeat_input.shape[0], 2*repeat_input.shape[1])
    
            #always required for the lagrangian term
            self.xin.append(x)
            x = x.clamp(0.0, 1.0)
            x = f(x)
            self.xout.append(x.detach()) #just always attach!
            
        x = self.att*x #final attenuation layer for the PNN
        return x
#         return torch.sum(self.A*x, dim=1) + self.b
    
    def lagrangian(self):
        """
        The lagrangian function that will be added to the loss.
        """
        lag_layers = [self.hparams.lag_amp*
                      clamp_lag(x, 0.0, 1.0, self.hparams.lag_factor)
                      for x in self.xin]
        #use stack to convert python list to pytorch tensor, then take the mean
        out = torch.mean(torch.stack(lag_layers))

        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer