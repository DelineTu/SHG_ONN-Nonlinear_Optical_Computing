import torch
import torch.nn as nn
from .pl_utils import *

class Manifold(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, dim, factor_init=0.8, offset_init=0.2):
        super().__init__()
        self.factors = nn.Parameter(factor_init*torch.ones(dim))
        self.offsets = nn.Parameter(offset_init*torch.zeros(dim))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factors + self.offsets
        return self.out
    
class Attenuator(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, dim, factor_init=0.8):
        super().__init__()
        self.factors = nn.Parameter(factor_init*torch.ones(dim))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factors
        return self.out
    
class SingleManifold(nn.Module):
    """
    Initializes mostly in the identity map
    """
    def __init__(self, factor_init=0.8, offset_init=0.2):
        super().__init__()
        self.factor = nn.Parameter(factor_init*torch.tensor(factor_init))
        self.offset = nn.Parameter(offset_init*torch.tensor(offset_init))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factor + self.offset
        return self.out
    
class SingleMult(nn.Module):
    def __init__(self, factor_init=0.8):
        super().__init__()
        self.factor = nn.Parameter(factor_init*torch.tensor(factor_init))
        
    def forward(self, x):
        self.x = x
        self.out = x*self.factor 
        return self.out