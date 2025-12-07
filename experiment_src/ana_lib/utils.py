"""
Random util functions for data analysis
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from importlib import reload
import inspect
import os

def FWHM(X,Y):
    spline = UnivariateSpline(X, Y-np.max(Y)/2, s=0)
    ans = spline.roots() # find the roots
    return ans[-1] - ans[0]

def print_code(func):
    print(inspect.getsource(func))
    
def vec2mat(vec):
    """
    Transform a flattened vector into a matrix form
    Used for covariance matrix and the A matrix
    """
    dim = int(np.floor(np.sqrt(len(vec))))
    return vec.reshape(dim, dim)

def custom_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    #print(f"Directory '{dir_name}' created or already exists.")
        