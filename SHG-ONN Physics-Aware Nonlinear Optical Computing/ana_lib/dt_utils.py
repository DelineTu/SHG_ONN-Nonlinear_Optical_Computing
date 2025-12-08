from numpy import random
import numpy as np

def lp_xlist(Nx, xdim, Nlp, low=0.0, high=1.0):
    """
    Nx: Is the number of samples you want
    xdim: Is the dimension of the output
    Nlp: The number of points to low pass over
    Thus the effective dimension of the input is about xdim/Nlp
    """
    xlist = random.uniform(low, high, [Nx, xdim+Nlp-1])
    y = np.array([np.convolve(x, np.ones(Nlp)/Nlp, mode='valid') for x in xlist])
    mid = (low+high)/2
    y = (y-mid)*np.sqrt(Nlp) + mid
    return y.clip(0, 1)


def random_xlist(Nx, xdim, low=0.0, high=1.0):
    """
    Nx: Is the number of samples you want
    xdim: Is the dimension of the output
    Thus the effective dimension of the input is about xdim/Nlp
    """
    y = random.uniform(low, high, [Nx, xdim])
    return y.clip(0, 1)