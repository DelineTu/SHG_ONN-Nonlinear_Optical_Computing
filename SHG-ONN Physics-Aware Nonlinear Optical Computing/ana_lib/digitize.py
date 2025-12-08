"""
File contains all function pertaining to digitization. 
Some files that take care of quantizing output layers have also been added on Nov 20. 
"""

import torch
import numpy as np
import torch.nn.functional as F

#digitizex 函数用于将一维浮点数数组 data 下采样并转换为具有固定数量级别的二值化或离散化数组。这个过程包括插值、细化（增加采样点）和平均化。以下是该函数的详细解释：
#输入参数：
#data：原始的一维浮点数数组。
#xlevels：期望输出的数组长度，即最终的级别数。
#fine_factor：细化因子，默认为 20。这个因子决定了在每个目标级别之间插入多少个额外的点。
def digitizex(data, xlevels, fine_factor=20):
    #First do downsampling in x
    ###生成新的采样点：
    #创建一个新的采样点数组 x，其长度为 fine_factor * xlevels，均匀分布在原始数据索引范围内。
    x=np.linspace(0,len(data),fine_factor*xlevels)
    #插值
    #使用线性插值方法，根据原始数据 data 和其索引位置 np.arange(0, len(data))，计算新采样点 x 处的数据值，得到细化后的数组 datafine。
    datafine=np.interp(x,np.arange(0,len(data)),data)
    #now here get a new set of data that is mean over each thing
    ### 平均化
    #初始化输出数组：
    datad = np.zeros(xlevels)
    #计算每个级别的平均值：
    #这一步将细化后的数据 datafine 分成 xlevels 段，每段包含 fine_factor 个点，并对每段进行平均，结果存储在 datad 中。
    for i in range(xlevels):
        datad[i] = np.sum(datafine[i*fine_factor:(i+1)*fine_factor])/fine_factor
    #返回经过下采样和平均化的数组 datad，其长度为 xlevels。
    return datad

def digitizey(x, levels, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()
    x = (levels-1)*(x - min_val)/(max_val-min_val)
    x = x.round()
    x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)*(levels-1))
    x = x*(max_val-min_val)/(levels-1) + min_val
    return x

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels=2**4, min_val=None, max_val=None):
        x = digitizey(x, levels=levels, min_val=min_val, max_val=max_val)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

pt_digitize = FakeQuantOp.apply #Define the function that can be used outside

def torch_linspace(left, right, N, device):
    """
    Return a linspace with left and right being tensors with autograd.
    Passing in the device makes it possible for left and right to be floats!
    """
    return left + (right-left)*torch.linspace(0, 1, N, device=device)

def mean_downsampling(out, Ncompress):
    out_reshape = out.reshape(out.shape[0], out.shape[1]//Ncompress, Ncompress)
    out_compressed = out_reshape.mean(dim=-1)
    return out_compressed

def torch_interp_linear(x_dat, y_dat, x_int):
    """
    x_dat, y_dat is the data to be interpolated
    x_int are the points that you would like to interpolate on
    x_dat and x_int are assumed to be 1D tensors
    while y_dat is a 2D tensor, where the first index is the batch index
    """
    xdiff = x_dat.repeat(len(x_int), 1).T - x_int
    xdiff[xdiff > 0] = -100000000.0
    xind = xdiff.argmax(dim=0)
    x0 = x_dat[xind]
    x1 = x_dat[xind+1]
    y0 = y_dat[:, xind]
    y1 = y_dat[:, xind+1]
    y = y0 + (x_int-x0)*(y1-y0)/(x1-x0)
    return y

def torch_interp_nearest(x_dat, y_dat, x_int):
    """
    x_dat, y_dat is the data to be interpolated
    x_int are the points that you would like to interpolate on
    
    x_dat and x_int are assumed to be 1D tensors
    while y_dat is a 2D tensor, where the first index is the batch index
    """
    xdiff = torch.abs(x_dat.repeat(len(x_int), 1).T - x_int)
    xind = (xdiff.argmin(dim=0)).detach()
    
    y = y_dat[:, xind]
    return y

def haha():
    print("asdfasdf")
    
def haha2():
    print("asdfasdf")