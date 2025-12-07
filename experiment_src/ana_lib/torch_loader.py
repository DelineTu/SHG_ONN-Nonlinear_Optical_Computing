"""
Simple functions for performing basic data wrangling of the spectrums
"""
import numpy as np
import torch
import torch.utils.data as data

#x: 输入数据，NumPy 数组格式，第一维为批处理维度。
#y: 输出数据，NumPy 数组格式，第一维为批处理维度。
#train_ratio: 训练集占总数据集的比例，默认值为 0.9。
#Nbatch: 训练集的批处理大小，默认值为 100。测试集的批处理大小为整个测试集的大小。
def np2loaders(x, y, train_ratio=0.9, Nbatch = 100):
    """
    Produces the training and testing pytorch dataloaders given numpy inputs
    x: input data in numpy format. First dimension is batch dimension
    y: output data in numpy format. First dimension is batch dimension
    train_ratio: The ratio of dataset that will be training data
    Nbatch: batchsize for the training set
    Note the testset has a batchsize of the whole training set
    """
    Ntotal = x.shape[0]
    Ntrain = int(np.floor(Ntotal*train_ratio))
    train_inds = np.arange(Ntrain)
    val_inds = np.arange(Ntrain, Ntotal)

    X_train = torch.tensor(x[train_inds]).float()
    X_val = torch.tensor(x[val_inds]).float()

    Y_train = torch.tensor(y[train_inds]).float()
    Y_val = torch.tensor(y[val_inds]).float()

    train_dataset = data.TensorDataset(X_train, Y_train)
    val_dataset = data.TensorDataset(X_val, Y_val)

    train_loader = data.DataLoader(train_dataset, Nbatch)
    val_loader = data.DataLoader(val_dataset, val_dataset.tensors[0].shape[0])

    return train_loader, val_loader

def load_loaders(file, num_labels, nums, train_batch_size, 
                 val_batch_size=None, test_batch_size=None, seed=0):
    train_num, val_num, test_num = nums
    torch.manual_seed(seed)
    data_dict = torch.load(file)
    x_tensor = data_dict['x_tensor']
    y_tensor = data_dict['y_tensor']

    target_list = list(range(7))

    x_tensor_list = []
    y_tensor_list = []

    for target in target_list:
        mask = y_tensor == target
        x_tensor_list.append(x_tensor[mask])
        y_tensor_list.append(y_tensor[mask])

    if val_batch_size is None:
        val_batch_size = val_num*num_labels
        test_batch_size = test_num*num_labels

    train_dataset_list = []
    val_dataset_list = []
    test_dataset_list = []

    x = x_tensor_list[0]
    y = y_tensor_list[0]
    for (x, y) in zip(x_tensor_list, y_tensor_list):
        dataset = data.TensorDataset(x, y)
        train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_num, val_num, test_num])
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
        test_dataset_list.append(test_dataset)

    train_dataset = data.ConcatDataset(train_dataset_list)
    val_dataset = data.ConcatDataset(val_dataset_list)
    test_dataset = data.ConcatDataset(test_dataset_list)

    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True) 
    test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True) 
    return train_loader, val_loader, test_loader