"""
Utility functions for pytorch_lightning.
It also includes useful functions for pure pytorch as well!
"""
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

import glob
import os
from functools import reduce

def default_scheduler(lr_scheduler):
    return {'scheduler':lr_scheduler, 'monitor':'val_checkpoint_on'}

def get_ckpt(dir):
    """
    Returns a ckpt file given a directory
    """
    return glob.glob(os.path.join(dir,'*.ckpt'))[0]

def get_parameters(modules):
    return reduce(lambda a, b: a + b, [list(x.parameters()) for x in modules])

#用于将原有目录替换为当前目录
def get_dirname(): 
    if os.name == "nt":
        current_dir = os.getcwd().replace("C:\\Users\\to232\\Dropbox\\nonlinear_NN_data\\", "")
        current_dir = current_dir.replace("\\", "--")

    if os.name == "posix":
        current_dir = os.getcwd()
        replace_dir = os.path.join(os.environ['HOME'], "Dropbox", "nonlinear_NN_data", "")
        current_dir = current_dir.replace(replace_dir, "")
        current_dir = current_dir.replace("/", "--")
    return current_dir

def test_plmodel(plmodel, data_loader, Nrepeat=4):
    """
    Function that is helpful for evaluating a model multiple times in the presence of noise! 
    """
    accuracies = []
    losses = []
    for i in range(Nrepeat):
        for batch in data_loader:
            batch = [i.to(plmodel.device) for i in batch]
            with torch.no_grad():
                out = plmodel.validation_step(batch, 0)
            accuracies.append(out['val_accu'])
            losses.append(out['val_loss'])
    return torch.stack(accuracies).cpu().numpy(), torch.stack(losses).cpu().numpy()

def relu_approx(x, factor=20.0):
    """
    A soft-relu function
    The default factor of 20.0 is descent for a turn on at around -0.1 
    """
    return F.softplus(x*factor)/factor

def clamp_lag(x, low=0.0, high=1.0, factor=20): 
    """
    Lagrangian loss term to clamp the value of x between low and high.
    To play around with this code do:
    xlist = torch.tensor(np.linspace(-1, 2, 100))
    ylist = [clamp_lag(x, low=0, high=1, factor=20) for x in xlist]
    plt.plot(xlist, ylist)
    plt.grid()
    """
    return torch.mean(relu_approx(-(x-low), factor) + relu_approx(x-high, factor))

def load_model_without_grad(file, *args, **kwargs):
    """
    Loads a module from a .p file and turns off the gradient in all the parameters of the model.
    """
    model = torch.load(file, *args, **kwargs)
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_logger(pname, name):
    """
    Return pytorch lightning loggers.
    pname refers to the "project name" (which hosts multiple runs), while name refers to the name of the run
    In particular, it returns both the csv and wandb loggers that will output the log of both files.
    The wandb name will append the dirname to make things more clear.
    """
    csv_logger = pl.loggers.CSVLogger('logs', pname, name)
    # wandb_pname = wandb_pname = f"{get_dirname()}--{pname}"  # the name that will appear in wandb
    # 改为纯pname，否则太长了
    wandb_pname = wandb_pname = f"pname"  # the name that will appear in wandb
    wandb_logger = pl.loggers.WandbLogger(name=name, project=wandb_pname)
    logger = [csv_logger, wandb_logger]
    return logger

def SilentTrainer(*args, **kwargs):
    """
    Silent Trainer just returns an instance of the pl.Trainer where the progress bar and other stuff has been silenced.
    FWIW here are some of the main parameters that are called in the Trainer instance
    trainer = SilentTrainer(max_epochs=max_epochs, logger=logger, gpus=[device],
                        checkpoint_callback=checkpoint_cb, 
                        log_save_interval=50, row_log_interval=4)
    """
    return pl.Trainer(*args, weights_summary=None, progress_bar_refresh_rate=0, **kwargs)
    
def CustomCheckpoint(log_dir):
    checkpoint_file = os.path.join(log_dir, "{epoch}-{val_loss:.5f}")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(checkpoint_file)
    return checkpoint_cb

def get_log_dir(pname, name):
    return os.path.join("logs", pname, name)

def custom_torch_load(file, **kwargs):
    """
    Loads the model via torch.load but also turns off all the gradients
    """
    model = torch.load(file, **kwargs)
    for param in model.parameters():
        param.requires_grad = True
    return model