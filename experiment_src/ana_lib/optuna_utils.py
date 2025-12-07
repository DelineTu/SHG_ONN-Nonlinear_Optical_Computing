import numpy as np
import os
from IPython.core.debugger import set_trace

import torch
import optuna
import pytorch_lightning as pl
import shutil

def create_study(NAS_name, sampler=optuna.samplers.TPESampler(), direction="minimize"):
    pruner = optuna.pruners.NopPruner()
    storage = f'sqlite:///{NAS_name}.db' #way to specify an SQL database
    study = optuna.create_study(pruner=pruner, sampler=sampler, 
            storage=storage, study_name="", load_if_exists=True, direction=direction)
    return study

def delete_study(NAS_name):
    try:
        os.remove(f"{NAS_name}.db")
    except:
        print(f"{NAS_name}.db don't exist")
        
#Maybe in here I should have some commented code for how I have used the NAS code?