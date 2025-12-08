"""
Provides base class for pytorch_lightning modules with training and validation steps.

NOTE: Recently, I changed the naming from xxxModel to xxxPlModel to be more consistent with the notation that was used in custom_plmodels.p
"""

import numpy as np
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
# import ipdb

class RegressionPlModel(pl.LightningModule):
    """
    Employs Mean Square Error loss to perform regression
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', torch.sqrt(loss))
        return result

class Classification1DPlModel(pl.LightningModule):
    """
    Classifies by matching the 1D real number output to the closest integer
    The model employs thus employs the mseloss 
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 
        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        result.log('train_accu', accuracy)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 

        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        result.log('val_accu', accuracy)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 

        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        return result
    
    
class Classification1DLagPlModel(pl.LightningModule):
    """
    Classifies by matching the 1D real number output to the closest integer
    The loss function has both an mseloss as well as a lagrangian term
    Thus for this particular module, you need to define the lagrangian function in the class!
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    @abc.abstractmethod
    def lagrangian(self):
        """
        A lagragian loss term that will be added to the loss function during backprop!
        If this method is not included, you cannot inherent from this class.
        """
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) + self.lagrangian()
        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        result.log('train_accu', accuracy)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) + self.lagrangian()

        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        result.log('val_accu', accuracy)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) + self.lagrangian()
        
        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        #don't log!
        return result
    
class ClassificationPlModel(pl.LightningModule):
    """
    Classifies the usual way where the cross entropy loss is used!
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y) 
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        result.log('train_accu', accuracy)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y) 
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        result.log('val_accu', accuracy)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y) 
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        return result
    
class ClassificationLagPlModel(pl.LightningModule):
    """
    Classifies via conventional ML approach!
    The loss function has both an cross entropy as well as a lagrangian term
    Thus for this particular module, you need to define the lagrangian function in the class!
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    @abc.abstractmethod
    def lagrangian(self):
        """
        A lagragian loss term that will be added to the loss function during backprop!
        If this method is not included, you cannot inherent from this class.
        """
        
    def _innerstep(self, batch):
        """
        Function that runs model, evaluate loss and accuracy
        """
        x, y = batch
        out = self(x)
        cr_loss = F.cross_entropy(out, y) 
        lag_loss = self.lagrangian() 
        loss = cr_loss + lag_loss
        
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)        
        return cr_loss, lag_loss, loss, accuracy
    
    def training_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.TrainResult(loss)
        result.log('train_cr_loss', cr_loss)
        result.log('train_lag_loss', lag_loss)
        result.log('train_loss', loss)
        result.log('train_accu', accuracy)
        
        if hasattr(self, 'log_dict'):
            for key, val in self.log_dict.items():
                result.log(key, val)
        
        return result
    
    def validation_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_cr_loss', cr_loss)
        result.log('val_lag_loss', lag_loss)
        result.log('val_loss', loss)
        result.log('val_accu', accuracy)
        

            
        return result
    
    def test_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        return result
    
    
    
class ClassificationLagPlModelCR(pl.LightningModule):
    """
    Classifies via conventional ML approach!
    The loss function has both an cross entropy as well as a lagrangian term
    Thus for this particular module, you need to define the lagrangian function in the class!
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    @abc.abstractmethod
    def lagrangian(self):
        """
        A lagragian loss term that will be added to the loss function during backprop!
        If this method is not included, you cannot inherent from this class.
        """
        
    def _innerstep(self, batch):
        """
        Function that runs model, evaluate loss and accuracy
        """
        x, y = batch
        out = self(x)
        #takes the log of the output..
        
        cr_loss = F.nll_loss(torch.log(out), y)
#         cr_loss = F.cross_entropy(out, y) 
        lag_loss = self.lagrangian() 
        loss = cr_loss + lag_loss
        
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)        
        return cr_loss, lag_loss, loss, accuracy
    
    def training_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.TrainResult(loss)
        result.log('train_cr_loss', cr_loss)
        result.log('train_lag_loss', lag_loss)
        result.log('train_loss', loss)
        result.log('train_accu', accuracy)
        
        if hasattr(self, 'log_dict'):
            for key, val in self.log_dict.items():
                result.log(key, val)
        
        return result
    
    def validation_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_cr_loss', cr_loss)
        result.log('val_lag_loss', lag_loss)
        result.log('val_loss', loss)
        result.log('val_accu', accuracy)
        

            
        return result
    
    def test_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        return result