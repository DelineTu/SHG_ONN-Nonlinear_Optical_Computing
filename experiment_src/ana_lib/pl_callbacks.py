import os
import torch.nn.functional as F

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.metrics import Accuracy

from .torch_loader import *

accu_metric = Accuracy()

class TestCallback(Callback):
    """
    Function that tests the current code and saves the data in such a way that it will
    be saved in the future. 
    # IMPORTANT: transfer parameters from ref_model to test_model 
    """
    def __init__(self, test_epoch, test_loader, test_model, ref_model, savedir,
                     Nrepeat=1, test_name="test", ref_name="ref"):
        self.test_epoch = test_epoch #this is testing every test_epoch epoch
        self.test_model = test_model
        self.test_loader = test_loader
        self.ref_model = ref_model
        self.Nrepeat = Nrepeat
        self.savedir = savedir
        self.test_name = test_name
        self.ref_name = ref_name
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        test_epoch = self.test_epoch
        test_loader = self.test_loader
        test_model = self.test_model
        ref_model = self.ref_model
        Nrepeat = self.Nrepeat
        savedir = self.savedir
        
        epoch = trainer.current_epoch
        state_dict = ref_model.state_dict()
        fname = os.path.join(savedir, f"epoch_{epoch}.p")
        
        if epoch % test_epoch == 0:
            # IMPORTANT: transfer parameters from ref_model to test_model 
            test_model.load_state_dict(state_dict) 
            x, y = next(iter(test_loader))
            x = x.to("cuda") 
            y = y.to("cuda")
            x = x.repeat(Nrepeat, 1)
            y = y.repeat(Nrepeat)
            
            test_y_hat = test_model(x)
            test_y_pred = torch.max(test_y_hat, 1)[1]
            test_accu = accu_metric(test_y_pred, y)
            test_loss = F.cross_entropy(test_y_hat, y)

            ref_y_hat = ref_model(x)
            ref_y_pred = torch.max(ref_y_hat, 1)[1]
            ref_accu = accu_metric(ref_y_pred, y)
            ref_loss = F.cross_entropy(ref_y_hat, y)
            
            metrics = dict(epoch=epoch)
            metrics[self.test_name+"_accu"] = test_accu
            metrics[self.test_name+"_loss"] = test_loss
            metrics[self.ref_name+"_accu"] = ref_accu
            metrics[self.ref_name+"_loss"] = ref_loss
            
            if not hasattr(ref_model, "save_dict"):
                ref_model.save_dict = dict()
            if not hasattr(test_model, "save_dict"):
                test_model.save_dict = dict()
            
            save_dict = dict(state_dict=state_dict, metrics=metrics,
                             x = x, y=y,
                            ref_save_dict=ref_model.save_dict,
                            test_save_dict=test_model.save_dict
                            )
            save_data = torch.save(save_dict, fname)
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

class RetrainCallback(Callback):
    """
    rt stands for retrain in this code!
    in most cases of course this stands for the digital twin
    """
    def __init__(self, rt_epoch, rt_xlist, rt_trainer, rt_model, f_exp, name="dt_error"):
        super().__init__()
        self.rt_epoch = rt_epoch
        self.rt_xlist = rt_xlist
        self.rt_trainer = rt_trainer
        self.rt_model = rt_model
        self.name = name
        self.f_exp = f_exp

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        epoch = trainer.current_epoch
        rt_epoch = self.rt_epoch
        rt_xlist = self.rt_xlist
        rt_trainer = self.rt_trainer
        rt_model = self.rt_model
        
        if epoch % rt_epoch == 0:
            self.f_exp(rt_xlist)
            rt_train_loader, rt_val_loader = np2loaders(rt_xlist, self.f_exp.y_np, 
                                              train_ratio=0.8, Nbatch=100)
            rt_trainer.fit(rt_model, rt_train_loader, rt_val_loader)
            result = rt_trainer.test(rt_model, rt_val_loader)
            dt_error = result[0]['test_loss']
            metrics = dict(epoch=epoch)
            metrics[self.name] = dt_error
            trainer.logger.log_metrics(metrics, step=trainer.global_step)