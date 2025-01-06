"""
Pre-training diffusion policy
"""
import logging
import wandb
import numpy as np
import time
import torch
log = logging.getLogger(__name__)
from util.timer import Timer, sec2HMS
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device
from model.flow.shortcut_flow import ShortCutFlow

class TrainShortCutFlowAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.parallel = cfg.train.get("parallel", 0)==1
        self.total_time =0
        
    def run(self):
        timer = Timer()
        self.epoch = 1
        
        start_time=time.time()

        if self.parallel and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model.to('cuda')
            model_wrapper = torch.nn.DataParallel(self.model)
            self.model=model_wrapper.module

        for epoch in range(self.n_epochs):            
            # train
            self.model.train()
            
            loss_train_epoch = []
            loss_train_bootstrap_epoch = []
            loss_train_flow_epoch = []
            v_t_magnitude_train_epoch=[]
            v_prime_magnitude_train_epoch=[]
            
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train, device="cuda:0")
                # print(f"self.dataset_train.device={self.dataset_train.device}")
                
                act, cond = batch_train
                
                self.optimizer.zero_grad()

                loss_train, train_info = self.model.compute_loss(actions=act, observations=cond["state"])
                
                loss_train.backward()
                
                self.optimizer.step()
                
                # logging information
                loss_train_epoch.append(loss_train.item())
                loss_train_bootstrap_epoch.append(train_info['loss_bootstrap'].item())
                loss_train_flow_epoch.append(train_info['loss_flow'].item())
                v_t_magnitude_train_epoch.append(train_info['v_t_magnitude'].item())
                v_prime_magnitude_train_epoch.append(train_info['v_prime_magnitude'].item())
            
            train_end_time=time.time()
            train_duration = train_end_time-start_time
            train_duration_step = train_duration/len(self.dataloader_train)
        
            # statistics during training.
            loss_train, loss_train_var, loss_train_bootstrap, loss_train_bootstrap_var, loss_train_flow, loss_train_flow_var, v_t_magnitude_train, \
                v_t_magnitude_train_var, v_prime_magnitude_train, v_prime_magnitude_train_var = self.statistics(loss_train_epoch,loss_train_bootstrap_epoch,loss_train_flow_epoch,v_t_magnitude_train_epoch,v_prime_magnitude_train_epoch)
            
            # validate
            self.model.eval()
            
            loss_val_epoch = []
            loss_val_bootstrap_epoch = []
            loss_val_flow_epoch = []
            v_t_magnitude_val_epoch=[]
            v_prime_magnitude_val_epoch=[]
            
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                # currently no validation step. 
                eval_start_time=time.time()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    
                    act, cond = batch_train
                    
                    # we evaluate the model on the eval ste by two metrics. 
                    # 1. the loss. Comparing the evaluation loss with the training loss tells us whether model overfits. 
                    loss_val, val_info = self.model.compute_loss(actions=act, observations=cond["state"])
                    
                    # 2. the difference between generated actions and the real actions. this tells us directly how well the model behaves on the eval set. 
                    # one_step_denoise_err tells you how well 
                    one_step_denoise_err, N_step_generationn_err= self.model.evaluate(act_real=act, obs=cond["state"],metric=torch.nn.MSELoss())
                    
                    # logging information
                    loss_val_epoch.append(loss_val.item())
                    loss_val_bootstrap_epoch.append(val_info['loss_bootstrap'].item())
                    loss_val_flow_epoch.append(val_info['loss_flow'].item())
                    v_t_magnitude_val_epoch.append(val_info['v_t_magnitude'].item())
                    v_prime_magnitude_val_epoch.append(val_info['v_prime_magnitude'].item())
                    
                self.model.train()
            
                eval_end_time=time.time()
                eval_duration = eval_end_time-eval_start_time
                eval_duration_step = eval_duration/len(self.dataloader_val)
            
            # statistics during validation.
            loss_val, loss_val_var, loss_val_bootstrap, loss_val_bootstrap_var, loss_val_flow, loss_val_flow_var, v_t_magnitude_val, \
                v_t_magnitude_val_var, v_prime_magnitude_val, v_prime_magnitude_val_var= self.statistics(loss_val_epoch,loss_val_bootstrap_epoch,loss_val_flow_epoch,v_t_magnitude_val_epoch,v_prime_magnitude_val_epoch)
            
            # update lr
            self.lr_scheduler.step()
            
            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
        
            end_time=time.time()
            epoch_duration = end_time-start_time
            start_time=end_time
            
            # log loss
            if self.epoch % self.log_freq == 0:
                self.log(locals())
            
            # count
            self.epoch += 1
           
    
    def statistics(self,loss_epoch,loss_bootstrap_epoch,loss_flow_epoch,v_t_magnitude_epoch,v_prime_magnitude_epoch):
        # statistics during ing.
        loss = np.mean(loss_epoch)
        loss_var=np.var(loss_epoch)
        
        loss_bootstrap = np.mean(loss_bootstrap_epoch)
        loss_bootstrap_var=np.var(loss_bootstrap_epoch)
        
        loss_flow = np.mean(loss_flow_epoch)
        loss_flow_var=np.var(loss_flow_epoch)
        
        v_t_magnitude_ = np.mean(v_t_magnitude_epoch)
        v_t_magnitude_var = np.var(v_t_magnitude_epoch)
        
        v_prime_magnitude_ = np.mean(v_prime_magnitude_epoch)
        v_prime_magnitude_var = np.var(v_prime_magnitude_epoch)
        
        self.model: ShortCutFlow
        
        return loss, loss_var, loss_bootstrap, loss_bootstrap_var, loss_flow, loss_flow_var, v_t_magnitude_, v_t_magnitude_var, v_prime_magnitude_, v_prime_magnitude_var
        
    def log(self, locs, width=80, pad=35):
        bootstrap_flow_batchsize_ratio = (self.model.bootstrap_batchsize/self.model.flow_batchsize)
        eval_time = locs["eval_duration"] if self.dataloader_val is not None else 0
        eval_time_per_step = locs["eval_duration_step"] if self.dataloader_val is not None else 0
        
        strs = f" \033[1m Learning iteration {self.epoch}/{self.n_epochs} \033[0m "
        log_string = (f"""\n{'#' * width}\n"""
               f"""{strs.center(width, ' ')}\n\n"""
               f"""{'Epoch:':>{pad}} {self.epoch}\n"""
               f"""{'Training time: ':>{pad}} {locs["train_duration"]:.4f} s, per batch: {locs["train_duration_step"]:.4f} s\n"""
               f"""{'Eval time: ':>{pad}} {eval_time:.4f} s, per batch: {eval_time_per_step:.4f}\n"""
               f"""{'Time per epoch: ':>{pad}} {locs["epoch_duration"]:.4f} s\n"""         
               f"""{'Estimated Remaining Time: ':>{pad}} {
                   sec2HMS(locs["epoch_duration"]*(self.n_epochs-locs["epoch"]))
                   }\n"""
               f"""{'Train loss:':>{pad}} {locs["loss_train"]:8.4f}\n"""
               f"""{'Train loss variance:':>{pad}} {locs["loss_train_var"]:8.4f}\n"""
               f"""{'Train bootstrap loss:':>{pad}} {locs["loss_train_bootstrap"]:8.4f}\n"""
               f"""{'Train bootstrap loss variance:':>{pad}} {locs["loss_train_bootstrap_var"]:8.4f}\n"""
               f"""{'Train flow loss:':>{pad}} {locs["loss_train_flow"]:8.4f}\n"""
               f"""{'Train flow loss variance:':>{pad}} {locs["loss_train_flow_var"]:8.4f}\n"""
               f"""{'Flow target amplitude:':>{pad}} {locs["v_t_magnitude_train"]:8.4f}\n"""
               f"""{'Flow target variance:':>{pad}} {locs["v_t_magnitude_train_var"]:8.4f}\n"""
               f"""{"Flow estimate amplitude:":>{pad}} {locs["v_prime_magnitude_train"]:8.4f}\n"""
               f"""{"Flow estimate variance:":>{pad}} {locs["v_prime_magnitude_train_var"]:8.4f}\n"""
               f"""{'Batchsize: bootstrap/flow:':>{pad}} {bootstrap_flow_batchsize_ratio:.4f}\n"""
        )
        if self.dataloader_val is not None:
            log_string.append(f"""{'One step denoise error:':>{pad}} {locs["one_step_denoise_err"]:8.4f}\n""")
            for i, err in enumerate(locs["N_step_generation_err"]):
                log_string.append(f"""{'N step reconstruction error:':>{pad}} {'{2**i} steps '} {err:8.4f}\n""")       
        
        log.info(log_string)
        if self.use_wandb:
            wandb.log(
                {
                    "loss - train": locs["loss_train"],
                    "loss - train variance": locs["loss_train_var"],
                    
                    "loss - train bootstrap": locs["loss_train_bootstrap"],
                    "loss - train bootstrap variance": locs["loss_train_bootstrap_var"],
                    
                    "loss - train flow": locs["loss_train_flow"],
                    "loss - train flow variance": locs["loss_train_flow_var"],
                    
                    "v_t_magnitude - train": locs["v_t_magnitude_train"],
                    "v_t_magnitude - train variance": locs["v_t_magnitude_train_var"],
                    
                    "v_prime_magnitude - train": locs["v_prime_magnitude_train"],
                    "v_prime_magnitude - train variance": locs["v_prime_magnitude_train_var"],
                },
                step=self.epoch,
                commit=True,
            )
            if self.dataloader_val is not None and locs["loss_val"] is not None:
                wandb.log(
                    {
                        "loss - val": locs["loss_val"],
                        "loss - val variance": locs["loss_val_var"],
                        
                        "loss - val bootstrap": locs["loss_val_bootstrap"],
                        "loss - val bootstrap variance": locs["loss_val_bootstrap_var"],
                        
                        "loss - val flow": locs["loss_val_flow"],
                        "loss - val flow variance": locs["loss_val_flow_var"],
                        
                        "v_t_magnitude - val": locs["v_t_magnitude_val"],
                        "v_t_magnitude - val variance": locs["v_t_magnitude_val_var"],
                        
                        "v_prime_magnitude - val": locs["v_prime_magnitude_val"],
                        "v_prime_magnitude - val variance": locs["v_prime_magnitude_val_var"],
                    },
                    step=self.epoch, 
                    commit=False
                )
            