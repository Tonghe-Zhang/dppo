import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
log = logging.getLogger(__name__)
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")

class ReFlow(nn.Module):
    def __init__(self, 
                 network_path, 
                 network,
                 device,
                 horizon_steps, 
                 action_dim, 
                 act_min,
                 act_max,
                 obs_dim, 
                 max_denoising_steps, 
                 seed,
                 batch_size,
                 ):
        super().__init__()
        if int(max_denoising_steps) <=0:
            raise ValueError('max_denoising_steps must be positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.network = network.to(device)
        ####################################################################################
        self.device = device
        
        self.network_path = network_path
        data = torch.load(self.network_path, weights_only=True, map_location=self.device)
        print(f"{data.keys()}")
        if 'actor' in data.keys():
            self.load_state_dict(data['actor'])
        elif 'model' in data.keys():
            self.load_state_dict(data['model'])
        else:
            raise ValueError(f'Unrecognized checkpoint. data.keys() including {data.keys()}. ')
        # epoch = data["epoch"]
        # self.load_state_dict(data["model"])    #
        # self.ema_model.load_state_dict(data["ema"])
        print(f"Loaded dict from {self.network_path}")
        

        self.act_horizon = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.act_horizon, self.action_dim)
        
        self.act_range=(act_min,act_max)
        
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.batch_size = batch_size
        
        
        self.show_inference_process = False
        """
        for hopper, 
        self.bootstrap_batchsize=32
        self.flow_batchsize=96
        """
    
    def generate_target(self, x1:Tensor, cond:dict):
        '''
        inputs:
            x1. tensor. real data. torch.Tensor(batch_size, horizon_steps, action_dim)
            cond. dict. containing...
                'state': observation in robotics: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        outputs: 
            (xt, t, obs): tuple, the inputs to the model. containing...
                t:       torch.Tensor(batchsize)
                x_t:     torch.Tensor(batchsize, horizon_steps, action_dim)
                obs:     torch.Tensor(batchsize, cond_steps, obs_dim)
            v:           torch.Tensor(batchsize, horizon_steps, action_dim)
        
        outputs:
            (xt, t, obs): tuple, the inputs to the model. containing...
                t:  corruption ratio        torch.Tensor(batchsize)
                xt: corrupted data. torch.  torch.Tensor(batchsize, horizon_steps, action_dim)
            v:  tensor. target velocity, from x0 to x1 (v=x1-x0). the desired output of the model. 
        '''
        
        
        obs =cond['state']
        
        # random time, or mixture ratio between (0,1). different for each sample, but he same for each channel. 
        t=torch.randn(self.batch_size,device=self.device)
        t_broadcast=(torch.ones_like(x1, device=self.device) * t.view(self.batch_size, 1, 1)).to(self.device)
        # generate random noise
        x0=torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        # generate corrupted data
        
        xt= t_broadcast* x1 + (1-t_broadcast)* x0
        # generate target
        v =x1-x0

        # print(f"xt.shape={xt.shape}")
        # print(f"t.shape={t.shape}")
        # print(f"obs.shape={obs.shape}")
        
        return (xt, t), v
    
    def loss(self, xt, t, obs, v):
        
        v_hat = self.network(xt, t, obs)
        
        loss = F.mse_loss(input=v_hat, target=v)
        
        return loss
    
    @torch.no_grad()
    def sample(self, cond:dict, inference_steps:int, inference_batch_size=None, record_intermediate=False):
        '''
        inputs:
            cond: dict, contatinin...
                'state': obs. observation in robotics. torch.Tensor(batchsize, cond_steps, obs_dim)
            num_step: number of denoising steps in a single generation. 
            record_intermediate: whether to return predictions at each step
        outputs:
            if `record_intermediate` is False, returns a Sample() namedtuple, 
            whose .trajectory attribute is a tensor of shape `(inference_batch_size, self.data_shape)`, 
            that is `(inference_batch_size, self.horizon_steps, self.action_dim)' for mujoco task. 
            if `record_intermediate` is True,  xt_list. tensor of shape `[num_steps,self.data_shape]`
        '''
        
        obs = cond['state']
        if inference_batch_size is None:
            inference_batch_size=self.batch_size
        
        if record_intermediate:
            x_hat_list=torch.zeros((inference_steps,)+self.data_shape)  # [num_steps,self.data_shape]
        
        x_hat=torch.randn((inference_batch_size,)+self.data_shape, device=self.device)    # [batchsize, Ta, Da]  Ta; action horizon, da: action dimension 
        
        dt = (1/inference_steps)* torch.ones_like(x_hat).to(self.device)
        
        steps = torch.linspace(0,1,inference_steps).repeat(inference_batch_size, 1).to(self.device)                       # [batchsize, num_steps]
        
        for i in range(inference_steps):
            t = steps[:,i]

            vt=self.network(x_hat,t,cond)
            
            x_hat+= vt* dt
            
            if self.show_inference_process:
                print(f"[infer]: {i}/{inference_steps}")
            if record_intermediate:
                x_hat_list[i] = x_hat
        
        if record_intermediate:
            return Sample(trajectories=x_hat, chains=x_hat_list)
        return Sample(trajectories=x_hat, chains=None)