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
                 sample_t_type = 'uniform'
                 ):
        super().__init__()
        if int(max_denoising_steps) <=0:
            raise ValueError('max_denoising_steps must be positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.network = network.to(device)
        # TODO: load exising model from network_path ${base_policy_path}
        
        self.device = device
        
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        
        self.act_range=(act_min,act_max)
        
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.batch_size = batch_size
        
        self.sample_t_type = sample_t_type
        """
        for hopper, 
        self.bootstrap_batchsize=32
        self.flow_batchsize=96
        """
    def generate_trajectory(self,x1:Tensor, x0:Tensor, t:Tensor):
        """
        generate rectified flow trajectory xt= t x1+(1-t) x0
        """
        t_broadcast=(torch.ones_like(x1, device=self.device) * t.view(self.batch_size, 1, 1)).to(self.device)
        
        xt= t_broadcast* x1 + (1-t_broadcast)* x0
        
        return xt
    
    def sample_time(self, time_sample_type='uniform', **kwargs):
        """
        generate samples of a distribution in [0, 1)
        """
        supported_time_sample_type =['uniform', 'logitnormal', 'beta']
        if time_sample_type=='uniform':
            return torch.rand(self.batch_size, device=self.device)
        elif time_sample_type=='logitnormal':
            m = kwargs.get("m", 0)  # Default mean is 0
            s = kwargs.get("s", 1)  # Default standard deviation is 1
            normal_samples = torch.normal(mean=m, std=s, size=(self.batch_size,), device=self.device)# Generate normal samples
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)# Apply the logistic function
            return logit_normal_samples
        elif time_sample_type=='beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha parameter
            beta = kwargs.get("beta", 1.0)    # Default beta parameter
            s = kwargs.get("s", 0.999)  # Default cutoff deviation is 0.999
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((self.batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(f'Unknown time_sample_type = {time_sample_type}. We only support {supported_time_sample_type}')
        
    def generate_target(self, x1:Tensor):
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
        
        # random time, or mixture ratio between (0,1). different for each sample, but he same for each channel. 
        t=self.sample_time(self.sample_t_type)
        
        # generate random noise
        x0=torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        
        # generate corrupted data
        xt = self.generate_trajectory(x1, x0, t)
        
        # generate target
        v =x1-x0
        
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
            if `record_intermediate` is False, xt. tensor of shape `self.data_shape`
            if `record_intermediate` is True,  xt_list. tensor of shape `[num_steps,self.data_shape]`
        '''
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
            if record_intermediate:
                x_hat_list[i] = x_hat
        
        if record_intermediate:
            return Sample(trajectories=x_hat, chains=x_hat_list)
        return Sample(trajectories=x_hat, chains=None)