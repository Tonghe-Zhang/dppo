import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
log = logging.getLogger(__name__)
from model.flow.mlp_flow import FlowMLP
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class ShortCutFlow(nn.Module):
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
                 batch_size=512,
                 bootstrap_every=4,
                 ):

        super().__init__()
        if int(max_denoising_steps) <=0:
            raise ValueError('max_denoising_steps must be positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.network = network.to(device)
        
        self.device = device
        
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.act_range=(act_min,act_max)
        
        self.obs_dim = obs_dim
        
        self.max_denoising_steps = int(max_denoising_steps)
        
        self.batch_size = batch_size
        self.bootstrap_every = bootstrap_every
        
        self.bootstrap_batchsize=self.batch_size//self.bootstrap_every
        # print(f"self.bootstrap_batchsize={self.bootstrap_batchsize}")
        
        
        self.flow_batchsize=self.batch_size-self.bootstrap_batchsize
        # print(f"self.flow_batchsize={self.flow_batchsize}")
        
        """
        for hopper, 
        self.bootstrap_batchsize=32
        self.flow_batchsize=96
        """  

    
    @torch.no_grad
    def sample(self, cond:dict, inference_steps:int, record_intermediate=False):
        ''' 
        use euclidean samples to integrate noise along the estimated velocity field, to obtain estimated action. 
        
        inputs:
            cond: dict, contataining...
                obs: torch.Tensor(batch_size, cond_steps, obs_dim)
            inference_steps: int
            record_intermediate: bool, whether to record intermediate actions.
        output:
            act: torch.Tensor(batch_size, horizon_steps, action_dim)  
        '''
        obs = cond['state']
        if inference_steps <=0:
            raise ValueError('inference_steps must be positive integer')
        inference_batch_size = obs.shape[0]
        
        # start from pure noise. 
        act= torch.randn(inference_batch_size,self.horizon_steps, self.action_dim).to(self.device)

        dt_base_value = np.log2(inference_steps).astype(np.int32)
        dt_base = torch.ones(inference_batch_size, device=self.device)*dt_base_value
        
        dt = torch.tensor(1.0 / inference_steps).to(self.device)     
        
        act_list = []
        for step in range(inference_steps):
            t = torch.ones(inference_batch_size,device=self.device)*(step / inference_steps)
            self.network: FlowMLP
            
            velocity = self.network(act, t, dt_base, obs, train=False)   # should we clamp here...? it was during training...
            
            act = act + velocity * dt
            
            if record_intermediate:
                act_list.append(act)
        
        act = torch.clamp(act,*self.act_range) # ensure the return value is within reasonable range. 
        return Sample(trajectories=act, chains = act_list if record_intermediate else None)
    
    def evaluate(self, act_real: Tensor, obs: Tensor, metric: torch.nn.Module = torch.nn.MSELoss()) -> Tuple[float, List[float]]:
        '''
        compare the generated images with the real images, by testing both flow-matching and shortcut results. 
        compare them using some metrics, currently we use the MSE between actions.
        
        inputs:
            act_real: torch.Tensor(batch_size, cond_steps, action_dim)
            obs: torch.Tensor(batch_size, cond_steps, obs_dim)
            metric: torch.nn.Loss
        outputs:
            one_step_error_midpoint: float
            N_step_error_from_pure_noise_list: list[float]
        '''
        one_step_error_midpoint = self.eval_one_step_from_midpoint(act_real, obs, metric)
        
        N_step_error_from_pure_noise_list = self.eval_generation_from_pure_noise_list(act_real, obs, metric)

        return one_step_error_midpoint, N_step_error_from_pure_noise_list

    
    def eval_one_step_from_midpoint(self,act_real:Tensor, obs:Tensor, metric=torch.nn.MSELoss())->float:
        '''
        test whether one-step generation can recover image from any noisy midpoint. 
        
        Here we start from noisy image with various noise mixture ratio t, convert noisy image to clean image in one step, 
        and then compare the synthetic action (image) with the real action (image). 
        All those actions (images) are generated in parallel.  
        '''
        act_noise = torch.randn(act_real.shape)
        
        dt_base = torch.arange(self.max_denoising_steps, device = self.device)   #[0,1,2,3,... ,7]
        dt = 2.0 ** (-dt_base)                          # [1,1/2,1/4,1/8,1/16,...,1/128]
        t = 1 - dt                                      # we start from the midpoint, which is only one step to t=1. 
        
        t_broadcasted = torch.ones_like(act_noise, device= self.device) * t
        dt_broadcasted = torch.ones_like(act_noise, device= self.device) * dt
        
        # create x_t, which is the real image corrupted at the last step, at t=1-dt.
        x_t = (1 - (1 - 1e-5) * t_broadcasted) * act_noise + t_broadcasted * act_real

        # predict the flow at the last steps
        v_pred = self.network(x_t, t, dt_base, obs)

        # take one integration step: x1 = xt + v dt
        act_pred_onestep = x_t + v_pred* dt_broadcasted
        
        act_pred_onestep = torch.clamp(act_pred_onestep,*self.act_range)
        
        # calculate the one-step error (MSE) between the predicted action and the real action.
        one_step_error = metric(act_real, act_pred_onestep)
        return one_step_error.item()
        
    
    def eval_generation_from_pure_noise(self,act_real,obs,metric)->List[float]:
        '''
        test whether generating image from pure noise in N steps faithfully recovers the real action (conditioned on the corresponding observation).
        '''        
        denoise_steps_list = [2**i for i in range(0,1+np.log2(self.max_denoising_steps).astype(np.int32))] #[1,2,4,8,16,32,...128]
        N_step_error_list=[0.0 for _ in range(len(denoise_steps_list))] 
        
        for i, denoising_steps in enumerate(denoise_steps_list):
            # generate in n steps. 
            dt_base = torch.arange(denoising_steps,device=self.device)
            dt = torch.tensor(1.0 / denoising_steps)
            dt_broadcast = dt.unsqueeze(-1).unsqueeze(-1).to(self.device) 
            
            # start from noise. 
            act = torch.randn(act_real.shape)
            for stepN in range(denoising_steps):
                t = torch.Tensor([stepN / denoising_steps], device=self.device)
                velocity_pred = self.network(act, t, dt_base, obs, train=False)
                act = act +  velocity_pred* dt_broadcast
            
            act=torch.clamp(act, *self.act_range)
            
            N_step_error_list[i] = metric(act_real,act).item()
            
        return N_step_error_list
    
    def compute_loss(self, actions:Tensor, observations:Tensor):
        '''
        inputs:
        actions: torch.Tensor(batch_size, horizon_steps, action_dim)
        observations: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        outputs:
        loss: torch.Tensor(1)
        info: dict containing loss and other useful information for logging
        '''
        
        # create supervision
        x_t, v_t, t, dt_base, obs = self.get_bootstrap_flow_targets(act=actions, obs=observations)
        
        # forward pass, get the velocity estimate for both bootstrapping (dt>0) and flow-matching (dt=0)
        v_prime = self.network(x_t, t, dt_base, obs, train=True)
        
        # calculate the bootstrapping loss and the flow-matching loss separately, for logging purpose. 
        loss_bootstrap=F.mse_loss(v_prime[:self.bootstrap_batchsize],v_t[:self.bootstrap_batchsize])
        
        loss_flow=F.mse_loss(v_prime[self.bootstrap_batchsize:],v_t[self.bootstrap_batchsize:])

        # the real loss function is the mse in bootstrapping and flow-matching, and we optimize these two together.     
        loss = F.mse_loss(v_prime, v_t, reduce='mean')

        info = {
            'loss': loss,
            'loss_bootstrap': loss_bootstrap,
            'loss_flow': loss_flow,
            'v_t_magnitude': torch.square(torch.mean(v_t**2)),
            'v_prime_magnitude': torch.square(torch.mean(v_prime**2)),
        }
        
        return loss, info

    def get_bootstrap_flow_targets(self, act, obs):
        '''
        inputs:
        act: torch.Tensor(batch_size, horizon_steps, action_dim)
        obs: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        returns:
        act_t_sample, v_t_sample, t_sample, dt_base_sample, obs_sample
        '''
        # Generate Bootstrapping targets
        act_t_bs,t_bs,dt_base_bs, vt_bs, obs_bs = self.get_bootstrap_target(act,obs)
        
        # Generate Flow-Matching Targets
        act_t_flow, t_flow, dt_base_flow, vt_flow, obs_flow = self.get_flow_target(act,obs)
        
        # Merge Flow with Bootstrap 
        act_t_sample=torch.cat([act_t_bs,act_t_flow])
        t_sample=torch.cat([t_bs,t_flow])
        dt_base_sample=torch.cat([dt_base_bs,dt_base_flow])
        v_t_sample=torch.cat([vt_bs,vt_flow])
        obs_sample=torch.cat([obs_bs,obs_flow])
        
        return act_t_sample, v_t_sample, t_sample, dt_base_sample, obs_sample
    
    def get_flow_target(self, act:Tensor,obs:Tensor):
        '''
        Generate supervision for the flow. The flow-matching target is the direction from noise to signal. x1-x0
        
        inputs:
        act: torch.Tensor(batch_size, horizon_steps, action_dim)
        obs: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        outputs: 
        act_t_flow:   torch.Tensor(flow_batchsize, horizon_steps, action_dim)
        t_flow:       torch.Tensor(flow_batchsize)
        dt_base_flow: torch.Tensor(flow_batchsize)
        vt_flow:      torch.Tensor(flow_batchsize, horizon_steps, action_dim)
        oflow_flow:   torch.Tensor(flow_batchsize, cond_steps, oflow_dim)
        
        '''
        # 4) =========== Generate Flow-Matching Targets ============
        # 4.0) Sample t for flow-matching
        t_flow=torch.randint(low=0, high=self.max_denoising_steps, size=(self.flow_batchsize,), dtype=torch.float32, device=self.device)
        t_flow /= self.max_denoising_steps
        t_broadcasted = t_flow.unsqueeze(-1).unsqueeze(-1)

        # 4.1) Sample flow pairs, define the flow target as the direction from noise to signal. 
        obs_flow = obs[:self.flow_batchsize]
        act_1_flow = act[:self.flow_batchsize] #bs is for `bootstrap'
        act_0_flow = torch.randn(act_1_flow.shape, device=self.device)
        
        # print("Shape of t_broadcasted:", t_broadcasted.shape)
        # print("Shape of act_0_flow:", act_0_flow.shape)
        # print("Shape of act_1_flow:", act_1_flow.shape)

        act_t_flow = (1 - (1 - 1e-5) * t_broadcasted) * act_0_flow + t_broadcasted * act_1_flow   
        
        vt_flow = act_1_flow - (1 - 1e-5) * act_0_flow
        
        dt_flow = np.log2(self.max_denoising_steps).astype(np.int32)
        dt_base_flow = torch.ones(self.flow_batchsize, dtype=torch.int32, device=self.device) * dt_flow
        
        return act_t_flow, t_flow, dt_base_flow, vt_flow, obs_flow
        

    def get_bootstrap_target(self, act:Tensor,obs:Tensor):
        '''
        Generate supervision for the self-distillation of one-step flow via bootstrapping.  
        If you take two steps consecutively, it should be equivalent to taking a larger stride with the magnitude of two steps. 
        
        (i) s(x_t, t, m) * dt +  s(x_t', t+dt, m)* dt  =   s(x_t, t, m/2) * 2dt
            where x_t' = x_t + s(x_t, t, m) * dt, 
        
        or equivalently, 
        
        (ii) 1/2 [ v(a_t, t, log 2m) * dt +  v(a_t', t+dt/2, log 2m)* dt ]   =   v(a_t, t, log m) * dt
        
             where a_t' = a_t + v(a_t, t, 2m) * dt/2, and the LHS is the bootstrap target. 
             
        in (ii), 
            log m is the logarithm of the number of denoising steps when we take a larger stride, and we name it `dt_base`
            dt is the euclidean time interval, 
            a_t is the actions(images), the velocity may also condition on the observation (lables)
            t is the time or noise mixture ratio. 
        
        inputs:
        act: torch.Tensor(batch_size, horizon_steps, action_dim)
        obs: torch.Tensor(batch_size, cond_steps, obs_dim)
        
        outputs:
        act_t_bs:   torch.Tensor(bootstrap_batchsize, horizon_steps, action_dim)
        t_bs:       torch.Tensor(bootstrap_batchsize)
        dt_base_bs: torch.Tensor(bootstrap_batchsize)
        vt_bs:      torch.Tensor(bootstrap_batchsize, horizon_steps, action_dim)
        obs_bs:     torch.Tensor(bootstrap_batchsize, cond_steps, obs_dim)
        '''
        
        log2_sections=np.log2(self.max_denoising_steps).astype(np.int32)
        dt_base_bs = (log2_sections - 1 - torch.arange(log2_sections)).repeat_interleave(self.bootstrap_batchsize // log2_sections)
        dt_base_bs = torch.cat([dt_base_bs, torch.zeros(self.bootstrap_batchsize - dt_base_bs.shape[0])]).to(self.device)
        dt_sections = torch.pow(2, dt_base_bs)
        num_dt_cfg = self.bootstrap_batchsize // log2_sections
        
        # 1) =========== Sample dt for bootstrapping ============
        dt = 1 / (2 ** (dt_base_bs))
        dt_base_bootstrap = dt_base_bs + 1
        dt_bootstrap = dt / 2
        
        # 2) =========== Generate Bootstrap Targets ============
        # 2.0) sample time       
        t_bs = torch.empty(self.bootstrap_batchsize, dtype=torch.float32, device=self.device)
        for i in range(self.bootstrap_batchsize):
            high_value = int(dt_sections[i].item())  # Convert to int since dt_sections contains float values
            t_bs[i] = torch.randint(low=0, high=high_value, size=(1,), dtype=torch.float32).item()
        
        t_bs /= dt_sections
        # print(t_bs)
        # print(t_bs.shape)

        t_broadcasted=t_bs.unsqueeze(-1).unsqueeze(-1)
        
        # 2.1) create the noise-corrupted action, act_t
        obs_bs = obs[:self.bootstrap_batchsize]
        act_1_bs = act[:self.bootstrap_batchsize] #bs is for `bootstrap'
        act_0 = torch.randn(act_1_bs.shape, device=self.device)
        
        
        
        act_t_bs = (1 - (1 - 1e-5) * t_broadcasted) * act_0 + t_broadcasted * act_1_bs
        
        # 2.2) do integration twice to bootstrap the result of integrating once, to get v_bs.
        
        # print(f"self.device={self.device}")
        # print(f"t_bs.device={t_bs.device}")
        # print(f"act_bs.device={act_t_bs.device}")
        # print(f"dt_base_bootstrap.device={dt_base_bootstrap.device}")
        # print(f"obs_bs.device={obs_bs.device}")
        
        v_b1 = self.network(act_t_bs, t_bs, dt_base_bootstrap, obs_bs, train=False)
        t2 = t_bs + dt_bootstrap
        
        act_t2 = act_t_bs + dt_bootstrap.unsqueeze(-1).unsqueeze(-1) * v_b1
        act_t2 = torch.clamp(act_t2, *self.act_range)
        
        v_b2 = self.network(act_t2, t2, dt_base_bootstrap, obs_bs, train=False)
        v_bs = (v_b1 + v_b2) / 2
        vt_bs=torch.clamp(v_bs, *self.act_range)   # `t` is for target. 
        
        
        # print(f"M={self.max_denoising_steps}")
        # print(f"dt_base_bs={dt_base_bs}")
        # print(f"dt_sections={dt_sections}")

        return act_t_bs,t_bs, dt_base_bs, vt_bs, obs_bs