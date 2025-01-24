import logging
import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple, List
log = logging.getLogger(__name__)
from model.flow.reflow import ReFlow
from collections import namedtuple
from torch.distributions.normal import Normal

Sample = namedtuple("Sample", "trajectories chains")

class PPOFlow(nn.Module):
    def __init__(self, 
                 device,
                 actor,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef=None
                 ):
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps      # number of steps for inference.
        self.ft_denoising_steps = inference_steps   # could be adjusted
        
        self.actor_old = actor
        self.load_actor(actor_policy_path)
        self.actor_old.to(self.device)
        for param in self.actor_old.parameters():
            param.requires_grad = False             # don't train this copy, just use it to load checkpoint. 
        
        ###############################################################
        self.actor_ft = copy.deepcopy(self.actor_old).to(self.device)
        for param in self.actor_ft.parameters():
            param.requires_grad = True
        logging.info("Cloned model for fine-tuning")
        
        self.critic = critic
        self.critic.to(self.device)
        ###############################################################
        
        self.report_network_params()
        
        
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        self.noise_scheduler_type = noise_scheduler_type
        # we can make noise scheduler learnable. 
        # also we should makes the scheduler return noise that is clipped. 

        self.randn_clip_value = randn_clip_value
        # prevent extreme values sampled from gaussian. deviation from mean should stay within `randn_clip_value` times of std.
        
        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std
        
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        
        self.set_logprob_noise_levels()
        
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )
    
    def load_actor(self, network_path, use_ema=False):
        if network_path:
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
            if not use_ema:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            else: 
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
    
    def stochastic_interpolate(self,t, device='cpu'):
        if self.noise_scheduler_type == 'vp':
            a = 0.2 #2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.noise_scheduler_type == 'lin':
            k=0.2
            b=0.0
            std = k*t+b
        else:
            raise NotImplementedError
        
        return std.to(self.device) if device != 'cpu' else std
    
    @torch.no_grad()
    def set_logprob_noise_levels(self):
        # we use this function to calculate log probabilities faster. 
        
        self.logprob_noise_levels = torch.zeros(self.inference_steps, device=self.device)
        
        steps = torch.linspace(0, 1, self.inference_steps)
        
        for i, t in enumerate(steps):
            self.logprob_noise_levels[i] = self.stochastic_interpolate(t)

        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std)

    
    def get_logprobs(self, cond:dict, x_chain:Tensor, get_entropy =False, normalize_time_horizon=False, normalize_dimension=False):
        '''
        inputs:
            x_chain: torch.Tensor of shape `[batchsize, self.inference_steps+1, self.horizon_steps, self.act_dim]`
           
        outputs:
            log_prob. tensor of shape `[batchsize]`
            entropy_rate_est: tensor of shape `[batchsize]
            
        explanation:
            p(x0|s)       = N(x0|0, 1)
            p(xt+1|xt, s) = N(xt+1 | xt + v(xt, s)1/K; sigma_t^2)
            
            log p(xK|s) = log p(x0) + \sum_{t=0}^{K-1} log p(xt+1|xt, s)
            H(X0:K)     = H(x0|s)     + \sum_{t=0}^{K-1} H(Xt+1|X_t, s)
            entropy rate H(X) = H(X0:K)/(K+1) asymptotically converges to the entropy per symbol when K goes to infinity.
            we view the actions at each dimension and horizon as conditionally independent on the state s and previous action. 
        '''        
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1,:, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        
        # initial probability
        init_dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:,0].reshape(B,-1)).sum(-1)   # [batchsize]
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)                          # [batchsize]
        
        # transition probabilities
        chains_vel  = torch.zeros_like(chains_prev, device=self.device)         # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        steps = torch.linspace(0, 1, self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, self.inference_steps]
        dt = 1.0/self.inference_steps
        for i in range(self.inference_steps):
            t = steps[:,i]
            xt = x_chain[:,i]                                                   # [batchsize, self.horizon_steps , self.act_dim]
            vt=self.actor_ft(xt, t, cond)                                       # [batchsize, self.horizon_steps, self.act_dim]
            chains_vel[:,i] = vt.flatten(-2,-1)                                 # [batchsize, self.horizon_steps x self.act_dim]
        
        chains_mean = chains_prev + chains_vel * dt                             # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        chains_stds = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.act_dim_total)  # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        chains_next = x_chain[:, 1:, :, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        
        chains_dist = Normal(chains_mean, chains_stds)
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)               # [batchsize, self.inference_steps] sum up self.horizon_steps x self.act_dim 
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)                       # [batchsize, self.inference_steps] Sum over all dimensions
        
        # logprobability of the markov chain.
        logprob = logprob_init + logprob_trans.sum(-1)                          # [batchsize] accumulate over inference steps (Markov property)
        
        # entropy rate estimate of the markov chain
        if get_entropy:
            entropy_rate_est = (entropy_init + entropy_trans.sum(-1))/(self.inference_steps + 1) # [batchsize] 
        
        if normalize_time_horizon:
            logprob = logprob / (self.inference_steps+1)
        if normalize_dimension:
            return logprob/self.act_dim_total, entropy_rate_est/self.act_dim_total if get_entropy else None
        return logprob, entropy_rate_est if get_entropy else None
    

    @torch.no_grad()
    def sample_first_point(self, B:int):
        '''
        B: batchsize
        '''
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        xt= dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)                    # mean() or sum() 
        xt=xt.reshape(B,self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob
    
    @torch.no_grad()
    def get_actions(self, cond:dict, eval_mode:bool, save_chains=False, normalize_time_horizon=False, normalize_dimension=False):
        '''
        inputs:
            cond: dict, contatinin...
                'state': obs. observation in robotics. torch.Tensor(batchsize, cond_steps, obs_dim)
            deterministic: bool, whether use deterministic inference or stochastic interpolate
            inference_steps: number of denoising steps in a single generation. 
            save_chains: whether to return predictions at each step
        outputs:
            if save_chains: (x_hat, x_hat_list, logprob)
            else: (x_hat, logprob)
            here, 
                xt. tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
                x_chains. tensor of shape `[self.inference_steps +1 ,self.data_shape]`: x0, x1, x2, ... xK
                logprob. tensor of shape `[batchsize]` or None
        '''
        
        # when doing deterministic sampling should calculate logprob again.
        B=cond["state"].shape[0]
        dt = (1/self.inference_steps)* torch.ones(B, self.horizon_steps, self.action_dim, device=self.device)
        steps = torch.linspace(0,1,self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, num_steps]
        if save_chains:
            x_chain=torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
        
        # sample first point
        xt, log_prob = self.sample_first_point(B)
        if save_chains:
            x_chain[:, 0] = xt
        
        for i in range(self.inference_steps):
            t = steps[:,i]
            vt=self.actor_ft(xt, t, cond)
            xt += vt* dt
            
            # add noise during training
            std = self.stochastic_interpolate(t).unsqueeze(-1).unsqueeze(-1).repeat(1, *xt.shape[1:])
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt, std)
            if not eval_mode:
                xt = dist.sample().clamp_(dist.loc - self.randn_clip_value * dist.scale,
                                          dist.loc + self.randn_clip_value * dist.scale).to(self.device)
            
            # prevent last action overflow
            if i == self.inference_steps-1:
                xt = xt.clamp_(self.act_min, self.act_max)                      
            
            # compute logprobs for eval or train
            log_prob += dist.log_prob(xt).sum(dim=(-2,-1)).to(self.device)

            if save_chains:
                x_chain[:, i+1] = xt
        
        if normalize_time_horizon:
            log_prob = log_prob/(self.inference_steps+1)
        if normalize_dimension:
            log_prob = log_prob/self.act_dim_total
        return (xt, x_chain, log_prob) if save_chains else (xt, log_prob)
    
    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        normalize_time_horizon=False,
        normalize_dimension=False
    ):
        """
        PPO loss
        obs: dict with key state/rgb; more recent obs at the end
            "state": (B, To, Do)
            "rgb": (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B,)
        advantages: (B,)
        oldlogprobs: (B,)
        use_bc_loss: whether to add BC regularization loss
        normalize_dimension: whether to normalize logprobs and entropy rates over all horiton steps and action dimensions
        reward_horizon: action horizon that backpropagates gradient, omitted for now.
        Here, B = n_steps x n_envs
        """
        
        newlogprobs, entropy= self.get_logprobs(obs, chains, get_entropy=True, normalize_time_horizon=normalize_time_horizon,normalize_dimension=normalize_dimension)
        
        log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std={newlogprobs.std():5.3f}")
        log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std={newlogprobs.std():5.3f}")
        
        # newlogprobs = newlogprobs.clamp(min=-5, max=2)
        # oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
        
        # batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # Get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use. 
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # bc loss
        bc_loss = 0.0
        if use_bc_loss:
            raise NotImplementedError
    
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item()
        )
