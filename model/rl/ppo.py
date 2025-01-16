
# from huazhe's class
# 
from functools import partial
import os
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
# from agent.ddpg import DDPGAgent
from copy import deepcopy
from model.rl.rl_buffers import PPOReplayBuffer
import logging
logger = logging.getLogger(__name__)
import torch
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(self, 
                #  network_path,      # from which we load the actor
                 actor,
                 critic,
                 horizon_steps,
                 action_dim,
                 #  cond_steps,
                 #  obs_dim,
                 #  gamma,
                 device,
                 nstep=1,            # nstep replay buffer
                 ratio_clip_range=0.2,     # clip importance sampling ratio
                 value_clip_range=None,
                 update_epochs=10,   # in each call to self.update, the model learn the same batch (the whole buffer) in this manner times. 
                 value_coef=0.5,     # value function losses' coefficient
                 entropy_coef=0.00,  # entropy regularization loss's coefficient 0.01
                 use_bc_loss_coeff=0.0,
                 mini_batch_size=512,# minibatch size for each actual gradient descent
                 ):
        super().__init__()
        self.device = device
        
        self.actor = actor
        self.critic = critic
        self.to(self.device)
        
        self.horizon_steps = horizon_steps 
        self.action_dim = action_dim
        

        self.n_step = nstep
        
        self.ratio_clip_range = ratio_clip_range                # this is the epislon used in clipped advantage objective.
        self.value_clip_range = value_clip_range                # when we use clipped value loss.
        self.action_deviate_clip_range = 3.0                    # control the deviation of sampled action from its mean, in multiples of std.
        self.target_kl = 1.0
        self.log_prob_min = -5.0
        self.log_prob_max = 2.0                                 # strictly speaking for gaussians this should be zero. 
        
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.bc_loss_coeff = use_bc_loss_coeff
        
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        self.shuffle_buffer= False
        self.n_critic_warmup_itr = 0
        self.max_grad_norm = None
        self.itr = 0                        # number of calls to self.update()
        self.break_optim=False              # flag to break optimization when approximate kl divergence between successive steps is too large.
        # self.ortho_init()                 # todo: orthogonal initialization
        # self.tau = tau                    # potential: poliak averaging (soft update)
        # self.gamma = gamma ** nstep     
        

    @torch.no_grad()
    def get_value(self, cond):
        if isinstance(cond, dict):
            cond = cond["state"]
        ret = self.critic(cond) # [batchsize, 1]
        ret = ret.view(ret.shape[0])     #[batchsize]
        return ret
    
    @torch.no_grad()
    def get_action_logprob(self, cond, deterministic=False):
        ''' 
        infer action by stochastic policy and compute its log probability, we use this to sample transition tuples and store to replay buffer.s
        return values: 
            action:        [num_venvs,horizon_steps,action_dim]
            log_prob_mean: [num_venvs,]
        '''
        # pass through a deterministic model
        action_mean, log_std = self.actor.forward(cond)
        
        action_mean = action_mean.view(-1, self.horizon_steps, self.action_dim).to(self.device)
        
        if deterministic:
            return action_mean, None
        
        std = log_std.view(-1, self.horizon_steps, self.action_dim).exp().to(self.device)
        
        dist = Normal(action_mean, std)
        
        # sample action from noisy distribution
        action = dist.sample()
        
        # clip action within few standard deviations from the mean. prevent extreme values
        action.clamp_(
            dist.loc - self.action_deviate_clip_range * dist.scale,
            dist.loc + self.action_deviate_clip_range * dist.scale,
        )
        log_prob = dist.log_prob(action)
        log_prob_mean = log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)
        # (num_venvs, x) we treat the rest of the dimension as independent, conditioned on the state. 
        # So the log likelihood should add up together. 
        
        
        return action, log_prob_mean
        
        
    def evaluate_actions(self, state, action):
        '''
        This function is called when we calculate the loss of PPO. we allow gradient flow. 
        return the log probability and entropy estimate of \pi(a|s)
        inputs: 
            state:  [bs, cond_horizon, obs_dim]
            action: [bs, act_horizon, act_dim]
        outputs:
            log_prob_mean:[bs,]
            entropy:    [bs,]
        '''
        action_mean, log_std = self.actor.forward({"state":state})
        action_mean = action_mean.view(-1, self.horizon_steps, self.action_dim).to(self.device)
        std = log_std.view(-1, self.horizon_steps, self.action_dim).exp().to(self.device)
        
        dist = Normal(action_mean, std)
        
        # device closed-form for log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        log_prob_mean = log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)
        entropy_mean = entropy.reshape(entropy.shape[0], -1).sum(dim=-1)
        
        return log_prob_mean, entropy_mean

    def get_policy_loss(self, 
            log_prob, 
            old_log_prob, 
            advantage
        ):
        """
        Return clipped surrogate loss given log_prob, old_log_prob, advantage
        log_prob:     Float[Tensor, "batch_size"]
        old_log_prob: Float[Tensor, "batch_size"]
        advantage: Float[Tensor, "batch_size"]
        
        return: -> Tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]
        """
        # normalize advantage within batch
        advantage = self.normalize_adv(advantage)
        
        # clip log probabilities to prevent excessive values
        log_prob, old_log_prob = self.clip_log_prob(log_prob, old_log_prob)
        
        # obtain importance sampling ratio
        log_ratio = log_prob-old_log_prob.detach()
        ratio=torch.exp(log_ratio)
        
        # check how many times ratio is clipped and whether approximated kl divergence in successive steps is too large.
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).nanmean()
            clipfrac = ((ratio - 1.0).abs() > self.ratio_clip_range).float().mean()
            
        # clipped surrogate loss
        clipped_ratio=torch.clamp(ratio,1.0-self.ratio_clip_range,1.0+self.ratio_clip_range)
        
        policy_loss= -torch.min(ratio*advantage.detach(), clipped_ratio*advantage.detach()).mean()
        
        return policy_loss, ratio.mean(), approx_kl, clipfrac
    
    def get_value_loss(self, 
            value, 
            old_value, 
            returns
        ) :
        """
        Return value loss given value, old_value, returns
        value: Float[Tensor, "batch_size"], 
        old_value: Float[Tensor, "batch_size"], 
        returns: Float[Tensor, "batch_size"]
        
        return value: -> Float[Tensor, ""]
        in the original implementation, the epsilon is chosen as the same value as the ratio clippling parameter, that is 0.2
        """
        if self.value_clip_range:
            clipped_value=torch.clamp(value, 
                                      min=old_value-self.value_clip_range,
                                      max=old_value+self.value_clip_range)
            value_loss = 0.5 *(torch.max((value-returns)**2, (clipped_value-returns)**2)).mean()
        else:
            value_loss = 0.5 * ((value-returns)**2).mean()
        return value_loss
    
    def get_entropy_loss(self, entropy):
        """
        Return entropy loss given entropy
        entropy: Float[Tensor, "batch_size"]
        return value:  -> Float[Tensor, ""]
        """
        entropy_loss= -torch.mean(entropy)

        return entropy_loss
    
    def normalize_adv(self, advantage):
        '''
        do a batch normalization for advantage to make it zero mean 1 std.
        '''
        return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    def clip_log_prob(self, log_prob, old_log_prob):
        '''
        prevent excessively large log probability values. 
        '''
        log_prob_clamped = log_prob.clamp(min=self.log_prob_min, max=self.log_prob_max) 
        old_log_prob_clamped = old_log_prob.clamp(min=self.log_prob_min, max=self.log_prob_max)
        return  log_prob_clamped, old_log_prob_clamped

    
    def update_step(self, batch):
        state, action, old_log_prob, old_value, advantage, returns = batch    # [bs, To, Do]   [bs, Ta, Da]  [bs]  [bs]  [bs] [bs]

        log_prob, entropy = self.evaluate_actions(state, action)              # [bs]  [bs]
        
        value = self.critic(state)

        # obtain policy gradient loss
        policy_loss, ratio_mean, approx_kl_mean, clip_frac= self.get_policy_loss(log_prob, old_log_prob, advantage)

        # obtain value function loss
        value_loss = self.get_value_loss(value, old_value, returns)

        # obtain entropy loss
        entropy_loss = self.get_entropy_loss(entropy)
        
        # total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # optimize total loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        loss.backward()
        
        self.critic_optimizer.step()
        if self.itr >= self.n_critic_warmup_itr:
            self.actor_optimizer.step()
            if self.max_grad_norm: 
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        
        # Stop gradient update if KL difference reaches target
        if self.target_kl is not None and approx_kl_mean > self.target_kl:
            self.break_optim = True
        
        return { 'loss' : loss.item(),
                 'actor_loss': (policy_loss + self.entropy_coef * entropy_loss).item(),
                 'critic_loss': (self.value_coef * value_loss).item(),
                 'policy_loss': policy_loss.item(),
                 'value_loss': value_loss.item(),
                 'entropy_loss': entropy_loss.item(),
                 'ratio': ratio_mean.item(),
                 'approx_kl': approx_kl_mean.item(),
                 'clip_frac': clip_frac.item()}

    def update(self, buffer: PPOReplayBuffer):
        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        actor_losses = [] 
        critic_losses = []
        ratios = []
        approx_kls = []
        clip_fracs=[]

        buffer_size = buffer.size * buffer.num_envs  
        indices = np.arange(buffer_size)
        
        states, actions, old_log_probs, old_values, advantages, returns = buffer.make_dataset()
        '''
        states:  [capacity x num_envs, obs_horizon, obs_dim]
        actions: [capacity x num_envs, act_horizon, act_dim]
        old_log_probs: [capacity x num_envs]
        old_values: [capacity x num_envs]
        advantages: [capacity x num_envs]
        returns: [capacity x num_envs]
        '''
        self.break_optim = False
        for update_epoch in range(self.update_epochs):
            if self.break_optim == True:
                    break
            # random shuffle dataset
            if self.shuffle_buffer:
                np.random.shuffle(indices)
            # do gradient descent on minibatches
            for start in range(0, buffer_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                minibatch_idx = indices[start:end]
                
                batch = (
                    states[minibatch_idx],
                    actions[minibatch_idx],
                    old_log_probs[minibatch_idx],
                    old_values[minibatch_idx],
                    advantages[minibatch_idx],
                    returns[minibatch_idx]
                )
                
                ret_dict = self.update_step(batch)
                
                if self.break_optim == True:
                    break
                # log losses of final epoch per update
                if update_epoch == self.update_epochs - 1:
                    losses.append(ret_dict['loss'])
                    actor_losses.append(ret_dict['actor_loss'])
                    critic_losses.append(ret_dict['critic_loss'])
                    policy_losses.append(ret_dict['policy_loss'])
                    value_losses.append(ret_dict['value_loss'])
                    entropy_losses.append(ret_dict['entropy_loss'])
                    ratios.append(ret_dict['ratio'])
                    approx_kls.append(ret_dict['approx_kl'])
                    clip_fracs.append(ret_dict['clip_frac'])

        # take average over `minibatch` samples.
        loss    = np.mean(losses)
        actor_loss = np.mean(actor_losses)
        critic_loss = np.mean(critic_losses)
        pg_loss = np.mean(policy_losses) 
        v_loss  = np.mean(value_losses)
        entropy_loss = np.mean(entropy_losses)
        ratio = np.mean(ratios)
        approx_kl = np.mean(approx_kls)
        clip_frac = np.mean(clip_fracs)
        
        if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        
        self.itr +=1
        return loss, pg_loss, v_loss, entropy_loss, approx_kl, clip_frac, ratio, actor_loss, critic_loss

    def train(self):
        # turn on all the droupout/batchnorm/layernorm
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        self.actor.train()
        self.critic.train()
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)