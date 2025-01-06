import torch.nn as nn
import os
import torch
from copy import deepcopy
import numpy as np
from agent.finetune.rl_finetune.algorithm.helpers import get_schedule
from model.flow.reflow import ReFlow
class TD3ReFlow:
    def __init__(self, 
                 actor, 
                 critic, 
                 device, 
                 lr_actor, 
                 lr_critic, 
                 gamma,
                 tau, 
                 target_update_interval:int, 
                 policy_update_interval:int,
                 noise_clip,
                 policy_noise,
                 eps_schedule,
                 nstep):
        
        actor: ReFlow
        self.actor_net = actor.to(device)
        self.actor_target = deepcopy(self.actor_net).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)
        
        self.critic_net = critic.to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
        
        self.critic_net_2 = critic.to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)
        
        self.tau = tau   # for poliak averaging
        self.gamma = gamma ** nstep
        self.device = device
        self.target_update_interval = target_update_interval
        self.policy_update_interval = policy_update_interval #  for delayed policy update in td3
        
        self.exploration_scheduler = get_schedule(eps_schedule)
        
        # for insertion of noise into the deterministic policy network
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        
        self.train_step = 0
    
    @torch.no_grad()
    def act(self, cond:dict, num_denoising_steps:int, stochastic_policy=True):
        self.actor_net: ReFlow
        action_sample = self.actor_net.sample(cond=cond, 
                                       inference_steps=num_denoising_steps, 
                                       inference_batch_size=1,     # this is suspicious...
                                       record_intermediate=False)
        action= action_sample.trajectories[:, :self.actor_net.act_horizon]    #muulti-step action
        # add exploration noise
        if stochastic_policy:
            action += torch.randn_like(action) * self.exploration_scheduler(self.train_step)
            action = action.clamp_(self.actor_net.act_range[0], self.actor_net.act_range[1])
        return action.cpu().numpy()
    
    def update(self, batch, weights=None)->dict:
        '''
        update the environment frequently, while update the actor less frequently. 
        '''
        state, action, reward, next_state, done = batch
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights)

        log_dict = {'critic_loss': critic_loss, 
                    'critic_loss_2': critic_loss_2, 
                    'td_error': td_error}
        
        if not self.train_step % self.policy_update_interval:
            log_dict['actor_loss'] = self.update_actor(state)
            self.poliak(self.actor_target, self.actor_net)

        if not self.train_step % self.target_update_interval:
            self.poliak(self.critic_target_2, self.critic_net_2)
            self.poliak(self.critic_target, self.critic_net)

        self.train_step += 1
        return log_dict
    
    
    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q = self.critic_net(state, action)
        Q2 = self.critic_net_2(state, action)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action) * self.policy_noise
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp_(self.actor_net.action_space.low, self.actor_net.action_space.high)

            Q_target = self.critic_target(next_state, next_action)
            Q_target_2 = self.critic_target_2(next_state, next_action)
            Q_target = reward + (1 - done) * self.gamma * torch.min(Q_target, Q_target_2)

            td_error = torch.abs(Q - Q_target)
    
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1) # useful to clip grads
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net_2.parameters(), 1) # useful to clip grads
        self.critic_optimizer_2.step()
        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()

    def update_actor(self, state):
        pred_action = self.actor_net(state)
        actor_loss = -self.critic_net(state, pred_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()
    
    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q = self.critic_net(state, action)
        Q2 = self.critic_net_2(state, action)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action) * self.policy_noise
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp_(self.actor_net.action_space.low, self.actor_net.action_space.high)

            Q_target = self.critic_target(next_state, next_action)
            Q_target_2 = self.critic_target_2(next_state, next_action)
            Q_target = reward + (1 - done) * self.gamma * torch.min(Q_target, Q_target_2)

            td_error = torch.abs(Q - Q_target)
    
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1) # useful to clip grads
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net_2.parameters(), 1) # useful to clip grads
        self.critic_optimizer_2.step()
        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()    
    
    def poliak(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
            
    
    def save_model(self, save_path:str):
        data = {
            "epoch": self.epoch,
            "actor": self.actor_net.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic_net.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_2": self.critic_net_2.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "critic_optimizer_2": self.critic_optimizer_2.state_dict()
            # "ema": self.ema_model.state_dict(),
        }
        torch.save(data, save_path)
        print(f"model saved to {save_path}")
        
    def train(self):
        self.actor_net.train()
        self.critic_net.train()
        self.critic_net_2.train()
    def eval(self):
        self.actor_net.eval()
        self.critic_net.eval()
        self.critic_net_2.eval()
        # the last two lines are different from huazhe's implemetation