"""
DPPO fine-tuning.
run this line to finetune hopper-v2: 
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_reflow_mlp device=cuda:7 wandb=null
"""
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from util.scheduler import CosineAnnealingWarmupRestarts
from agent.finetune.train_ppo_agent import TrainPPOAgent
from flow.ft_ppo.ppoflow import PPOFlow
from agent.finetune.buffer import PPOFlowBuffer
# define buffer on cpu or cuda. Currently GPU version is not offering significant acceleration...communication could be a bottleneck. It just increases GPU volatile utilization from 7% to 13%

class TrainPPOFlowAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.ft_denoising_steps = self.model.ft_denoising_steps
        self.normalize_entropy_logprob_dim = True   # normalize entropy and logprobability over horizon steps and action dimension. so that we don't need to adjust entropy coeff when env scales up. 
        self.normalize_time_horizon = True          # normalize time horizon when calculating the logprob of the markov chain of a single simulated action. 
        self.lr_schedule = cfg.train.lr_schedule
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr
        self.critic_lr = cfg.train.critic_lr
        
        self.mode: PPOFlow
        
        self.buffer = PPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps= self.ft_denoising_steps, 
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        )
    def adjust_finetune_schedule(self):
        pass

    @torch.no_grad()
    def get_samples_logprobs(self, cond:dict, ret_device='cpu', save_chains=True, normalize_time_horizon=False, normalize_dimension=False):
        # returns: action_samples are still numpy because mujoco engine receives np.
        if save_chains:
            action_samples, chains_venv, logprob_venv  = self.model.get_actions(cond, eval_mode=self.eval_mode, save_chains=save_chains, normalize_time_horizon=normalize_time_horizon, normalize_dimension=normalize_dimension)        # n_envs , horizon_steps , act_dim
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv, logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
        else:
            action_samples, logprob_venv  = self.model.get_actions(cond, eval_mode=self.eval_mode, save_chains=save_chains, normalize_time_horizon=normalize_time_horizon, normalize_dimension=normalize_dimension)
            return action_samples.cpu().numpy(), logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
        
    
    def get_value(self, cond:dict, device='cpu'):
        # cond contains a floating-point torch.tensor on self.device
        if device == 'cpu':
            value_venv = self.model.critic.forward(cond).cpu().numpy().flatten()
        else:
            value_venv = self.model.critic.forward(cond).squeeze().float().to(self.device)
        return value_venv
    
    # overload
    def update_lr(self):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # adapt learning rate according to kl divergence on each minibatch.
            return
        else: # use predefined lr scheduler. 
            super().update_lr()
    
    def update_lr_adaptive_kl(self, approx_kl):
        min_lr = 1e-20
        max_lr = 1e-2
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_lr, self.critic_lr * 1.5)
            tune = 'increases'
        for actor_param_group, critic_param_group in zip(self.actor_optimizer.param_groups, self.critic_optimizer.param_groups):
            actor_param_group["lr"] = self.actor_lr
            critic_param_group["lr"] = self.critic_lr
        log.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
        
    def agent_update(self, verbose=True):
        obs, chains, returns, oldvalues, advantages, oldlogprobs = self.buffer.make_dataset()
        
        # Explained variation of future rewards using value function
        explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        # Update policy and critic
        clipfracs_list = []
        self.total_steps = self.n_steps * self.n_envs
        
        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    returns[inds_b], 
                    oldvalues[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b] 
                )
                
                # minibatch gradient descent
                self.model: PPOFlow
                pg_loss, entropy_loss, v_loss, bc_loss, \
                clipfrac, approx_kl, ratio= self.model.loss(*minibatch, use_bc_loss=self.use_bc_loss, normalize_time_horizon=self.normalize_time_horizon, normalize_dimension=self.normalize_entropy_logprob_dim)
                
                if verbose:
                    log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={approx_kl:.2e}")
                
                if self.target_kl and self.lr_schedule == 'adaptive_kl':
                    self.update_lr_adaptive_kl(approx_kl)
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
                
                clipfracs_list += [clipfrac]
                
                # update policy and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                # debug the losses
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                actor_old_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_old.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                log.info(f"before clipping: actor_norm={actor_norm:1.2f}, critic_norm={critic_norm:1.2f}, actor_old_norm={actor_old_norm:1.2f}")
                
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                if self.lr_schedule=='fixed' and self.target_kl and approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
            if self.lr_schedule=='fixed' and kl_change_too_much:
                break
            
        clip_fracs=np.mean(clipfracs_list)
        self.train_ret_dict = {
                    "loss": loss,
                    "pg_loss": pg_loss,
                    "v_loss": v_loss,
                    "entropy_loss": entropy_loss,
                    "bc_loss": bc_loss,
                    "approx_kl": approx_kl,
                    "ratio": ratio,
                    "clipfracs": clip_fracs,
                    "explained_var": explained_var,
                }
    
    def run(self):
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env() # for gpu version, add device=self.device
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    value_venv = self.get_value(cond=cond) # for gpu version add , device=self.device
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(cond=cond, normalize_time_horizon=self.normalize_time_horizon, normalize_dimension=self.normalize_entropy_logprob_dim) # for gpu version, add , device=self.device
                
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic) # for gpu version, add device=self.device
                self.agent_update()
            
            self.plot_state_trajecories() #(only in D3IL)

            self.update_lr()
            
            # update finetune scheduler of ReFlow Policy
            self.adjust_finetune_schedule()
            self.save_model()
            self.log()                                          # diffusion_min_sampling_std
            
            self.itr += 1