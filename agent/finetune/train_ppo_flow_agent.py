"""
DPPO fine-tuning.
run this line to finetune hopper-v2: 
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_reflow_mlp device=cuda:7 wandb=null
"""
import os
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
    def run(self):
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        if self.resume:
            self.resume_training()
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
            
    def adjust_finetune_schedule(self):
        pass
    
    def resume_training(self):
        log.info(f"Resuming training...")
        data = torch.load(self.resume_path, weights_only=True)
        self.itr = data["itr"]
        self.n_train_itr += self.itr
        self.cnt_train_step = self.n_envs * self.act_steps * self.itr if 'cnt_train_step' not in data.keys() else data["cnt_train_step"]
        
        if "model_full" in data.keys():
            self.model.load_state_dict(data["model_full"])
        elif "model" in data.keys():
            self.model.load_state_dict(data["model"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        log.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")
        log.info(f"Model loaded from path={self.resume_path}")
        
    # overload...
    def save_model(self):
        """
        overload. 
        saves model to disk; no ema recorded. 
        TODO: save ema
        """
        policy_state_dict = self.model.actor_ft.policy.state_dict()
        
        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "model": {'network.'+key :value for key, value in policy_state_dict.items()},
            "model_full": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        # when you load state_dict to resume training, use model_full. when you load state_dict for eval, load model.
        
        
        # always save the last model for resume of training. 
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        # log.info(f"\n Saved last model at itr {self.itr} to {save_path}\n ")
        
        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")
        
        # save the best model evaluated so far 
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir,f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far =False

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
    def update_lr(self, val_metric=None):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # adapt learning rate according to kl divergence on each minibatch.
            return
        else: # use predefined lr scheduler. 
            super().update_lr()
    
    def update_lr_adaptive_kl(self, approx_kl):
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        min_critic_lr = 1e-5
        max_critic_lr = 1e-3
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_critic_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_critic_lr, self.critic_lr * 1.5)
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
                clipfrac, approx_kl, ratio, \
                oldlogprob_min, oldlogprob_max, oldlogprob_std,\
                    newlogprob_min, newlogprob_max, newlogprob_std = self.model.loss(*minibatch, use_bc_loss=self.use_bc_loss, normalize_time_horizon=self.normalize_time_horizon, normalize_dimension=self.normalize_entropy_logprob_dim)
                
                if verbose:
                    log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={approx_kl:.2e}")
                
                if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)>1e-6:
                    raise ValueError(f"ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
                
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
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            }
    
    