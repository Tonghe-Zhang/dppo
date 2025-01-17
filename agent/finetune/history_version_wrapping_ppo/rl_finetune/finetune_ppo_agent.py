"""
Parent PPO fine-tuning agent class.
"""
from typing import Optional
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts
import numpy as np
import pickle
import wandb
log = logging.getLogger(__name__)
from agent.finetune.history_version_wrapping_ppo.rl_finetune.finetune_agent import FinetuneAgent
import os
from tqdm import tqdm as tqdm
from model.rl.rl_buffers import PPOReplayBuffer
from model.rl.ppo import PPOAgent
''' 
action_venv, log_probs_venv = self.agent.get_action_logprob(cond=cond, deterministic=self.eval_mode)
value_venv = self.agent.get_value(state_venv) 
loss, pg_loss, v_loss, approx_kl, ratio = self.agent.update(self.buffer)
'''

class FinetunePPOAgent(FinetuneAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device

        # Batch size for logprobs calculations after an iteration --- prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # DPPO author: note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Warm up period for critic before actor updates
        self.agent.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr
        
        self.agent.max_grad_norm = cfg.train.get("max_grad_norm", None)
    
        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.get("target_kl", np.float32('inf'))

        ## overload....
        # Number of times the collected data is used in gradient update
        self.agent.update_epochs = cfg.train.update_epochs

        # Entropy loss coefficient
        self.agent.entropy_coef = cfg.train.entropy_coef

        # Value loss coefficient
        self.agent.value_coef= cfg.train.value_coef

        # each gradient step takes on minibatch of transitions
        self.agent.mini_batch_size = cfg.train.get("mini_batch_size",1024)
        
        # Use base policy
        self.agent.bc_loss_coeff  = cfg.train.bc_loss_coef
        self.use_bc_loss = False if self.agent.bc_loss_coeff == 0.0 else True
        
        # importance sampling ratio clipping range and action sampling clipping range
        self.agent.ratio_clip_range = cfg.train.clip_ploss_coef
        self.agent.action_deviate_clip_range = cfg.train.randn_clip_value
        self.agent.target_kl = cfg.train.target_kl
        
        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        
        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1.0)

        # Warm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer overload
        self.agent.actor_optimizer = torch.optim.AdamW(
            self.agent.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        # use cosine scheduler with linear warmup
        self.agent.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.agent.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.agent.critic_optimizer = torch.optim.AdamW(
            self.agent.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.agent.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.agent.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        
        log.info(f"Finished initializing {self.__class__.__name__}. \n \
            self.reward_scale_running={self.reward_scale_running} \n  \
            self.agent.update_epochs={self.agent.update_epochs} \n \
            self.agent.mini_batch_size={self.agent.mini_batch_size} \n \
            self.agent.entropy_coef={self.agent.entropy_coef} \n \
            self.agent.value_coef = {self.agent.value_coef} \n \
            self.agent.bc_loss_coeff={self.agent.bc_loss_coeff} \n")

        self.train_ret_tuple = None
    def create_buffer(self, buffer_type='PPO_state'):
        # state-only buffer
        if buffer_type == 'PPO_state':
            self.buffer = PPOReplayBuffer(capacity=self.n_steps, 
                                        num_envs=self.n_envs, 
                                        state_size =self.obs_dim,
                                        state_horizon= self.n_cond_step, 
                                        action_size=self.act_dim,
                                        action_horizon= self.horizon_steps, 
                                        device = self.device, 
                                        seed = self.seed,
                                        gamma = self.gamma,
                                        gae_lambda=self.gae_lambda,
                                        reward_scale_running=self.reward_scale_running)
        else:
            raise ValueError(f"Unsupported buffer type {buffer_type}.")
    
    def log_and_save(self):  
        self.run_results.append(
                {
                    "itr": self.itr,
                    "step": self.cnt_train_step,
                }
            )
        
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            
            # record evaluation results: self.episode_reward, self.num_episode_finished, self.avg_episode_reward, self.avg_best_reward, self.success_rate
            if self.eval_mode:
                log.info( 
                    f"\n eval: success rate {self.success_rate:8.4f} | avg episode reward {self.avg_episode_reward:8.4f} | avg best reward {self.avg_best_reward:8.4f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - eval": self.success_rate,
                            "avg episode reward - eval": self.avg_episode_reward,
                            "avg best reward - eval": self.avg_best_reward,
                            "num episode - eval":self.num_episode_finished,
                        },
                        step=self.itr,
                        commit=False,
                    )
                self.run_results[-1]["eval_success_rate"] = self.success_rate
                self.run_results[-1]["eval_episode_reward"] = self.avg_episode_reward
                self.run_results[-1]["eval_best_reward"] = self.avg_best_reward
            # record training losses
            else:
                loss, pg_loss, v_loss, entropy_loss, approx_kl, clip_fracs, ratio, actor_loss, critic_loss = self.train_ret_tuple #bc_loss,  clipfracs 
                log.info(
                    f"\n train itr={self.itr}: step {self.cnt_train_step/1e6:2.3f} M | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | entropy loss {entropy_loss:8.4f} | actor loss {actor_loss:8.4f} critic loss {critic_loss:8.4f} |reward {self.avg_episode_reward:8.4f} | time elapsed:{time:8.4f}"
                )#bc loss {bc_loss:8.4f} 
                if self.use_wandb:
                    wandb.log(
                        {
                            "total env step": self.cnt_train_step,
                            "loss": loss,
                            "actor loss": actor_loss, 
                            "critic loss": critic_loss,
                            "pg loss": pg_loss,
                            "value loss": v_loss,
                            # "bc loss": bc_loss,
                            "entropy_loss": entropy_loss,
                            "approx kl": approx_kl,
                            "clipfrac": clip_fracs,
                            "ratio": ratio,
                            # "clipfrac": np.mean(clipfracs),
                            "avg episode reward - train": self.avg_episode_reward,
                            "num episode - train": self.num_episode_finished,
                            "avg best reward - train": self.avg_best_reward,
                            "num episode - train":self.num_episode_finished,
                            "actor lr": self.agent.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.agent.critic_optimizer.param_groups[0]["lr"],
                        },
                        step=self.itr,
                        commit=True,
                    )
                self.run_results[-1]["train_episode_reward"] = self.avg_episode_reward
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)
    
    def save_model(self):
        """
        overload. 
        saves model to disk; no ema recorded. 
        TODO: save ema
        """
        data = {
            "itr": self.itr,
            "agent": self.agent.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer": self.agent.critic_optimizer.state_dict(),
        }
        
        # always save the last model
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        # log.info(f"\n Saved model at itr {self.itr} to {save_path}\n ")
        
        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model to {save_path}\n ")
        
        # save the best model evaluated so far 
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir,f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model to {save_path}\n ")
            
            self.is_best_so_far =False
    
    def move_to(self, transition):
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = transition 
        obs_venv = { 
                         "state": torch.tensor(obs_venv["state"], dtype=torch.float32).to(self.device)
                         }
        reward_venv = torch.tensor(reward_venv,dtype=torch.float32).to(self.device)
        terminated_venv = torch.tensor(terminated_venv,dtype=torch.float32).to(self.device)
        truncated_venv = torch.tensor(truncated_venv,dtype=torch.float32).to(self.device)
        
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv
        
    def run(self):
        self.prepare()
        
        self.create_buffer('PPO_state')
        
        obs_venv = None
        done_venv = None
        self.buffer.clear()
        
        for itr in tqdm(range(self.n_train_itr),desc="Training Progress"):
            
            self.prepare_video_paths()

            self.switch_model_mode()

            self.buffer.clear()
            
            # Handling manual reset when we start and end evaluation (or specified by config file). 
            # After that, a new state is generated and a new episode begins. 
            # In fact, after `done`, the environment automatically and natually resets, and in that case, we use the last observation to continue the episode. 
            
            if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
                prev_obs_venv = {"state": torch.tensor(prev_obs_venv["state"], dtype=torch.float32).to(self.device)}
                self.buffer.firsts_trajs[0] = 1
            else:
                self.buffer.firsts_trajs[0] = self.buffer.done
            
            # rollout `self.n_steps` steps for all the envs in parallel, filling the buffer. 
            for step in range(self.n_steps): #tqdm(range(self.n_steps), dynamic_ncols=True, smoothing=0.1, desc="Roll out a sample"):
                with torch.no_grad():
                    # get action and its log probability
                    action_venv, log_prob_venv= self.agent.get_action_logprob(
                        cond=prev_obs_venv,
                        deterministic=self.eval_mode, # sample actions deterministically during evaluation, but add some randomness during training
                    )                        
                
                # Apply multi-step action
                transition = self.venv.step(action_venv.cpu().numpy()) # to accomodate mujoco's simulation on cpu.
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.move_to(transition)  # move to gpu
                
                if not self.eval_mode:  # update buffer, do not record evaluation data
                    with torch.no_grad():
                        done_venv = terminated_venv.bool() | truncated_venv.bool()
                        value_venv = self.agent.get_value(prev_obs_venv)
                        self.buffer.add((prev_obs_venv["state"], action_venv, obs_venv["state"], reward_venv, done_venv, terminated_venv, value_venv, log_prob_venv))
                
                # proceed to next state
                prev_obs_venv = obs_venv
                
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0
                
            # calculate the success rates and average episode rewards for all the rollouts in this iteration
            self.summarize_iteration_reward(reward_trajs=self.buffer.reward, firsts_trajs=self.buffer.firsts_trajs)
            
            if not self.eval_mode:
                with torch.no_grad():
                    
                    # normalize rewards, compute adv and returns
                    if self.reward_scale_running: self.buffer.normalize_reward() 
                
                    self.buffer.compute_advantages_and_returns(self.agent)
                
                # optimize the networks
                self.agent: PPOAgent
                self.train_ret_tuple = self.agent.update(self.buffer)
            
            # Save the last, best, and intermediate models
            self.save_model()
            
            # Log loss and save metrics
            self.log_and_save()
            
            self.itr  += 1