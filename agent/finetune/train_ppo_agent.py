"""
Parent PPO fine-tuning agent class.

"""

from typing import Optional
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)
from agent.finetune.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler


# appended by Tonghe
import os
import numpy as np
from util.timer import Timer
import os
import numpy as np
import pickle
import wandb
class TrainPPOAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Batch size for logprobs calculations after an iteration --- 
        # prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Warm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        # use cosine scheduler with linear warmup
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef = cfg.train.get("ent_coef", 0.01)

        # Value loss coefficient
        self.vf_coef = cfg.train.get("vf_coef", 0.5)

        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        
        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)
        
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(cfg.env.n_envs)

        # Use base policy
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_coeff: float = cfg.train.get("bc_loss_coeff", 0)
        
        
        
        # appended by Tonghe:
        self.current_best_reward = np.float32('-inf')
        self.is_best_so_far = False 
        self.total_steps = self.n_steps * self.n_envs # total number of actions
        self.buffer = None

    def reset_actor_optimizer(self):
        """Not used anywhere currently"""
        new_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=self.cfg.train.actor_lr,
            weight_decay=self.cfg.train.actor_weight_decay,
        )
        new_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.actor_optimizer = new_optimizer

        new_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.actor_lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        new_scheduler.load_state_dict(self.actor_lr_scheduler.state_dict())
        self.actor_lr_scheduler = new_scheduler
        log.info("Reset actor optimizer")
    
    
    
    # appended by Tonghe
    def prepare_video_path(self):
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        self.options_venv = [{} for _ in range(self.n_envs)]
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                self.options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )
    def set_model_mode(self):
        # Define train or eval - all envs restart
        self.eval_mode = self.itr % self.val_freq == 0 and not self.force_train
        self.model.eval() if self.eval_mode else self.model.train()
        self.last_itr_eval = self.eval_mode
    def prepare_run(self):
        # Start training loop
        self.timer = Timer()
        self.run_results = []
        self.cnt_train_step = 0
        self.last_itr_eval = False
        self.done_venv = np.zeros((1, self.n_envs))
    def reset_env(self):
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            self.prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            self.buffer.firsts_trajs[0] = 1
        else:
            # if done at the end of last iteration, the envs are just reset
            self.buffer.firsts_trajs[0] = self.done_venv
    def save_model(self):
        """
        overload. 
        saves model to disk; no ema recorded. 
        TODO: save ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        
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
    
    def plot_state_trajecories(self): 
        if not self.traj_plotter:
            return 
        if self.itr % self.render_freq == 0 and self.n_render > 0:
                self.traj_plotter(
                    obs_full_trajs=self.obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )
    
    
    def update_step(self, batch):
        raise NotImplementedError
    
    def agent_update(self):
        # put all the samples in n_steps x n_envs in a line
        obs, samples, returns, values, advantages, logprobs = self.buffer.make_dataset()
        
        # Explained variation of future rewards using value function
        explained_var = self.buffer.get_explained_var(values, returns)
        
        clipfracs = []
        
        # generate a random minibatch of data. 
        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            
            indices = torch.randperm(self.total_steps, device=self.device)
            
            for start in range(0, self.total_steps, self.batch_size):
                end = start + self.batch_size
                minibatch_idx = indices[start:end]

                batch = (
                    {"state": obs[minibatch_idx]},
                    samples[minibatch_idx],
                    returns[minibatch_idx],
                    values[minibatch_idx],
                    advantages[minibatch_idx],
                    logprobs[minibatch_idx]
                )

                # minibatch gradient descent
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.update_step(batch) 
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_loss_coeff
                
                clipfracs += [clipfrac]
                
                # update policy and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm : torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Stop gradient update if KL difference reaches target
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_change_too_much = True
                    break
            if kl_change_too_much:
                break
        
        self.train_ret_tuple = loss, pg_loss, v_loss, entropy_loss, std, approx_kl, ratio, clipfracs, explained_var
    
    
    def log(self):
        self.run_results.append(
                {
                    "itr": self.itr,
                    "step": self.cnt_train_step,
                }
            )
        if self.save_trajs:
            self.run_results[-1]["self.obs_full_trajs"] = self.obs_full_trajs
            self.run_results[-1]["self.obs_trajs"] = self.obs_trajs
            self.run_results[-1]["action_trajs"] = self.samples_trajs
            self.run_results[-1]["self.reward_trajs"] = self.reward_trajs
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                log.info(
                    f"Evaluation at self.itr={self.itr}: success rate {self.buffer.success_rate:8.3f} | avg episode reward {self.buffer.avg_episode_reward:8.3f} | avg best reward (per action) {self.buffer.avg_best_reward:8.3f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - eval": self.buffer.success_rate,
                            "avg episode reward - eval": self.buffer.avg_episode_reward,
                            "avg best reward - eval": self.buffer.avg_best_reward,
                            "num episode - eval": self.self.buffer.num_episode_finished,
                        },
                        step=self.itr,
                        commit=False,
                    )
                self.run_results[-1]["eval_success_rate"] = self.buffer.success_rate
                self.run_results[-1]["eval_episode_reward"] = self.buffer.avg_episode_reward
                self.run_results[-1]["eval_best_reward"] = self.buffer.avg_best_reward
                
                if self.current_best_reward < self.buffer.avg_episode_reward:
                    self.current_best_reward = self.buffer.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                loss, pg_loss, v_loss, entropy_loss, std, approx_kl, ratio, clipfracs, explained_var= self.train_ret_tuple
                log.info(
                    f"self.itr={self.itr}: total steps {self.cnt_train_step/1e6:4.3f} M | self.buffer.avg_episode_reward={self.buffer.avg_episode_reward:8.3f} \n |loss {loss:8.3f} | pg loss {pg_loss:8.3f} | value loss {v_loss:8.3f} | ent {-entropy_loss:8.3f} |  t:{time:8.3f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "total env step": self.cnt_train_step,
                            "loss": loss,
                            "pg loss": pg_loss,
                            "value loss": v_loss,
                            "entropy loss": -entropy_loss,
                            "std": std,
                            "approx kl": approx_kl,
                            "ratio": ratio,
                            "clipfrac": np.mean(clipfracs),
                            "explained variance": explained_var,
                            "avg episode reward - train": self.buffer.avg_episode_reward,
                            "num episode - train": self.self.buffer.num_episode_finished,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.critic_optimizer.param_groups[0]["lr"]
                        },
                        step=self.itr,
                        commit=True,
                    )
                self.run_results[-1]["train_episode_reward"] = self.buffer.avg_episode_reward
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)
                
                