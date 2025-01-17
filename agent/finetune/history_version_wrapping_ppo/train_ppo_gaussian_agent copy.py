"""
PPO training for Gaussian/GMM policy.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent


class TrainPPOGaussianAgent(TrainPPOAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.current_best_reward = np.float32('-inf')
        self.is_best_so_far = False 
    
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
    
    def reset_buffer(self):
        self.obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
        self.samples_trajs = np.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        
    def update_buffer(self, step, state_venv, output_actions_venv, reward_venv, terminated_venv, done_venv):
        self.obs_trajs["state"][step] = state_venv
        self.samples_trajs[step] = output_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
                
    def reset_env(self):
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            self.prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            self.firsts_trajs[0] = 1
        else:
            # if done at the end of last iteration, the envs are just reset
            self.firsts_trajs[0] = self.done_venv
    
    def update_full_obs(self):
        if self.save_full_observations:
            self.obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            self.obs_full_trajs = np.vstack((self.obs_full_trajs, self.prev_obs_venv["state"][:, -1][None]))
            
    
    def get_values(self, obs_ts): 
        # get values
        self.value_trajs = np.empty((0, self.n_envs))
        for obs in obs_ts:
            values = self.model.critic(obs).cpu().numpy().flatten()
            self.value_trajs = np.vstack(
                (self.value_trajs, values.reshape(-1, self.n_envs))
            )
    
    def get_logprobs(self,obs_ts): 
        # get log probs
        samples_t = einops.rearrange(
            torch.from_numpy(self.samples_trajs).float().to(self.device),
            "s e h d -> (s e) h d",
        )
        samples_ts = torch.split(samples_t, self.logprob_batch_size, dim=0)
        self.logprobs_trajs = np.empty((0))
        for obs_t, samples_t in zip(obs_ts, samples_ts):
            logprobs = (
                self.model.get_logprobs(obs_t, samples_t)[0].cpu().numpy()
            )
            self.logprobs_trajs = np.concatenate(
                (
                    self.logprobs_trajs,
                    logprobs.reshape(-1),
                )
            )
    
    def update_adv_returns(self, obs_venv): 
        # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
        obs_venv_ts = {
            "state": torch.from_numpy(obs_venv["state"])
            .float()
            .to(self.device)
        }
        
        self.advantages_trajs = np.zeros_like(self.reward_trajs)
        
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                nextvalues = self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]
            
            # delta = r + gamma*V(st+1) - V(st)
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * (1.0 - self.terminated_trajs[t])
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * (1.0 - self.terminated_trajs[t]) * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs
                    
    def run(self):
        
        self.prepare_run()
        
        self.done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            
            self.prepare_video_path()
            
            self.set_model_mode()
            
            self.reset_buffer()
            
            self.reset_env()
            
            self.update_full_obs()
            
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(self.prev_obs_venv["state"]).float().to(self.device)
                    }
                    samples = self.model.forward(
                        cond=cond,
                        deterministic=self.eval_mode,
                    )
                    output_venv = samples.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array([info["full_obs"]["state"] for info in info_venv])  # n_envs x act_steps x obs_dim
                    self.obs_full_trajs = np.vstack((self.obs_full_trajs, obs_full_venv.transpose(1, 0, 2)))
                
                self.update_buffer(step, self.prev_obs_venv["state"], output_venv, reward_venv,terminated_venv,done_venv)
                
                # update for next step
                self.prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.summarize_episode_reward()

            # Update models
            if not self.eval_mode:
                with torch.no_grad():
                    # split obesrvations
                    self.obs_trajs["state"] = torch.from_numpy(self.obs_trajs["state"]).float().to(self.device)

                    # split observations into batches to prevent out of memory
                    num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                    
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        self.obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    
                    self.get_values(obs_ts)
                    
                    self.get_logprobs(obs_ts)

                    # normalize reward with running variance
                    self.normalize_reward()

                    self.update_adv_returns(obs_venv)

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        self.obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                samples_k = einops.rearrange(
                    torch.tensor(self.samples_trajs, device=self.device).float(),
                    "s e h d -> (s e) h d",
                )
                returns_k = (
                    torch.tensor(self.returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(self.value_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(self.advantages_trajs, device=self.device).float().reshape(-1)
                )
                logprobs_k = torch.tensor(self.logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        obs_b = {"state": obs_k["state"][inds_b]}
                        samples_b = samples_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            std,
                        ) = self.model.loss(
                            obs_b,
                            samples_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]
                        
                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        
                        loss.backward()
                        
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                        self.critic_optimizer.step()

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break
                
                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)

                self.train_ret_tuple = loss, pg_loss, v_loss, entropy_loss, std, approx_kl, ratio, clipfracs, explained_var
            
            # Plot state trajectories (only in D3IL)
            self.plot_state_trajecories()

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
            # Save model
            self.save_model()
            
            # Log loss
            self.log()
            
            self.itr += 1

    def normalize_reward(self):
        '''
        normalize self.reward_trajs
        '''
        if self.reward_scale_running:
            reward_trajs_transpose = self.running_reward_scaler(
                reward=self.reward_trajs.T, first=self.firsts_trajs[:-1].T
            )
            self.reward_trajs = reward_trajs_transpose.T
    
    def summarize_episode_reward(self):
        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(self.firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            # print(f"episodes_start_end={len(episodes_start_end)}, {episodes_start_end[0]}")
            # print(f"self.reward_trajs={len(self.reward_trajs)}, {self.reward_trajs[0]}")
            reward_trajs_split = [
                self.reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if (
                self.furniture_sparse_reward
            ):  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
        else:
            episode_reward = np.array([])
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.avg_best_reward = 0
            self.success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")
       
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
                    f"eval: success rate {self.success_rate:8.3f} | avg episode reward {self.avg_episode_reward:8.3f} | avg best reward {self.avg_best_reward:8.3f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - eval": self.success_rate,
                            "avg episode reward - eval": self.avg_episode_reward,
                            "avg best reward - eval": self.avg_best_reward,
                            "num episode - eval": self.num_episode_finished,
                        },
                        step=self.itr,
                        commit=False,
                    )
                self.run_results[-1]["eval_success_rate"] = self.success_rate
                self.run_results[-1]["eval_episode_reward"] = self.avg_episode_reward
                self.run_results[-1]["eval_best_reward"] = self.avg_best_reward
                
                if self.current_best_reward < self.avg_episode_reward:
                    self.current_best_reward = self.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                loss, pg_loss, v_loss, entropy_loss, std, approx_kl, ratio, clipfracs, explained_var= self.train_ret_tuple
                log.info(
                    f"self.itr={self.itr}: total steps {self.cnt_train_step/1e6:4.3f} M | self.avg_episode_reward={self.avg_episode_reward:8.3f} \n |loss {loss:8.3f} | pg loss {pg_loss:8.3f} | value loss {v_loss:8.3f} | ent {-entropy_loss:8.3f} |  t:{time:8.3f}"
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
                            "avg episode reward - train": self.avg_episode_reward,
                            "num episode - train": self.num_episode_finished,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.critic_optimizer.param_groups[0]["lr"]
                        },
                        step=self.itr,
                        commit=True,
                    )
                self.run_results[-1]["train_episode_reward"] = self.avg_episode_reward
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)