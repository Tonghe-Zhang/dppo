"""
PPO training for Gaussian/GMM policy.
torch version. 
"""
import os
import pickle
import numpy as np
import torch
import logging
import wandb
from util.reward_scaling_ts import RunningRewardScaler
log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent

class TrainPPOGaussianAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = cfg.device # ensure device is properly set
        self.current_best_reward = torch.tensor(-float('inf'), device=self.device) #Use torch.tensor for consistency
        self.is_best_so_far = False
        self.total_steps = self.n_steps * self.n_envs  # total number of actions
        self.running_reward_scaler = RunningRewardScaler(num_envs=cfg.env.n_envs, gamma=self.gamma, device=self.device)
        
    def prepare_video_path(self):
        # Prepare video paths for each env - only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        self.options_venv = [{'video_path': None} for _ in range(self.n_envs)] # Initialize with None video paths.
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
        self.obs_trajs = {"state": torch.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim), device=self.device).contiguous().pin_memory()}
        self.samples_trajs = torch.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim), device=self.device).contiguous().pin_memory()
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), device=self.device).contiguous().pin_memory()
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), device=self.device).contiguous().pin_memory()
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), device=self.device).contiguous().pin_memory()
        self.done_venv = torch.zeros((1, self.n_envs), device=self.device).contiguous().pin_memory()


    def buffer_add(self, step, state_venv, output_actions_venv, reward_venv, terminated_venv, done_venv):
        self.obs_trajs["state"][step] = torch.tensor(state_venv, device=self.device)
        self.samples_trajs[step] = torch.tensor(output_actions_venv, device=self.device)
        self.reward_trajs[step] = torch.tensor(reward_venv, device=self.device)
        self.terminated_trajs[step] = torch.tensor(terminated_venv, device=self.device)
        self.firsts_trajs[step + 1] = torch.tensor(done_venv, device=self.device)

    def reset_env(self):
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            self.prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            self.firsts_trajs[0] = 1
        else:
            # if done at the end of last iteration, the envs are just reset
            self.firsts_trajs[0] = self.done_venv


    def update_full_obs(self, info_venv):
        if self.save_full_observations:
            self.obs_full_trajs = torch.empty((0, self.n_envs, self.obs_dim), device=self.device)
            self.obs_full_trajs = torch.cat((self.obs_full_trajs, torch.tensor(info_venv, device=self.device).view(1, self.n_envs, self.obs_dim)), dim=0)


    def get_values(self):
        self.value_trajs = torch.empty((self.n_steps, self.n_envs), device=self.device)
        for t, obs_venv in enumerate(self.obs_trajs["state"]):  # obs_venv: [self.n_envs, self.n_cond_step, self.obs_dim]
            self.value_trajs[t] = self.model.critic(obs_venv).flatten()

    def get_logprobs(self):
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        samples_t = self.samples_trajs
        obs_t = self.obs_trajs["state"]
        for t, (obs_venv, actions_venv) in enumerate(zip(obs_t, samples_t)):
            self.logprobs_trajs[t] = self.model.get_logprobs({"state": obs_venv}, actions_venv)[0].flatten()

    def update_adv_returns(self, obs_venv):
        # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
        obs_venv_ts = {"state": torch.tensor(obs_venv["state"]).float().to(self.device)}
        self.advantages_trajs = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        lastgaelam = torch.zeros(self.n_envs, device=self.device) # Initialize as tensor
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = self.model.critic(obs_venv_ts["state"]).view(-1)
            else:
                nextvalues = self.value_trajs[t + 1]

            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * (1.0 - self.terminated_trajs[t])
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * (1.0 - self.terminated_trajs[t]) * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    def make_dataset(self):
        obs = self.obs_trajs["state"].flatten(0, 1)
        samples = self.samples_trajs.flatten(0, 1)
        returns = self.returns_trajs.flatten(0, 1)
        values = self.value_trajs.flatten(0, 1)
        advantages = self.advantages_trajs.flatten(0, 1)
        logprobs = self.logprobs_trajs.flatten(0, 1)
        return obs, samples, returns, values, advantages, logprobs


    def get_explained_var(self, values, returns):
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        return explained_var

    def buffer_update(self, obs_venv):
        with torch.no_grad():
            self.get_values()
            self.get_logprobs()
            self.normalize_reward()
            self.update_adv_returns(obs_venv)

    def update_agent(self):
        obs, samples, returns, values, advantages, logprobs = self.make_dataset()
        explained_var = self.get_explained_var(values, returns)
        clipfracs = []
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
                    logprobs[minibatch_idx],
                )
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.model.loss(*batch, use_bc_loss=self.use_bc_loss)
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_loss_coeff
                clipfracs.append(clipfrac)
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_change_too_much = True
                    break
            if kl_change_too_much:
                break
        self.train_ret_tuple = loss, pg_loss, v_loss, entropy_loss, std, approx_kl, ratio, clipfracs, explained_var

    def save_full_observation(self, info_venv):
        if self.save_full_observations:
            obs_full_venv = torch.stack([torch.tensor(info["full_obs"]["state"], device=self.device) for info in info_venv])
            self.obs_full_trajs = torch.cat((self.obs_full_trajs, obs_full_venv.transpose(1, 0, 2)), dim=0)


    def run(self):
        self.prepare_run()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_buffer()
            self.reset_env()
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {"state": torch.tensor(self.prev_obs_venv["state"]).float().to(self.device)}
                    samples = self.model.forward(cond=cond, deterministic=self.eval_mode)
                    output_venv = samples.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                self.update_full_obs(info_venv)
                self.buffer_add(step, self.prev_obs_venv["state"], output_venv, reward_venv, terminated_venv, done_venv)
                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0
            self.summarize_episode_reward()
            if not self.eval_mode:
                self.buffer_update(obs_venv)
                self.update_agent()
            self.plot_state_trajecories()
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.save_model()
            self.log()
            self.itr += 1

    def normalize_reward(self):
        if self.reward_scale_running:
            self.reward_trajs = self.running_reward_scaler(self.reward_trajs.T, self.firsts_trajs[:-1].T).T

    def summarize_episode_reward(self):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = torch.nonzero(self.firsts_trajs[:, env_ind] == 1).squeeze(1) #More efficient way to find indices
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i+1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end -1))

        if episodes_start_end:
            reward_trajs_split = [
                self.reward_trajs[start:end+1, env_ind] for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            episode_reward = torch.stack([torch.sum(reward_traj) for reward_traj in reward_trajs_split])
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = torch.stack([torch.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split])
            self.avg_episode_reward = torch.mean(episode_reward.float())
            self.avg_best_reward = torch.mean(episode_best_reward.float())
            self.success_rate = torch.mean((episode_best_reward >= self.best_reward_threshold_for_success).float())
        else:
            self.num_episode_finished = 0
            self.avg_episode_reward = torch.tensor(0.0, device=self.device)
            self.avg_best_reward = torch.tensor(0.0, device=self.device)
            self.success_rate = torch.tensor(0.0, device=self.device)
            log.info("[WARNING] No episode completed within the iteration!")

    def save_model(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved best model (reward: {self.current_best_reward:.3f}) to {save_path}\n ")
            self.is_best_so_far = False

    def plot_state_trajecories(self):
        if self.traj_plotter and self.itr % self.render_freq == 0 and self.n_render > 0:
            self.traj_plotter(
                obs_full_trajs=self.obs_full_trajs.cpu().numpy(), #Convert to numpy for plotting
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=self.itr,
            )
    
    def log(self):
        self.run_results.append({"itr": self.itr, "step": self.cnt_train_step})
        if self.save_trajs:
            self.run_results[-1]["obs_full_trajs"] = self.obs_full_trajs.cpu().numpy()
            self.run_results[-1]["obs_trajs"] = self.obs_trajs
            self.run_results[-1]["action_trajs"] = self.samples_trajs.cpu().numpy()
            self.run_results[-1]["reward_trajs"] = self.reward_trajs.cpu().numpy()
        
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                log.info(
                    f"Eval at itr {self.itr}: success rate {self.success_rate.item():.3f} | avg episode reward {self.avg_episode_reward.item():.3f} | avg best reward {self.avg_best_reward.item():.3f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "success rate - eval": self.success_rate.item(),
                            "avg episode reward - eval": self.avg_episode_reward.item(),
                            "avg best reward - eval": self.avg_best_reward.item(),
                            "num episode - eval": self.num_episode_finished,
                        },
                        step=self.itr,
                        commit=False,
                    )
                self.run_results[-1]["eval_success_rate"] = self.success_rate.item()
                self.run_results[-1]["eval_episode_reward"] = self.avg_episode_reward.item()
                self.run_results[-1]["eval_best_reward"] = self.avg_best_reward.item()
                if self.current_best_reward < self.avg_episode_reward:
                    self.current_best_reward = self.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward.item():.3f}")
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