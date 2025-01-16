
import numpy as np
import torch
import logging
log = logging.getLogger(__name__)
from util.reward_scaling import RunningRewardScaler

class PPOReplayBuffer:
    def __init__(self,
                 n_steps,
                 n_envs, 
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device):
        
        self.n_steps = n_steps
        self.n_envs  = n_envs
        
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.action_dim = action_dim
        
        self.n_cond_step = n_cond_step 
        self.obs_dim  =obs_dim
         
        self.furniture_sparse_reward =furniture_sparse_reward  
        self.best_reward_threshold_for_success=best_reward_threshold_for_success
        self.save_full_observation= save_full_observation
        self.reward_scale_running =reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(num_envs=n_envs, gamma=gamma)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_const = reward_scale_const
        self.device = device

    def add(self, step, state_venv, output_actions_venv, reward_venv, terminated_venv, done_venv):
        self.obs_trajs["state"][step] = state_venv
        self.samples_trajs[step] = output_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
        
    
    def reset(self):
        self.obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
        self.samples_trajs = np.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

        
        
    def update(self,obs_venv, critic:torch.nn.Module, get_log_probs):
        with torch.no_grad():
            self.compute_values(critic)
            
            self.compute_logprobs(get_log_probs)

            # normalize reward with running variance
            self.normalize_reward()

            self.update_adv_returns(obs_venv, critic)
    
    def compute_values(self, critic:torch.nn.Module):
        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        for t, obs_venv in enumerate(self.obs_trajs["state"]):             # obs_venv: [self.n_envs, self.n_cond_step, self.obs_dim]
            obs_venv=torch.tensor(obs_venv).float().to(self.device)
            self.value_trajs[t] = critic.forward(obs_venv).cpu().numpy().flatten()
    
    
    def compute_logprobs(self, get_log_probs): 
        '''
        get_log_probs: 
            cond_venv, actions_venv  --> numpy.ndarray
            where
                cond_venv: dict
                actions_venv: numpy.ndarray
        '''
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs))
        samples_t = torch.from_numpy(self.samples_trajs).float().to(self.device)
        obs_t  = torch.from_numpy(self.obs_trajs["state"]).float().to(self.device)
        
        for t, (obs_venv, actions_venv) in enumerate(zip(obs_t, samples_t)):
            cond_venv = {"state": obs_venv}
            self.logprobs_trajs[t]  = get_log_probs(cond_venv, actions_venv)
    
    
    def update_full_obs(self):
        if self.save_full_observation:
            self.obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            self.obs_full_trajs = np.vstack((self.obs_full_trajs, self.prev_obs_venv["state"][:, -1][None]))
    
    def save_full_obs(self, info_venv):
        if self.save_full_observation:
            obs_full_venv = np.array([info["full_obs"]["state"] for info in info_venv])  # n_envs x act_steps x obs_dim
            self.obs_full_trajs = np.vstack((self.obs_full_trajs, obs_full_venv.transpose(1, 0, 2)))
    
    def normalize_reward(self):
        '''
        normalize self.reward_trajs
        '''
        if self.reward_scale_running:
            reward_trajs_transpose = self.running_reward_scaler(
                reward=self.reward_trajs.T, first=self.firsts_trajs[:-1].T
            )
            self.reward_trajs = reward_trajs_transpose.T
    
    def update_adv_returns(self, obs_venv, critic:torch.nn.Module): 
        # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
        obs_venv_ts = {
            "state": torch.from_numpy(obs_venv["state"])
            .float()
            .to(self.device)
        }
        
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))
        
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                nextvalues = critic.forward(obs_venv_ts).reshape(1, -1).cpu().numpy()
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
    
    def make_dataset(self): 
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1)
        samples= torch.tensor(self.samples_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)
        return (obs, samples, returns, values, advantages, logprobs)
    
    
    def get_explained_var(self, values, returns):
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)
        return explained_var
    
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