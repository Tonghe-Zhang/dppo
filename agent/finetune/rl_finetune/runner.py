"""
Base class for simulation in rl environment
"""
import os
import numpy as np
import torch
import logging
import random
from tqdm import tqdm as tqdm
from util.timer import Timer
log = logging.getLogger(__name__)
from env.gym_utils import make_async
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate 
from copy import deepcopy
import cv2
import wandb 
from agent.finetune.rl_finetune.algorithm.td3_reflow import TD3ReFlow
from agent.finetune.rl_finetune.algorithm.helpers import current_time
from .buffer import get_buffer


class RLrunner:        
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.record_hyperparam()
        
        self.set_seed()
        
        self.make_venv()
        
        self.create_agent_buffer()
        
        ############ could be overload #############
        # self.denoising_step = 
        self.denoising_step_trained = self.cfg.denoising_steps
        ############################################
        
    
    def run(self):
        
        loss_log_dict=self.prepare_run_log()
        
        self.itr = 0
        self.total_steps = 0
        self.best_reward = np.float64('-inf')
        
        pbar = tqdm(total=self.n_train_itr, desc="Training Progress")
        while self.itr < self.n_train_itr:
            self.agent.train()
            # TODO: render video every some iterations. 
            
            # rollout buffer
            firsts_trajs = np.zeros((self.n_rollout_steps + 1, self.n_envs))
            firsts_trajs[0] = 1
            reward_trajs = np.zeros((self.n_rollout_steps, self.n_envs))
            
            # Single rollout
            prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
            for step in range(1, self.n_rollout_steps+1):
                self.total_steps+=1
                
                # get observation
                cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}   # might contain visual informaiton, neglected for now.

                # play policy
                self.agent: TD3ReFlow
                action_venv = self.agent.act(cond, self.denoising_step)
                
                # environment transition
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                # append buffers
                self.buffer.add((cond["state"], action_venv, reward_venv, obs_venv, done_venv))     # need to extend to allow vectorized envs. 
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv
                
                # next step
                prev_obs_venv = obs_venv

                # update model with buffer.
                if self.total_steps > self.batch_size + self.nsteps:
                    batch = self.buffer.sample(self.cfg.batch_size)
                    
                    ret_dict = self.agent.update(batch)   # critic_loss, critic_loss2, td_loss, actor_loss
                    
                    for key in loss_log_dict.keys():
                        loss_log_dict[key][-1].append(ret_dict[key])
            
            # log training rollout
            train_log = self.episode_stats(firsts_trajs=firsts_trajs, reward_trajs=reward_trajs, source='train')
            if self.use_wandb: wandb.log(train_log, step=self.itr, commit=True)
            
            # save model
            if (not self.itr % self.save_interval ) or self.itr == self.n_train_itr:
                self.save()
            
            # TODO: update and save ema model. 
            
            # evaluate and save best model
            if not self.itr % self.eval_interval:
                eval_log=self.evaluate()
                self.save_best(eval_log, self.itr)
                if self.use_wandb: wandb.log(eval_log, step=self.itr, commit=True)
                
            # log loss information
            if not self.itr %  self.log_interval:
                if self.use_wandb: wandb.log(loss_log_dict, step=self.itr, commit=True) 

            self.itr += 1
            pbar.update(1)
        pbar.close()

    
    def create_agent_buffer(self):

        self.agent = instantiate(self.cfg.agent)
        log.info(f"Successfully created self.agent {self.agent}")   
        
        self.buffer = get_buffer(self.cfg.buffer, 
                                 num_envs = self.n_envs,
                                 state_size=(self.cfg.cond_steps, self.cfg.obs_dim), 
                                 action_size=(self.cfg.horizon_steps, self.cfg.action_dim), 
                                 device=self.device, seed=self.seed)
    
        log.info(f"Successfully created self.buffer {self.buffer}")   
        
    
    def prepare_run_log(self):
        # determine the root dir for this finetune run
        self.finetune_log_dir =os.path.join('log', 'finetune', self.benchmark, self.env_name, self.agent.__class__.__name__, current_time())
        os.makedirs(self.finetune_log_dir, exist_ok=True)
        
        # Dump the configuration as YAML file
        cfg_path = self.finetune_log_dir + "/cfg.yaml"
        with open(cfg_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        print(f"Configuration saved to {cfg_path}") 
        
        # determine checkpoint dir
        self.checkpoint_dir = os.path.join(self.finetune_log_dir, "ckpts")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # in case we also output videos
        self.render_dir = os.path.join(self.finetune_log_dir, "render")
        os.makedirs(self.render_dir, exist_ok=True)
        
        # the dictionary to record losses of each update step
        loss_keys = ['critic_loss', 'critic_loss_2', 'td_error', 'actor_loss']
        loss_log_dict = {key: [] for key in loss_keys}
        return loss_log_dict
    
    def record_hyperparam(self):
        # constants 
        self.device = self.cfg.device
        self.batch_size = self.cfg.batch_size
        self.n_envs = self.cfg.env.n_envs
        self.n_cond_step = self.cfg.cond_steps
        self.obs_dim = self.cfg.obs_dim
        self.action_dim = self.cfg.action_dim
        self.act_steps = self.cfg.act_steps
        self.horizon_steps = self.cfg.horizon_steps
        self.max_episode_steps = self.cfg.env.max_episode_steps
        self.reset_at_iteration = self.cfg.env.get("reset_at_iteration", True)
        self.denoising_step = self.cfg.denoising_steps
        ## training loop config.
        self.n_train_itr = self.cfg.train.n_train_itr
        self.n_rollout_steps = self.cfg.n_rollout_steps
        self.eval_interval = self.cfg.train.eval_interval
        self.save_interval = self.cfg.train.save_interval
        self.log_interval = self.cfg.train.log_interval
        # environment
        self.benchmark=self.cfg.benchmark
        self.env_name = self.cfg.env.name
        self.env_type = self.cfg.env.get("env_type", None)
        self.furniture_sparse_reward = (
            self.cfg.env.specific.get("sparse_reward", False)
            if "specific" in self.cfg.env
            else False
        )  # furniture specific, for best reward calculation
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if self.env_type == "furniture"
            else self.cfg.env.best_reward_threshold_for_success
        )
        log.info(f"Successfully recorded hyperparameters")
        
    def make_venv(self):
        # Make vectorized env
        self.venv = make_async(
            self.cfg.env.name,
            env_type=self.env_type,
            num_envs=self.cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=self.cfg.env.max_episode_steps,
            wrappers=self.cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=self.cfg.get("robomimic_env_cfg_path", None),
            shape_meta=self.cfg.get("shape_meta", None),
            use_image_obs=self.cfg.env.get("use_image_obs", False),
            render=self.cfg.env.get("render", False),
            render_offscreen=self.cfg.env.get("save_video", False),
            obs_dim=self.cfg.obs_dim,
            action_dim=self.cfg.action_dim,
            **self.cfg.env.specific if "specific" in self.cfg.env else {},
        )
        if not self.env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(self.cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        
        log.info(f"Successfully created self.venv={self.venv}")
        
        # self.eval_venv = deepcopy(self.venv) #if self.n_envs <= 1 else deepcopy(self.venv.envs[0]) # there could be a bug for their customized code. 
        
        
        self.eval_venv = make_async(
            self.cfg.env.name,
            env_type=self.env_type,
            num_envs=self.cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=self.cfg.env.max_episode_steps,
            wrappers=self.cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=self.cfg.get("robomimic_env_cfg_path", None),
            shape_meta=self.cfg.get("shape_meta", None),
            use_image_obs=self.cfg.env.get("use_image_obs", False),
            render=self.cfg.env.get("render", False),
            render_offscreen=self.cfg.env.get("save_video", False),
            obs_dim=self.cfg.obs_dim,
            action_dim=self.cfg.action_dim,
            **self.cfg.env.specific if "specific" in self.cfg.env else {},
        )
        log.info(f"Successfully created self.eval_venv={self.eval_venv}")
        
        
    def set_seed(self):
        self.seed = self.cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs

    def get_traj_length(self, episodes_start_end):
        """
        Calculates the average value of end - start for a list of tuples.
        
        Parameters:
        episodes_start_end (list of tuples): A list where each tuple is (env_ind, start, end).
        
        Returns:
        float: The average value of end - start. Returns 0 if the list is empty. It is the average length of episode without failing. 
        """
        total = 0
        count = len(episodes_start_end)
        
        for episode in episodes_start_end:
            _, start, end = episode  # Unpacking the tuple
            total += (end - start)
        
        traj_length = total / count if count > 0 else 0  # Avoid division by zero
        return traj_length


    def evaluate(self, VERBOSE_LOG=False):
        self.agent.eval()        
        
        firsts_trajs = np.zeros((self.n_rollout_steps + 1, self.n_envs))
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_rollout_steps, self.n_envs))
        
        # Collect a single trajectories for each env
        prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
        for step in tqdm(range(self.n_rollout_steps)) if VERBOSE_LOG else range(self.n_rollout_steps):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
            action_venv = self.agent.act(cond, self.denoising_step)

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.eval_venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        eval_log_dict=self.episode_stats(firsts_trajs=firsts_trajs, reward_trajs=reward_trajs, log_info=True)
        
        return eval_log_dict
    
    def episode_stats(self, firsts_trajs, reward_trajs, log_info=False, save_statistics=False, result_path=None, source='eval'):
        '''
        source: whether this episode is from training or evaluation. 
        '''
        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
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
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)
            
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            avg_traj_length=self.get_traj_length(episodes_start_end)
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_episode_reward_std=0.0
            avg_best_reward = 0
            avg_best_reward_std=0.0
            success_rate = 0
            avg_traj_length=self.get_traj_length(episodes_start_end)
            log.info("[WARNING] No episode completed within the iteration!")
        
        if log_info:
            log.info(
                f"avg_traj_length {avg_traj_length:4.2f} | num episode {num_episode_finished:4d}/{5} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.1f}±{avg_episode_reward_std:2.1f} | avg best reward {avg_best_reward:8.1f}±{avg_best_reward_std:2.1f} |"
            )
        if save_statistics:
            np.savez(
                result_path,
                num_episode=num_episode_finished,
                eval_success_rate=success_rate,
                eval_episode_reward=avg_episode_reward,
                eval_best_reward=avg_best_reward
            )
        
        log_dict={
            f"{source}_avg_traj_length": avg_traj_length,
            f"{source}_success_rate": success_rate,
            f"{source}_avg_episode_reward": avg_episode_reward,
            f"{source}_avg_episode_reward_std": avg_episode_reward_std,
            f"{source}_avg_best_reward": avg_best_reward,
            f"{source}_avg_best_reward_std": avg_best_reward_std,
        }
        return log_dict
    
    def save(self):
        save_path = os.path.join(self.checkpoint_dir, f"{self.epoch}.pt")
        print(f"Saving runner to {save_path}...")
        self.agent.save_model(save_path)
    
    def save_best(self, eval_log_dict, itr):
        if eval_log_dict["avg_episode_reward"] > self.best_reward:
            self.best_reward = eval_log_dict["avg_episode_reward"]
            print(f"[iter={itr}]. Saving the runner with highest avg_episode_reward: {self.best_reward:8.2f} in evaluation to {save_path}...")
            save_path = os.path.join(self.checkpoint_dir, f"best.pt")
            self.agent.save_model(save_path)
        
    
        
        
        
        
        
        