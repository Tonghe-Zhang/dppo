"""
Parent fine-tuning agent class.
"""
import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
import random
from util.timer import Timer
log = logging.getLogger(__name__)
from env.gym_utils import make_async


class FinetuneAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.save_full_observations = cfg.env.get("save_full_observations", False)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )  # furniture specific, for best reward calculation

        # Build model and load checkpoint
        self.agent = hydra.utils.instantiate(cfg.agent)
        print(f"Successfully loaded agent {self.agent} \n actor={self.agent.actor}\n critic={self.agent.critic}")
        
        # Training params
        self.itr = 0
        self.n_train_itr = cfg.train.n_train_itr
        self.val_freq = cfg.train.val_freq
        self.force_train = cfg.train.get("force_train", False)
        self.n_steps = cfg.train.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs = cfg.train.get("save_trajs", False)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.render_freq = cfg.train.render.freq
        self.n_render = cfg.train.render.num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        self.traj_plotter = (
            hydra.utils.instantiate(cfg.train.plotter)
            if "plotter" in cfg.train
            else None
        )

        self.avg_episode_reward = 0.0 
        self.avg_episode_reward_best = np.float32('-inf')
        log.info(f"Finished initializing {self.__class__.__name__}")
        
    def run(self):
        raise ValueError("Not Implemented Error!")

    def save_model(self, optional_str:str):
        """
        saves model to disk; no ema recorded. 
        TODO: also save optimizer
        """
        data = {
            "itr": self.itr,
            "model": self.agent.state_dict(),
        }
        if optional_str:
            savepath = os.path.join(self.checkpoint_dir, f"{optional_str}.pt")
        else:
            savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"\n Saved model to {savepath}\n ")


    def load(self, itr, optional_str:str):
        """
        loads model from disk
        """
        if optional_str:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        else:
            loadpath = os.path.join(self.checkpoint_dir, f"{optional_str}.pt")
        
        data = torch.load(loadpath, weights_only=True)
        self.itr = data["itr"]
        self.agent.load_state_dict(data["agent"])
        # optional: also load the optimizers. could be overload to do that. 

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

    def prepare(self):
        
        self.timer = Timer()
        
        # logging information, will be saved. 
        self.run_results = []

        # if the last iteration is evaluation iter. This is turned true during evaluation, and fall back to false when training resumes. 
        self.last_itr_eval = False
        
        # total number of gradient descents (optimize the actor and critic)
        self.itr = 0
        
        # total number of inference steps for all the environments, equal to number of times we forward pass through the model (sample an action from observation)
        self.cnt_train_step = 0
        
        # whether the model in current iteration is the best evaluated so far. We will record best model's checkpoints. 
        self.is_best_so_far = False 
        
    
    def prepare_video_paths(self):
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        self.options_venv = [{} for _ in range(self.n_envs)]
        
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                self.options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )
    
    def switch_model_mode(self):
        # Define train or eval - all envs restart
        self.eval_mode = self.itr % self.val_freq == 0 and not self.force_train
        self.agent.eval() if self.eval_mode else self.agent.train()
        self.last_itr_eval = self.eval_mode    
    
    # def reset_venv(self, prev_obs_venv, verbose=False):
    #     # print(f"self.itr={self.itr}, self.reset_at_iteration={self.reset_at_iteration}, self.eval_mode={self.eval_mode}, self.last_itr_eval={self.last_itr_eval}")
    #     # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        
            # if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            #     initial_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            #     initial_obs_venv = {"state": torch.tensor(initial_obs_venv["state"], dtype=torch.float32).to(self.device)}
            #     self.buffer.firsts_trajs[0] = 1
    #         if verbose: 
    #             log.info(f"self.reset_at_iteration={self.reset_at_iteration}, rself.eval_mode={self.eval_mode}, self.last_itr_eval={self.last_itr_eval} reset. ")
    #     else:
    #         # if done at the end of last iteration, the envs are just automatially reset, and we automatically come to a new episode. 
    #         self.buffer.firsts_trajs[0] = self.buffer.done_venv
    #         if verbose: 
    #             log.info(f"into case 2, do not reset. return previous observation. ")
    #         initial_obs_venv = prev_obs_venv
    #     return initial_obs_venv
    
    
    def summarize_iteration_reward(self, reward_trajs:torch.Tensor, firsts_trajs:torch.Tensor):
        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. 
        # Only count episodes that finish within the iteration
        
        # calculate the start and end of episodes for each environment
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = torch.nonzero(firsts_trajs[:, env_ind] == 1, as_tuple=True)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i].item()
                end = env_steps[i + 1].item()
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        # for each valid episode
        if len(episodes_start_end) > 0:
            # print(f"episodes_start_end={len(episodes_start_end)}, {episodes_start_end[0]}")
            # print(f"reward_trajs={len(reward_trajs)}, {reward_trajs[0]}")
                
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            self.episode_reward = torch.tensor([torch.sum(reward_traj) for reward_traj in reward_trajs_split])
            
            if self.furniture_sparse_reward:  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = self.episode_reward
            else:
                episode_best_reward = torch.tensor([
                    torch.max(reward_traj) / self.act_steps
                    for reward_traj in reward_trajs_split
                ])
            self.avg_episode_reward = torch.mean(self.episode_reward.float())
            self.avg_best_reward = torch.mean(episode_best_reward.float())
            self.success_rate = torch.mean((episode_best_reward >= self.best_reward_threshold_for_success).float())
        else:
            self.episode_reward = torch.tensor([])
            self.num_episode_finished = 0
            self.avg_episode_reward = torch.tensor(0.0)
            self.avg_best_reward = torch.tensor(0.0)
            self.success_rate = torch.tensor(0.0)
            # log.info("[WARNING] No episode completed within the iteration!")

        if self.eval_mode and self.avg_episode_reward > self.avg_episode_reward_best:
            self.avg_episode_reward_best = self.avg_episode_reward
            self.is_best_so_far = True