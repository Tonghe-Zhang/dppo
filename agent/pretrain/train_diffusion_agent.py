"""
Pre-training diffusion policy

> run this command to run silently but save to wandb
>>> nohup python script/run.py --config-name=pre_diffusion_mlp  --config-dir=cfg/gym/pretrain/hopper-medium-v2 >nohup99.log 2>&1 &

> run this command with VERBOSE_LOG=True to run with debugging information and do not save to wandb
>>> python script/run.py --config-name=pre_diffusion_mlp  --config-dir=cfg/gym/pretrain/hopper-medium-v2 wandb=null


"""
import os 
from tqdm import tqdm as tqdm
import logging
import wandb
import numpy as np
from tqdm import tqdm
log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device
import random
import torch

verbose=False
VERBOSE_LOG= True# False# 

class EnvConfig:
    def __init__(self, n_envs, name, max_episode_steps, reset_at_iteration, save_video, best_reward_threshold_for_success, wrappers, n_steps, render_num):
        self.n_envs = n_envs
        self.name = name
        self.max_episode_steps = max_episode_steps
        self.reset_at_iteration = reset_at_iteration
        self.save_video = save_video
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.wrappers = wrappers
        self.n_steps = n_steps
        self.render_num = render_num 

class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Create an instance of EnvConfig with the values from the annotated environment
        env_config = EnvConfig(
            n_envs=50,
            name=cfg.env,
            max_episode_steps=1000,
            reset_at_iteration=False,
            save_video=False,  # Change to True if needed
            best_reward_threshold_for_success=3,
            wrappers={
                "mujoco_locomotion_lowdim": {
                    "normalization_path": f"/home/zhangtonghe/dppo/data/gym/{cfg.env}/normalization.npz"
                },
                "multi_step": {
                    "n_obs_steps": cfg.cond_steps,
                    "n_action_steps": cfg.horizon_steps,
                    "max_episode_steps": 1000,
                    "reset_within_step": True
                }
            },
            n_steps=50,
            render_num=0
        )    

        self.best_reward=0.0
        self.env_name = env_config.name
        env_type = cfg.get("env_type", None)
        from env.gym_utils import make_async
        self.venv = make_async(
            self.env_name,
            env_type=env_type,
            num_envs=env_config.n_envs,
            asynchronous=True,
            max_episode_steps=env_config.max_episode_steps,
            wrappers=env_config.wrappers,
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=False, #env_config.get("use_image_obs", False),
            render=False, # env_config.get("render", False),
            render_offscreen=env_config.save_video,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(env_config.n_envs)]
            )
        self.n_envs = env_config.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.horizon_steps # act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = env_config.max_episode_steps
        self.reset_at_iteration = env_config.reset_at_iteration
        self.furniture_sparse_reward= False
        # Now, replace references to cfg and its parameters with eval_config
        self.n_steps = env_config.n_steps
        self.best_reward_threshold_for_success = env_config.best_reward_threshold_for_success
        
        # Logging, rendering
        self.logdir = cfg.logdir_eval
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = env_config.render_num
        self.render_video = env_config.save_video  # Assuming you want to use the save_video from env_config
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
    
    def save_best_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best.pt")
        torch.save(data, savepath)
        log.info(f"Saved the best model to {savepath}, which has highest avg_episode_reward: {self.best_reward:8.2f}.")

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


    def test(self):
        timer = Timer()
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        
        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps)) if VERBOSE_LOG else range(self.n_steps):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                from model.diffusion.diffusion import DiffusionModel
                self.model: DiffusionModel
                samples = self.model(cond=cond, deterministic=True)
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
                    
        avg_traj_length = 0.0 
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
        
        time = timer()
        log.info(
                f"eval: avg_traj_length {avg_traj_length:4.2f} | num episode {num_episode_finished:4d}/{5} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.1f}±{avg_episode_reward_std:2.1f} | avg best reward {avg_best_reward:8.1f}±{avg_best_reward_std:2.1f} |"
            )
        
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )
        return avg_traj_length, success_rate, avg_episode_reward, avg_episode_reward_std, avg_best_reward, avg_best_reward_std
        
    
    def run(self):
        timer = Timer()
        self.epoch = 1
        for epoch in tqdm(range(self.n_epochs)):
            # train
            loss_train_epoch = []
            # for step, batch_train in tqdm(enumerate(self.dataloader_train)):
            for step, batch_train in enumerate(self.dataloader_train):
                if verbose:
                    print(len(self.dataloader_train))
                    exit()
                    print(batch_train)   
                    print(type(batch_train.actions))
                    print(batch_train.actions.shape)

                    print(type(batch_train.conditions))
                    print(batch_train.conditions.keys())
                    print(batch_train.conditions['state'].shape)
                    exit()
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train = self.model.loss(*batch_train)
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
                if VERBOSE_LOG:
                    print(f"epoch: {epoch}/{self.n_epochs}={epoch/self.n_epochs*100:2.2f}%, steps: {step}, loss: {loss_train.item():3.4}", end="\r")
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            # for RL, self.dataloader_val is None. So you just skip this part. 
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # save model # default is 100 by pre_diffusion_mlp.yaml
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:  # default is every step. 
                # test
                avg_traj_length, suc_rate, avg_episode_r, avg_episode_r_std, avg_best_r, avg_best_r_std=self.test()
                if avg_episode_r > self.best_reward:
                    self.best_reward = avg_episode_r
                    self.save_best_model()
                    log.info(f"Current Best model saved at epoch {self.epoch}")
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                            "avg_traj_length": avg_traj_length,
                            "success_rate": suc_rate,
                            "avg_episode_reward": avg_episode_r,
                            "avg_episode_reward_std": avg_episode_r_std,
                            "avg_best_reward": avg_best_r,
                            "avg_best_reward_std": avg_best_r_std,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "best_reward": self.best_reward
                        },
                        step=self.epoch,
                        commit=True,
                    )
            # count
            self.epoch += 1
