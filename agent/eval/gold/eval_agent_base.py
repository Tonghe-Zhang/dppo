"""
Parent eval agent class.

"""

import os
import numpy as np
import torch
import hydra
import logging
import random
from tqdm import tqdm as tqdm
from util.timer import Timer
log = logging.getLogger(__name__)
from env.gym_utils import make_async
from omegaconf import DictConfig, OmegaConf


class EvalAgent:
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.base_policy_path=cfg.base_policy_path
        self.eval_log_dir=None
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        ############ could be overload #############
        self.record_video =False
        self.record_env_index=-1
        self.render_onscreen =False
        self.denoising_steps = None
        self.denoising_steps_trained = None
        ############################################
        
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
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )  # furniture specific, for best reward calculation

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        
        # Eval params
        self.n_steps = cfg.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.render_num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"

    
    def run(self):
        # Start training loop
        import os
        self.eval_log_dir =f'agent/eval/visualize/{self.model.__class__.__name__}/{self.env_name}/{self.current_time()}/'
        os.makedirs(self.eval_log_dir, exist_ok=True)
        # Dump the configuration to the YAML file
        cfg_path = self.eval_log_dir + "/cfg.yaml"
        with open(cfg_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        print(f"Configuration saved to {cfg_path}")
        
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        denoising_steps_set = self.denoising_steps
        import matplotlib.pyplot as plt
        # Lists to store the results
        num_denoising_steps_list = []
        avg_single_step_freq_list = []
        avg_single_step_freq_std_list = []
        avg_traj_length_list = []
        avg_episode_reward_list = []
        avg_episode_reward_std_list=[]
        avg_best_reward_list=[]
        avg_best_reward_std_list = []
        success_rate_list = []
        num_episodes_finished_list=[]
        
        for num_denoising_steps in denoising_steps_set:
            self.venv.reset()
            result = self.single_run(num_denoising_steps, options_venv)
            # Unpack the result tuple
            
            num_denoising_steps, avg_single_step_freq, avg_single_step_freq_std, \
                avg_traj_length, avg_episode_reward, avg_episode_reward_std, \
                    avg_best_reward, avg_best_reward_std, num_episodes_finished, success_rate = result
            # Store the relevant results
            num_denoising_steps_list.append(num_denoising_steps)
            avg_single_step_freq_list.append(avg_single_step_freq)
            avg_single_step_freq_std_list.append(avg_single_step_freq_std)
            avg_traj_length_list.append(avg_traj_length)
            avg_episode_reward_list.append(avg_episode_reward)
            avg_best_reward_list.append(avg_best_reward)
            avg_episode_reward_std_list.append(avg_episode_reward_std)
            avg_best_reward_std_list.append(avg_best_reward_std)
            success_rate_list.append(success_rate)
            num_episodes_finished_list.append(num_episodes_finished)

        # save evaluation statistics as an npz
        dtype = [
            ('num_denoising_steps', int),
            ('avg_single_step_freq', float),
            ('avg_single_step_freq_std', float),
            ('avg_traj_length', float),
            ('avg_episode_reward', float),
            ('avg_best_reward', float),
            ('avg_episode_reward_std', float),
            ('avg_best_reward_std', float),
            ('success_rate', float),
            ('num_episodes_finished', int)
        ]

        data = np.zeros(len(num_denoising_steps_list), dtype=dtype)
        data['num_denoising_steps'] = num_denoising_steps_list
        data['avg_single_step_freq'] = avg_single_step_freq_list
        data['avg_single_step_freq_std'] = avg_single_step_freq_std_list
        data['avg_traj_length'] = avg_traj_length_list
        data['avg_episode_reward'] = avg_episode_reward_list
        data['avg_best_reward'] = avg_best_reward_list
        data['avg_episode_reward_std'] = avg_episode_reward_std_list
        data['avg_best_reward_std'] = avg_best_reward_std_list
        data['success_rate'] = success_rate_list
        data['num_episodes_finished'] = num_episodes_finished_list

        # Save the structured array to a file
        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')
        np.savez(eval_statistics_path, data=data)
        
        # read out and plot
        statistics = self.read_eval_statistics(npz_file_path=eval_statistics_path)
        self.plot_eval_statistics(statistics, self.eval_log_dir)

    def single_run(self, num_denoising_steps, options_venv):
        import cv2
        # Initialize video writer if recording is enabled
        video_writer = None
        if self.record_video:
            frame_width = 640
            frame_height = 480
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
            video_path =os.path.join(self.eval_log_dir, f'{self.model.__class__.__name__}_{self.env_name}_step{num_denoising_steps}.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps"
        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        single_step_duration_list = np.zeros(self.n_steps)
        
        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps), dynamic_ncols=True):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                ####################################################################################################################################################
                timer = Timer()
                samples = self.infer(cond, num_denoising_steps)
                single_step_duration = timer()
                single_step_duration_list[step] = single_step_duration
                ####################################################################################################################################################
                
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            ####################################################################################################################################################
            
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            
            if self.record_video:
                frame_tuple = self.venv.render(mode='rgb_array', height=frame_height, width=frame_width) #type=<class 'tuple'>, len(red)=1, ret[0] is a <class 'numpy.ndarray'> with shape (480, 640, 3) 
                # print(f"frame_tuple: type={type(frame_tuple)}, shape={len(frame_tuple)}, {frame_tuple[0].shape}, element type: {type(frame_tuple[0])}")
                # '''
                # this is a tuple with len(ret) = cfg.env.n_envs, each element frame_tuple[i] is what the i-th parallel environment records. 
                # to save memory we record only the environment specified by self.record_env_index. 
                # if you do not specify the height and width, then calling self.venv.render(mode='rgb_array') gives you ret[0] is a (500, 500, 3) uint8 numpy.ndarray. 
                # the reason it is of (500, 500) because in \\wsl.localhost\Ubuntu2204\root\anaconda3\envs\mujoco_py\lib\python3.10\site-packages\gym\envs\mujoco\mujoco_env.py
                # the default image size is DEFAULT_SIZE=500. 
                # '''
            if self.render_onscreen:
                self.venv.render(mode='human') # only show to the screen. this is made possible only when you have an actual screen. 
            # import inspect
            # example_function =self.venv.render
            # file_location = inspect.getfile(example_function)
            # source_code = inspect.getsource(example_function)
            # print(f"Function is defined in: {file_location}")
            # print(f"Source code of the function:{source_code}")
            # exit(0)
            # rgb_array = self.venv.render(mode='rgb_array')  # Use 'human' mode for on-screen rendering
            # print(rgb_array)
            
            
            ####################################################################################################################################################
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            
            # Record video frame if recording is enabled
            if video_writer is not None:
                frame = frame_tuple[self.record_env_index]
                # Add title to the frame
                cv2.putText(frame, self.video_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                video_writer.write(frame)

                # Add current reward to the frame
                # cv2.putText(frame, f"Reward: {reward_venv[self.record_env_index]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # video_writer.write(frame)
    
            # update for next step
            prev_obs_venv = obs_venv
        
        # Release video writer if recording is enabled
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to {video_path}")

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
            num_episodes_finished = len(reward_trajs_split)
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
                ) # single step best rewards in all the episodes finished by each environment
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)
            
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )

        else:
            episode_reward = np.array([])
            num_episodes_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")
        avg_traj_length=self.get_traj_length(episodes_start_end)
        # Log loss and save metrics
        
        
        # convert time to frequency
        ##################################################################################################################################
        single_step_frequency_list = 1/single_step_duration_list
        
        avg_single_step_freq =single_step_frequency_list.mean()
        avg_single_step_freq_std =single_step_frequency_list.std()
        ##################################################################################################################################
        
        log.info(
                f"""
            ########################################
            denois_step:         {num_denoising_steps}
            avg_single_step_freq:{avg_single_step_freq:3.2f} ± {avg_single_step_freq_std:3.2f} HZ
            avg_traj_length:     {avg_traj_length}
            avg_episode_reward:  {avg_episode_reward:8.1f} ± {avg_episode_reward_std:2.1f}
            avg_best_reward:     {avg_best_reward:8.1f} ± {avg_best_reward_std:2.1f}
            num_episode:         {num_episodes_finished:4d} | success_rate: {success_rate:8.4f}
            ########################################
            """
            )

        
        return num_denoising_steps, \
            avg_single_step_freq, avg_single_step_freq_std,\
            avg_traj_length, \
            avg_episode_reward, avg_episode_reward_std, \
            avg_best_reward, avg_best_reward_std, \
            num_episodes_finished, success_rate
    
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
    def read_eval_statistics(self, npz_file_path):
        import numpy as np
        import os

        # Load the .npz file
        loaded_data = np.load(npz_file_path)

        # Extract the structured array
        data = loaded_data['data']

        # Extract individual lists
        num_denoising_steps_list = data['num_denoising_steps']
        avg_single_step_freq_list= data['avg_single_step_freq']
        avg_single_step_freq_std_list = data['avg_single_step_freq_std']
        avg_traj_length_list = data['avg_traj_length']
        avg_episode_reward_list = data['avg_episode_reward']
        avg_best_reward_list = data['avg_best_reward']
        avg_episode_reward_std_list = data['avg_episode_reward_std']
        avg_best_reward_std_list = data['avg_best_reward_std']
        success_rate_list = data['success_rate']
        num_episodes_finished_list = data['num_episodes_finished']
        
        # return all these list
        eval_statistics=(num_denoising_steps_list, avg_single_step_freq_list, avg_single_step_freq_std_list, \
            avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, \
                avg_episode_reward_std_list, avg_best_reward_std_list, \
                    success_rate_list, num_episodes_finished_list)
        return eval_statistics
    
    def plot_eval_statistics(self, eval_statistics, log_dir:str):
        num_denoising_steps_list, avg_single_step_freq_list, avg_single_step_freq_std_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list = eval_statistics
        import matplotlib.pyplot as plt
        import os
        # Plotting
        plt.figure(figsize=(12, 8))

        # Plot average episode reward with shading
        plt.subplot(2, 3, 1)
        plt.semilogx(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
        plt.fill_between(num_denoising_steps_list,
                    [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                    [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                    color='b', alpha=0.2, label='Std Dev')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 2)
        plt.semilogx(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Trajectory Length ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Trajectory Length')
        plt.grid(True)
        plt.legend()

        # Plot average best reward with shading
        plt.subplot(2, 3, 4)
        plt.semilogx(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
        plt.fill_between(num_denoising_steps_list,
                    [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                    [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                    color='g', alpha=0.2, label='Std Dev')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

        # Plot success rate
        plt.subplot(2, 3, 5)
        plt.semilogx(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()

        # Plot inference time
        plt.subplot(2, 3, 3)
        plt.semilogx(num_denoising_steps_list, avg_single_step_freq_list, marker='o', label='Frequency', color='brown')
        plt.fill_between(num_denoising_steps_list,
                    [duration - std for duration, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                    [duration + std for duration, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                    color='brown', alpha=0.2, label='Std Dev')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Frequency ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Frequency')
        plt.grid(True)
        plt.legend()

        ''' # this is the number of episodes in the trial, which is inversly related to the trajectory length. kind of like redundant information. 
        plt.subplot(2, 3, 6)
        plt.semilogx(num_denoising_steps_list, num_episodes_finished_list, marker='o', label='Duration', color='skyblue')
        plt.axvline(x=self.denoising_steps_trained,  color='black', linestyle='--', label='Training Steps')
        plt.title('Finished Episodes ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Finished Episodes')
        plt.grid(True)
        plt.legend()
        '''
        plt.suptitle(f'{self.model.__class__.__name__}, {self.env_name}', fontsize=25)
        plt.tight_layout()

        fig_path =os.path.join(log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"figure saved to {fig_path}")
        plt.close()  # Close the figure to free up memory
    
    def current_time(self):
        from datetime import datetime
        # Get current time
        now = datetime.now()
        # Format the time to the desired pattern
        formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
        return formatted_time
    