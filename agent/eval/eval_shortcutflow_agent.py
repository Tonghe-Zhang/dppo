"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.
self.model: ShortCutFlow

"""
from tqdm import tqdm as tqdm
import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
import model
from model.flow.shortcut_flow import ShortCutFlow


# Save the figure
def current_time():
    from datetime import datetime
    # Get current time
    now = datetime.now()
    # Format the time to the desired pattern
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
    return formatted_time
        
class EvalShortCutFlowAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.inference_steps=cfg.denoising_steps
        self.cfg=cfg
        self.base_policy_path=cfg.base_policy_path
        self.eval_log_dir=None
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
    
    def run(self):
        # Start training loop
        import os
        self.eval_log_dir =f'agent/eval/visualize/shortcutflow/{current_time()}/'
        os.makedirs(self.eval_log_dir, exist_ok=True)
        
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        
        self.model: ShortCutFlow
        data = torch.load(self.base_policy_path, weights_only=True)
        epoch = data["epoch"]
        self.model.load_state_dict(data["model"])    #
        # self.ema_model.load_state_dict(data["ema"])
        print(f"Loaded dict from {self.base_policy_path}")
        
        
        denoising_steps_set = [1,2,4,8,16,32,64,128,256,512]
        import matplotlib.pyplot as plt
        # Lists to store the results
        num_denoising_steps_list = []
        duration_list = []
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
            num_denoising_steps, duration, avg_traj_length, avg_episode_reward, avg_episode_reward_std, avg_best_reward, avg_best_reward_std, num_episodes_finished, success_rate = result
            # Store the relevant results
            num_denoising_steps_list.append(num_denoising_steps)
            duration_list.append(duration)
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
            ('duration', float),
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
        data['duration'] = duration_list
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
    def read_eval_statistics(self, npz_file_path):
        import numpy as np
        import os

        # Load the .npz file
        loaded_data = np.load(npz_file_path)

        # Extract the structured array
        data = loaded_data['data']

        # Extract individual lists
        num_denoising_steps_list = data['num_denoising_steps']
        duration_list = data['duration']
        avg_traj_length_list = data['avg_traj_length']
        avg_episode_reward_list = data['avg_episode_reward']
        avg_best_reward_list = data['avg_best_reward']
        avg_episode_reward_std_list = data['avg_episode_reward_std']
        avg_best_reward_std_list = data['avg_best_reward_std']
        success_rate_list = data['success_rate']
        num_episodes_finished_list = data['num_episodes_finished']
        
        # return all these list
        eval_statistics=(num_denoising_steps_list, duration_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list)
        return eval_statistics
    
    def plot_eval_statistics(self, eval_statistics, log_dir:str):
        num_denoising_steps_list, duration_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list = eval_statistics
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
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 2)
        plt.semilogx(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
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
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

        # Plot success rate
        plt.subplot(2, 3, 5)
        plt.semilogx(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()

        # Plot inference time
        plt.subplot(2, 3, 3)
        plt.semilogx(num_denoising_steps_list, duration_list, marker='o', label='Duration', color='brown')
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Duration ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Duration')
        plt.grid(True)
        plt.legend()

        ''' # this is the number of episodes in the trial, which is inversly related to the trajectory length. kind of like redundant information. 
        plt.subplot(2, 3, 6)
        plt.semilogx(num_denoising_steps_list, num_episodes_finished_list, marker='o', label='Duration', color='skyblue')
        plt.axvline(x=self.model.max_denoising_steps,  color='black', linestyle='--', label='Training Steps')
        plt.title('Finished Episodes ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Finished Episodes')
        plt.grid(True)
        plt.legend()
        '''
        plt.suptitle('ShortCutFlow Policy with Varying Denoising Steps', fontsize=25)
        plt.tight_layout()

        fig_path =os.path.join(log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"figure saved to {fig_path}")
        plt.close()  # Close the figure to free up memory
    
    def single_run(self, num_denoising_steps, options_venv, record_env_index=None):
        import cv2
        # Initialize video writer if recording is enabled
        video_writer = None
        if record_env_index is not None:
            frame_width = self.venv.observation_space.shape[1]
            frame_height = self.venv.observation_space.shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
            video_path =os.path.join(self.eval_log_dir, f'step_{num_denoising_steps}.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
        
        timer = Timer()
        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        from tqdm import tqdm as tqdm
        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps), dynamic_ncols=True):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                
                self.model: ShortCutFlow
                samples = self.model.sample(cond=cond, inference_steps=num_denoising_steps)               
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
            
            # Record video frame if recording is enabled
            if video_writer is not None:
                frame = obs_venv["state"][record_env_index]  # Assuming "state" contains the frame
                frame = (frame * 255).astype(np.uint8)  # Convert to uint8 if necessary
                video_writer.write(frame)
            
            # update for next step
            prev_obs_venv = obs_venv
        
        # Release video writer if recording is enabled
        if video_writer is not None:
            video_writer.release()

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
                )
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
        duration = timer()
        log.info(
                f"denois_step:{num_denoising_steps} | time: {duration:3.2f} s | avg_traj_length {avg_traj_length} | avg episode reward {avg_episode_reward:8.1f}±{avg_episode_reward_std:2.1f} | avg best reward {avg_best_reward:8.1f}±{avg_best_reward_std:2.1f} | num episode {num_episodes_finished:4d} | success rate {success_rate:8.4f} | "
            )
        
        return num_denoising_steps, duration, avg_traj_length, avg_episode_reward, avg_episode_reward_std, avg_best_reward, avg_best_reward_std, num_episodes_finished, success_rate