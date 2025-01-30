
import numpy as np

def read_eval_statistics(npz_file_path):
    # Load the .npz file
    loaded_data = np.load(npz_file_path)

    # Extract the structured array
    data = loaded_data['data']

    # Extract individual lists
    num_denoising_steps_list = data['num_denoising_steps']
    avg_single_step_freq_list= data['avg_single_step_freq']
    avg_single_step_freq_std_list = data['avg_single_step_freq_std']
    avg_traj_length_list = data['avg_traj_length']
    avg_traj_length_list_std = data['avg_traj_length_std']
    avg_episode_reward_list = data['avg_episode_reward']
    avg_best_reward_list = data['avg_best_reward']
    avg_episode_reward_std_list = data['avg_episode_reward_std']
    avg_best_reward_std_list = data['avg_best_reward_std']
    success_rate_list = data['success_rate']
    success_rate_list_std = data['success_rate_std']
    num_episodes_finished_list = data['num_episodes_finished']
    
    # return all these list
    eval_statistics=(num_denoising_steps_list, avg_single_step_freq_list, avg_single_step_freq_std_list, \
        avg_traj_length_list, avg_traj_length_list_std, avg_episode_reward_list, avg_best_reward_list, \
            avg_episode_reward_std_list, avg_best_reward_std_list, \
                success_rate_list, success_rate_list_std, num_episodes_finished_list)
    return eval_statistics
