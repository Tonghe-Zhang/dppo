import re

# Define regex patterns to capture the statistics
patterns = {
    'denois_step': re.compile(r'denois_step:\s+(\d+)'),
    'avg_single_step_freq': re.compile(r'avg_single_step_freq:\s+([\d.]+) ± ([\d.]+) HZ'),
    'avg_traj_length': re.compile(r'avg_traj_length:\s+([\d.]+)'),
    'avg_episode_reward': re.compile(r'avg_episode_reward:\s+([\d.]+) ± ([\d.]+)'),
    'avg_best_reward': re.compile(r'avg_best_reward:\s+([\d.]+) ± ([\d.]+)'),
    'num_episode': re.compile(r'num_episode:\s+(\d+)'),
    'success_rate': re.compile(r'success_rate:\s+([\d.]+)')
}
log_data = """
[2024-12-12 12:53:00,095][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         1
            avg_single_step_freq:43.66 ± 27.42 HZ
            avg_traj_length:     109.43739565943238
            avg_episode_reward:    1376.4 ± 172.5
            avg_best_reward:          4.8 ± 0.4
            num_episode:          599 | success_rate:   1.0000
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:10<00:00,  7.08it/s]
[2024-12-12 12:54:10,776][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         2
            avg_single_step_freq:24.81 ± 9.00 HZ
            avg_traj_length:     34.41103992114342
            avg_episode_reward:     309.8 ± 167.8
            avg_best_reward:          3.1 ± 0.8
            num_episode:         2029 | success_rate:   0.6087
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:34<00:00,  5.27it/s]
[2024-12-12 12:55:45,791][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         3
            avg_single_step_freq:26.89 ± 8.67 HZ
            avg_traj_length:     68.11959798994975
            avg_episode_reward:     794.6 ± 287.9
            avg_best_reward:          4.6 ± 0.9
            num_episode:          995 | success_rate:   0.9045
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:42<00:00,  4.88it/s]
[2024-12-12 12:57:28,335][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         4
            avg_single_step_freq:29.47 ± 9.69 HZ
            avg_traj_length:     87.9817232375979
            avg_episode_reward:    1070.0 ± 263.8
            avg_best_reward:          4.9 ± 0.6
            num_episode:          766 | success_rate:   0.9843
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:57<00:00,  4.27it/s]
[2024-12-12 12:59:25,617][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         5
            avg_single_step_freq:29.19 ± 9.44 HZ
            avg_traj_length:     99.54122938530735
            avg_episode_reward:    1226.5 ± 316.5
            avg_best_reward:          4.9 ± 0.5
            num_episode:          667 | success_rate:   0.9880
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [02:20<00:00,  3.55it/s]
[2024-12-12 13:01:46,422][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         6
            avg_single_step_freq:29.50 ± 9.95 HZ
            avg_traj_length:     103.296875
            avg_episode_reward:    1276.9 ± 351.4
            avg_best_reward:          4.9 ± 0.4
            num_episode:          640 | success_rate:   0.9922
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [02:55<00:00,  2.85it/s]
[2024-12-12 13:04:41,921][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         8
            avg_single_step_freq:26.03 ± 8.64 HZ
            avg_traj_length:     108.69016393442622
            avg_episode_reward:    1352.8 ± 370.9
            avg_best_reward:          4.9 ± 0.4
            num_episode:          610 | success_rate:   0.9967
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [03:38<00:00,  2.29it/s]
[2024-12-12 13:08:20,133][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         10
            avg_single_step_freq:24.57 ± 9.52 HZ
            avg_traj_length:     111.77029360967185
            avg_episode_reward:    1394.2 ± 386.0
            avg_best_reward:          4.9 ± 0.4
            num_episode:          579 | success_rate:   0.9983
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [04:18<00:00,  1.94it/s]
[2024-12-12 13:12:38,371][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         12
            avg_single_step_freq:26.75 ± 9.30 HZ
            avg_traj_length:     111.53012048192771
            avg_episode_reward:    1391.8 ± 369.6
            avg_best_reward:          4.9 ± 0.4
            num_episode:          581 | success_rate:   1.0000
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [05:12<00:00,  1.60it/s]
[2024-12-12 13:17:50,790][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         14
            avg_single_step_freq:25.03 ± 8.16 HZ
            avg_traj_length:     111.74957118353345
            avg_episode_reward:    1396.2 ± 355.3
            avg_best_reward:          4.9 ± 0.4
            num_episode:          583 | success_rate:   1.0000
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [05:14<00:00,  1.59it/s]
[2024-12-12 13:23:05,250][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         16
            avg_single_step_freq:25.25 ± 7.98 HZ
            avg_traj_length:     111.58503401360544
            avg_episode_reward:    1393.0 ± 398.6
            avg_best_reward:          4.9 ± 0.4
            num_episode:          588 | success_rate:   0.9949
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [06:11<00:00,  1.35it/s]
[2024-12-12 13:29:17,077][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         18
            avg_single_step_freq:24.28 ± 8.38 HZ
            avg_traj_length:     111.61224489795919
            avg_episode_reward:    1394.6 ± 375.0
            avg_best_reward:          4.9 ± 0.4
            num_episode:          588 | success_rate:   0.9983
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [06:35<00:00,  1.26it/s]
[2024-12-12 13:35:52,965][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         20
            avg_single_step_freq:23.60 ± 8.36 HZ
            avg_traj_length:     110.72250423011845
            avg_episode_reward:    1382.0 ± 358.0
            avg_best_reward:          4.9 ± 0.4
            num_episode:          591 | success_rate:   0.9983
            ########################################

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [10:17<00:00,  1.23s/it]
[2024-12-12 13:46:10,421][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         32
            avg_single_step_freq:20.20 ± 8.06 HZ
            avg_traj_length:     110.56734006734007
            avg_episode_reward:    1378.9 ± 410.8
            avg_best_reward:          4.9 ± 0.4
            num_episode:          594 | success_rate:   0.9983
            ########################################

 15%|█████████████████▏                                                                                                                                      80%|█████████████████████████████████████████████████████████████████▉                | 402/500                                                             82%|███████████████████████████████████████████████████████████           | 412/500 [16:10<03:58,  2.71s/it]                                                       100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [19:46<00:00,  2.37s/it]
[2024-12-12 14:05:57,312][agent.eval.eval_agent_base][INFO] -
            ########################################
            denois_step:         64
            avg_single_step_freq:18.13 ± 6.78 HZ
            avg_traj_length:     106.15322580645162
            avg_episode_reward:    1321.5 ± 356.3
            avg_best_reward:          4.9 ± 0.4
            num_episode:          620 | success_rate:   0.9968
            ########################################
"""

# Initialize lists to store the extracted data
num_denoising_steps_list = []
avg_single_step_freq_list = []
avg_single_step_freq_std_list = []
avg_traj_length_list = []
avg_episode_reward_list = []
avg_episode_reward_std_list = []
avg_best_reward_list = []
avg_best_reward_std_list = []
success_rate_list = []
num_episodes_finished_list = []

# Split the log data into individual entries
log_entries = log_data.split('[2024-12-12')[1:]

# Parse each log entry
for entry in log_entries:
    denois_step = int(patterns['denois_step'].search(entry).group(1))
    avg_single_step_freq, avg_single_step_freq_std = map(float, patterns['avg_single_step_freq'].search(entry).groups())
    avg_traj_length = float(patterns['avg_traj_length'].search(entry).group(1))
    avg_episode_reward, avg_episode_reward_std = map(float, patterns['avg_episode_reward'].search(entry).groups())
    avg_best_reward, avg_best_reward_std = map(float, patterns['avg_best_reward'].search(entry).groups())
    num_episodes_finished = int(patterns['num_episode'].search(entry).group(1))
    success_rate = float(patterns['success_rate'].search(entry).group(1))

    # Append the extracted data to the lists
    num_denoising_steps_list.append(denois_step)
    avg_single_step_freq_list.append(avg_single_step_freq)
    avg_single_step_freq_std_list.append(avg_single_step_freq_std)
    avg_traj_length_list.append(avg_traj_length)
    avg_episode_reward_list.append(avg_episode_reward)
    avg_episode_reward_std_list.append(avg_episode_reward_std)
    avg_best_reward_list.append(avg_best_reward)
    avg_best_reward_std_list.append(avg_best_reward_std)
    success_rate_list.append(success_rate)
    num_episodes_finished_list.append(num_episodes_finished)

# Print the extracted data
print("num_denoising_steps_list:", num_denoising_steps_list)
print("avg_single_step_freq_list:", avg_single_step_freq_list)
print("avg_single_step_freq_std_list:", avg_single_step_freq_std_list)
print("avg_traj_length_list:", avg_traj_length_list)
print("avg_episode_reward_list:", avg_episode_reward_list)
print("avg_episode_reward_std_list:", avg_episode_reward_std_list)
print("avg_best_reward_list:", avg_best_reward_list)
print("avg_best_reward_std_list:", avg_best_reward_std_list)
print("success_rate_list:", success_rate_list)
print("num_episodes_finished_list:", num_episodes_finished_list)





# Prepare the evaluation statistics
eval_statistics = (
    num_denoising_steps_list, avg_single_step_freq_list, avg_single_step_freq_std_list,
    avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list,
    avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list,
    num_episodes_finished_list
)

def plot_eval_statistics(self, eval_statistics, log_dir: str):
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
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
    plt.title('Average Episode Reward ')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    plt.legend()

    # Plot average trajectory length
    plt.subplot(2, 3, 2)
    plt.semilogx(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
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
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
    plt.title('Average Best Reward ')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Best Reward')
    plt.grid(True)
    plt.legend()

    # Plot success rate
    plt.subplot(2, 3, 5)
    plt.semilogx(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
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
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
    plt.title('Inference Frequency ')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference Frequency')
    plt.grid(True)
    plt.legend()

    ''' # this is the number of episodes in the trial, which is inversly related to the trajectory length. kind of like redundant information.
    plt.subplot(2, 3, 6)
    plt.semilogx(num_denoising_steps_list, num_episodes_finished_list, marker='o', label='Duration', color='skyblue')
    plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
    plt.title('Finished Episodes ')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Finished Episodes')
    plt.grid(True)
    plt.legend()
    '''
    plt.suptitle(f'{self.model.__class__.__name__} Policy with Varying Denoising Steps', fontsize=25)
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



# Call the plot_eval_statistics function
plot_eval_statistics(eval_statistics, log_dir='/home/zhangtonghe/0-dppo-video-mujoco/agent/eval/visualize/ReFlow')