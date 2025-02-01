import matplotlib.pyplot as plt 
import os
from util.timer import current_time
from util.process import read_eval_statistics

    
def plot_eval_statistics(eval_statistics_pretrain,
                         eval_statistics_finetune,
                         inference_steps, 
                         model_name = 'ReFlow',
                         env_name = 'hopper-medium-v2',
                         log_dir=None):
    
    # Extract metrics for Pretrained model
    num_denoising_steps_list_pretrain, avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain, avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain, avg_episode_reward_list_pretrain, avg_best_reward_list_pretrain, \
    avg_episode_reward_std_list_pretrain, avg_best_reward_std_list_pretrain, success_rate_list_pretrain, success_rate_list_std_pretrain, num_episodes_finished_list_pretrain = eval_statistics_pretrain

    # Extract metrics for Finetuned model
    num_denoising_steps_list_finetune, avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune, avg_traj_length_list_finetune, avg_traj_length_list_std_finetune,avg_episode_reward_list_finetune, avg_best_reward_list_finetune, \
    avg_episode_reward_std_list_finetune, avg_best_reward_std_list_finetune, success_rate_list_finetune, success_rate_list_std_finetune, num_episodes_finished_list_finetune = eval_statistics_finetune

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot average episode reward with shading
    plt.subplot(2, 3, 1)
    plt.semilogx(num_denoising_steps_list_pretrain, avg_episode_reward_list_pretrain, marker='o', label='Pretrained', color='black')
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_pretrain, avg_episode_reward_std_list_pretrain)],
                [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_pretrain, avg_episode_reward_std_list_pretrain)],
                color='black', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_finetune, avg_episode_reward_list_finetune, marker='o', label='Finetuned', color='red')
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_finetune, avg_episode_reward_std_list_finetune)],
                [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_finetune, avg_episode_reward_std_list_finetune)],
                color='red', alpha=0.2)
    plt.axvline(x=inference_steps,  color='red', linestyle='--', label=f'Finetune Steps ({inference_steps})')
    plt.title('Average Episode Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    plt.legend()

    # Plot average trajectory length
    plt.subplot(2, 3, 2)
    plt.semilogx(num_denoising_steps_list_pretrain, avg_traj_length_list_pretrain, marker='o', label='Pretrained', color='black')
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain)],
                [avg_best + std for avg_best, std in zip(avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain)],
                color='black', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_finetune, avg_traj_length_list_finetune, marker='o', label='Finetuned', color='red')
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(avg_traj_length_list_finetune, avg_traj_length_list_std_finetune)],
                [avg_best + std for avg_best, std in zip(avg_traj_length_list_finetune, avg_traj_length_list_std_finetune)],
                color='red', alpha=0.2)
    plt.axvline(x=inference_steps,  color='red', linestyle='--', label=f'Finetune Steps ({inference_steps})')
    plt.title('Average Trajectory Length')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Trajectory Length')
    
    plt.grid(True)
    plt.legend()

    # Plot average best reward with shading
    plt.subplot(2, 3, 4)
    plt.semilogx(num_denoising_steps_list_pretrain, avg_best_reward_list_pretrain, marker='o', label='Pretrained', color='black')
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(avg_best_reward_list_pretrain, avg_best_reward_std_list_pretrain)],
                [avg_best + std for avg_best, std in zip(avg_best_reward_list_pretrain, avg_best_reward_std_list_pretrain)],
                color='black', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_finetune, avg_best_reward_list_finetune, marker='o', label='Finetuned', color='red')
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(avg_best_reward_list_finetune, avg_best_reward_std_list_finetune)],
                [avg_best + std for avg_best, std in zip(avg_best_reward_list_finetune, avg_best_reward_std_list_finetune)],
                color='red', alpha=0.2)
    plt.axvline(x=inference_steps,  color='red', linestyle='--', label=f'Finetune Steps ({inference_steps})')
    plt.title('Average Best Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Best Reward')
    plt.grid(True)
    plt.legend()

    # Plot success rate
    plt.subplot(2, 3, 5)
    plt.semilogx(num_denoising_steps_list_pretrain, success_rate_list_pretrain, marker='o', label='Pretrained', color='black')
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(success_rate_list_pretrain, success_rate_list_std_pretrain)],
                [avg_best + std for avg_best, std in zip(success_rate_list_pretrain, success_rate_list_std_pretrain)],
                color='black', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_finetune, success_rate_list_finetune, marker='o', label='Finetuned', color='red')
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(success_rate_list_finetune, success_rate_list_std_finetune)],
                [avg_best + std for avg_best, std in zip(success_rate_list_finetune, success_rate_list_std_finetune)],
                color='red', alpha=0.2)
    plt.axvline(x=inference_steps,  color='red', linestyle='--', label=f'Finetune Steps ({inference_steps})')
    plt.title('Success Rate')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()

    # Plot inference time
    plt.subplot(2, 3, 3)
    plt.semilogx(num_denoising_steps_list_pretrain, avg_single_step_freq_list_pretrain, marker='o', label='Pretrained', color='black')
    plt.fill_between(num_denoising_steps_list_pretrain,
                [freq - std for freq, std in zip(avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain)],
                [freq + std for freq, std in zip(avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain)],
                color='black', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_finetune, avg_single_step_freq_list_finetune, marker='o', label='Finetuned', color='red')
    plt.fill_between(num_denoising_steps_list_finetune,
                [freq - std for freq, std in zip(avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune)],
                [freq + std for freq, std in zip(avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune)],
                color='red', alpha=0.2)
    plt.axvline(x=inference_steps,  color='red', linestyle='--', label=f'Finetune Steps ({inference_steps})')
    plt.title('Inference Frequency')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference Frequency')
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"{model_name}, {env_name} \n steps = {', '.join(map(str, num_denoising_steps_list_pretrain))}", fontsize=25)
    plt.tight_layout()

    fig_path = os.path.join(log_dir, f'flowppo_compare.png')
    plt.savefig(fig_path)
    print(f"figure saved to {fig_path}")
    plt.close()
    

if __name__ == '__main__':
    
    MODEL_NAME='ReFlow'
    START = 1
    HOPPER=['agent/eval/visualize/flow/25-01-30-21-27-04/eval_statistics.npz',
            'agent/eval/visualize/flow/25-01-30-21-27-22/eval_statistics.npz',
            'hopper-v2']   #hopper-medium-v2
    
    HALFCHEETAH=['agent/eval/visualize/flow/25-02-02-00-24-33/eval_statistics.npz',
                 'agent/eval/visualize/flow/25-02-02-00-26-14/eval_statistics.npz',
                 'halfcheetah-v2'] # halfcheetah-medium-v2
    
    WALKER=['agent/eval/visualize/flow/25-02-02-00-23-14/eval_statistics.npz',
            'agent/eval/visualize/flow/25-02-02-00-23-10/eval_statistics.npz',
            'walker2d-v2'] # walker2d-medium-v2
    
    ANT = ['agent/eval/visualize/flow/25-02-02-00-32-29/eval_statistics.npz',
           'agent/eval/visualize/flow/25-02-02-00-34-27/eval_statistics.npz',
           'ant-v0'] #ant-medium-expert-v0
    
    pretrain_eval_path, finetune_eval_path, ENV_NAME= ANT
    
    
    pretrain_file = read_eval_statistics(pretrain_eval_path, start=START)
    finetune_file = read_eval_statistics(finetune_eval_path, start=START)
    
    logdir = os.path.join('agent','eval','visualize','flow', MODEL_NAME, ENV_NAME, f'{current_time()}')
    os.makedirs(logdir, exist_ok=True)
    inference_step=20
    plot_eval_statistics(pretrain_file, finetune_file, inference_step, MODEL_NAME, ENV_NAME, logdir)
