import numpy as np
import cv2
import gym
import os
# export PYTHONPATH=$PYTHONPATH:/mnt/d/Research/0-Robotics/Grad_Thesis_Diffusion_Robots/1-code/0-dppo-video-mujoco
# export PYTHONPATH=$PYTHONPATH:/mnt/d/Research/0-Robotics/Grad_Thesis_Diffusion_Robots/1-code/0-dppo-video-mujoco/model
import sys
import logging
import math
import hydra
from omegaconf import OmegaConf
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)
# Force CPU rendering
os.environ['MUJOCO_GL'] = 'osmesa'

from agent.eval.eval_diffusion_agent import EvalDiffusionAgent




@hydra.main(
    version_base=None,
    config_path='../cfg/gym/eval/hopper-v2',
    config_name='eval_diffusion_mlp') 
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    
    
    agent: EvalDiffusionAgent
    num_denoising_steps =20
    
    env = gym.make('Hopper-v3')
    observation = env.reset()
    while True:
        sample = agent.infer(cond={"state": observation}, num_denoising_steps=num_denoising_steps)
        
        action = sample.trajectory[0]
        
        observation, reward, done, info = env.step(action)

        env.render(mode='human')  # Use 'human' mode for on-screen rendering

        if done:
            observation = env.reset()  # Reset environment if done

    env.close()

if __name__ == "__main__":
    main()




 