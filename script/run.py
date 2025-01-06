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

@hydra.main(
    version_base=None,
    config_path='../cfg/gym/eval/halfcheetah-v2', #'../cfg/gym/eval/ant-medium-expert-v2',  hopper-v2  halfcheetah-v2 walker2d-medium-v2
    config_name='eval_reflow_mlp')# 'eval_reflow_mlp'
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)
    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
