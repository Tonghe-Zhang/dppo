"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.
self.model: Diffusion
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/gym/eval/hopper-v2
"""
from tqdm import tqdm as tqdm
import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent_base import EvalAgent
import model
from model.diffusion.diffusion import DiffusionModel

# Save the figure

    
class EvalDiffusionAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.record_video =True # False
        self.record_env_index=0
        self.render_onscreen =False #True
        self.denoising_steps =[20] #[1] #[1,2,4,8,16,32,64,128,256,512]
        self.denoising_steps_trained = self.cfg.denoising_steps
        ################################################################################################################################
    def infer(self,cond:dict, num_denoising_steps:int):
        ################################################      overload        #########################################################
        self.model: DiffusionModel
        self.model.denoising_steps = num_denoising_steps 
        self.model.calculate_parameters() # reset all the ddpm parameters based on number of generation steps. 
        samples = self.model.forward(cond=cond, deterministic=True)
        return samples
        ################################################################################################################################
        