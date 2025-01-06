"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.
self.model: Flow
"""
import os
import numpy as np
import torch
import logging
log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent
from model.flow.reflow import ReFlow

class EvalReFlowAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.record_video =False #True
        self.record_env_index=0
        self.render_onscreen =not self.record_video #False
        self.denoising_steps =[1] # [20] # [1,2,3,4,5,6,8,10,12,14,16,18,20] #32,64,128,256,512
        self.denoising_steps_trained = self.model.max_denoising_steps # actually this is meaning less for reflow. it could be infinity. 
        self.model.show_inference_process = False #True # whether to print each integration step during sampling. 
        ####################################################################################
    def infer(self, cond:dict, num_denoising_steps:int):
        ################################################      overload        #########################################################
        self.model: ReFlow
        samples = self.model.sample(cond=cond, inference_steps=num_denoising_steps, inference_batch_size=self.cfg.env.n_envs, record_intermediate=False)               
        # samples.trajectories: (inference_batch_size, self.horizon_steps, self.action_dim)
        return samples
    
    