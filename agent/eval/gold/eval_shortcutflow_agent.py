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
from agent.eval.eval_agent_base import EvalAgent
from model.flow.shortcut_flow import ShortCutFlow

class EvalShortCutFlowAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.record_video =True
        self.record_env_index=0
        self.render_onscreen =False
        self.denoising_steps =[128] # [1,2,4,8,16,32,64,128,256,512]
        self.denoising_steps_trained = self.model.max_denoising_steps # actually this is meaning less for reflow. it could be infinity. 
        ####################################################################################
    def infer(self, cond, num_denoising_steps):
        ################################################      overload        #########################################################
        self.model: ShortCutFlow
        samples = self.model.sample(cond=cond, inference_steps=num_denoising_steps)               
        return samples