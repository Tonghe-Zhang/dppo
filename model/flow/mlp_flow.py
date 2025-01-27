"""
MLP models for flow matching with learnable stochastic interpolate noise. 
"""
import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy
from typing import Tuple
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
log = logging.getLogger(__name__)


class FlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim=action_dim
        self.cond_dim=cond_dim
        self.time_dim=time_dim
        self.mlp_dims=mlp_dims
        self.activation_type=activation_type
        self.out_activation_type=out_activation_type
        self.use_layernorm=use_layernorm
        self.residual_style=residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        model = ResidualMLP if residual_style else MLP
        
        # obs encoder
        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        
        # velocity head
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        action,
        time,
        cond,
        output_embedding=False,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, Ta, Da = action.shape

        # flatten action chunk
        action = action.view(B, -1)

        # flatten obs history
        state = cond["state"].view(B, -1)

        # obs encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        
        # time encoder
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        # velocity head
        feature_vel = torch.cat([action, time_emb, cond_emb], dim=-1)
        vel = self.mlp_mean(feature_vel)
        
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_emb
        return vel.view(B, Ta, Da)

class NoisyFlowMLP(nn.Module):
    def __init__(
        self,
        policy:FlowMLP,
        denoising_steps,
        learn_explore_noise_from,
        inital_noise_scheduler_type,
        min_logprob_denoising_std,
        max_logprob_denoising_std,
        learn_explore_time_embedding,
        init_time_embedding,
        time_dim_explore,
        device
    ):
        super().__init__()
        self.device=device
        self.policy = policy.to(self.device)
        ''''
        input:  [batchsize, time_dim + cond_enc_dim]    log \sigma (t,s)
        output: positive tensor of shape [batchsize, self.denoising_steps, self.horizon_steps x self.act_dim]
        '''
        self.denoising_steps: int = denoising_steps
        self.learn_explore_noise_from: int = learn_explore_noise_from
        self.initial_noise_scheduler_type: str = inital_noise_scheduler_type
        self.min_logprob_denoising_std: float = min_logprob_denoising_std
        self.max_logprob_denoising_std: float = max_logprob_denoising_std
        
        self.logvar_min = torch.nn.Parameter(torch.log(torch.tensor(self.min_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
        self.logvar_max = torch.nn.Parameter(torch.log(torch.tensor(self.max_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
        self.learn_explore_time_embedding: bool  = learn_explore_time_embedding
        self.init_time_embedding: bool = init_time_embedding
        self.set_logprob_noise_levels()
        
        input_dim_noise = time_dim_explore + self.policy.cond_enc_dim if self.learn_explore_time_embedding else self.policy.time_dim + self.policy.cond_enc_dim
        
        self.mlp_logvar = MLP(
            [input_dim_noise] + [self.policy.act_dim_total],
            out_activation_type="Identity",
        ).to(self.device)
        
        if self.learn_explore_time_embedding:
            self.time_embedding_explore = nn.Embedding(num_embeddings=self.denoising_steps, 
                                                       embedding_dim = time_dim_explore, 
                                                       device=self.device)
            
            if self.init_time_embedding:
                self.init_embedding(embedding = self.time_embedding_explore, device=self.device, init_type ='lin', k=0.1, b=0.0)
    
    def init_embedding(self, embedding:nn.Embedding, device, init_type:str, **kwargs):
        num_embeddings = embedding.num_embeddings
        embedding_dim  = embedding.embedding_dim
        if init_type == 'lin':
            k = kwargs["k"]
            b = kwargs["b"]
            with torch.no_grad():
                for i in range(num_embeddings):
                    embedding.weight[i] = torch.tensor([[k * i + b for _ in range(embedding_dim)]], device=device)
        else:
            raise ValueError(f"Unsupported init_type = {init_type}")
        
        log.info(f"Initialized embedding.weight")
        if kwargs.get("verbose", False):
            log.info(f"Initialized embedding.weight={embedding.weight}")
        
    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        inputs:
            x: (B, Ta, Da)
            time: (B,) floating point in [0,1) flow matching time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching inference step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        vel, time_emb, cond_emb = self.policy.forward(action, time, cond, output_embedding=True)
        
        # noise head (for exploration). allow gradient flow.
        if step >= self.learn_explore_noise_from:
            if self.learn_explore_time_embedding:
                step_ts = torch.tensor(step, device = self.device).repeat(B)
                time_emb_explore = self.time_embedding_explore(step_ts)
                feature_noise    = torch.cat([time_emb_explore, cond_emb], dim=-1)
            else:
                feature_noise    = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            
            noise_logvar    = self.mlp_logvar(feature_noise)
            noise_std       = self.process_noise(noise_logvar)
        else:
            noise_std       = self.logprob_noise_levels[:, step].repeat(B,1)
        
        if learn_exploration_noise:
            return vel, noise_std
        else:
            return vel, noise_std.detach()
    
    def process_noise(self, noise_logvar):
        '''
        input:
            torch.Tensor([B, Ta , Da])
        output:
            torch.Tensor([B, 1, Ta * Da]), floating point values, bounded in [min_logprob_denoising_std, max_logprob_denoising_std]
        '''
        noise_logvar = noise_logvar
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std
    
    @torch.no_grad()
    def stochastic_interpolate(self,t):
        if self.initial_noise_scheduler_type == 'vp':
            a = 0.2 #2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.initial_noise_scheduler_type == 'lin':
            k=0.1
            b=0.0
            std = k*t+b
        else:
            raise NotImplementedError
        return std
    
    @torch.no_grad()
    def set_logprob_noise_levels(self):
        '''
        create noise std for logrporbability calcualion. 
        generate a tensor `self.logprob_noise_levels` of shape `[1, self.denoising_steps,  self.policy.horizion_steps x self.policy.act_dim]`
        '''
        self.logprob_noise_levels = torch.zeros(self.denoising_steps, device=self.device, requires_grad=False)
        steps = torch.linspace(0, 1, self.denoising_steps, device=self.device)
        for i, t in enumerate(steps):
            self.logprob_noise_levels[i] = self.stochastic_interpolate(t)
        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
        self.logprob_noise_levels = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.policy.horizon_steps *  self.policy.action_dim)