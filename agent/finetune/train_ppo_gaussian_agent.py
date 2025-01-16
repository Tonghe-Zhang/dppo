"""
PPO training for Gaussian/GMM policy.
"""
import torch
import logging
log = logging.getLogger(__name__)


from agent.finetune.train_ppo_agent import TrainPPOAgent
from agent.finetune.buffer import PPOReplayBuffer

class TrainPPOGaussianAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.buffer = PPOReplayBuffer(n_steps=self.n_steps, 
                                      n_envs=self.n_envs,
                                      horizon_steps=self.horizon_steps, 
                                      act_steps= self.act_steps,
                                      action_dim=self.action_dim,
                                      n_cond_step=self.n_cond_step, 
                                      obs_dim=self.obs_dim, 
                                      save_full_observation=self.save_full_observations,
                                      furniture_sparse_reward = self.furniture_sparse_reward,
                                      best_reward_threshold_for_success = self.best_reward_threshold_for_success,
                                      reward_scale_running = self.reward_scale_running,
                                      gamma = self.gamma,
                                      gae_lambda=self.gae_lambda,
                                      reward_scale_const = self.reward_scale_const,
                                      device=self.device)
    
    def get_log_probs(self, cond, action):
        return self.model.get_logprobs(cond, action)[0].cpu().numpy()
     
    def update_step(self, batch):
        '''
        batch: 
        {"state": obs[minibatch_idx]},
        samples[minibatch_idx]          : torch.Tensor(minibatch_size, n_act_steps, act_dim)
        returns[minibatch_idx],
        values[minibatch_idx],
        advantages[minibatch_idx],
        logprobs[minibatch_idx]]
        '''
        
        pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.model.loss(*batch, use_bc_loss=self.use_bc_loss)
        
        return pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std
     
    
    def run(self):
        
        self.prepare_run()
        
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.buffer.reset()
            self.reset_env()
            
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(self.prev_obs_venv["state"]).float().to(self.device)
                    }
                    samples = self.model.forward(
                        cond=cond,
                        deterministic=self.eval_mode,
                    )
                    output_venv = samples.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], output_venv, reward_venv,terminated_venv,done_venv)
                
                # update for next step
                self.prev_obs_venv = obs_venv

                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0 #not acounting for done within action chunk

            self.buffer.summarize_episode_reward()

            # Update
            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic, self.get_log_probs)
                self.agent_update()
            
            self.plot_state_trajecories() #(only in D3IL)

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
            self.save_model()
            
            self.log()
            
            self.itr += 1