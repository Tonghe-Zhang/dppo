# from huazhe's class
import torch
import numpy as np
from util.reward_scaling import RunningRewardScaler

# state-only replay buffer for high dimensional states, actions
class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device, seed):
        self.device = device
        self.state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.action = torch.zeros(capacity, action_size, dtype=torch.float).contiguous().pin_memory()
        self.reward = torch.zeros(capacity, dtype=torch.float).contiguous().pin_memory()
        self.next_state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.done = torch.zeros(capacity, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.state[self.idx] = torch.as_tensor(state)
        self.action[self.idx] = torch.as_tensor(action)
        self.reward[self.idx] = torch.as_tensor(reward)
        self.next_state[self.idx] = torch.as_tensor(next_state)
        self.done[self.idx] = torch.as_tensor(done)

        # update counters
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            self.reward[sample_idxs].to(self.device),
            self.next_state[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device)
        )
        return batch
    
class PPOReplayBuffer:
    def __init__(self, capacity, num_envs, state_size, action_size, device, seed, gamma, gae_lambda, 
                 state_horizon=1, action_horizon=1, reward_scale_running=False):
        """
        capacity: how many states for EACH environment. Total number of states should be `capacity x num_envs`
        """
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        
        self.state = torch.zeros(capacity, num_envs, state_horizon, state_size, dtype=torch.float).contiguous()
        
        self.action = torch.zeros(capacity, num_envs, action_horizon, action_size, dtype=torch.float).contiguous()
        
        self.next_state = torch.zeros(capacity, num_envs, state_horizon, state_size, dtype=torch.float).contiguous()
        
        self.reward = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous()
        
        self.terminated = torch.zeros(capacity, num_envs, dtype=torch.int).contiguous()
        
        self.firsts_trajs = torch.zeros(capacity+1, num_envs, dtype=torch.int).contiguous() # in case we manually restart amidst an on-going episodes, add a flag to record done or manual restart.
        
        # self.done = torch.zeros(capacity, num_envs, dtype=torch.int).contiguous() # terminate or truncated
        self.done = torch.zeros(1, num_envs, dtype=torch.int).contiguous() # terminate or truncated
        
        
        self.value = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous()
        
        self.log_prob = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous()
        
        self.advantage = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous()
        
        self.returns = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous()
        

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_envs = num_envs

        
        self.reward_scale_running = reward_scale_running # whether to apply running average to rewards before calculating advantage
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(num_envs)
        
        self.to_device()
    
    def to_device(self):
        self.state = self.state.to(self.device)
        self.action = self.action.to(self.device)
        self.next_state = self.next_state.to(self.device)
        
        self.reward = self.reward.to(self.device)
        
        self.done = self.done.to(self.device)
        self.terminated = self.terminated.to(self.device)
        
        self.value = self.value.to(self.device)
        self.returns = self.returns.to(self.device)
        self.advantage = self.advantage.to(self.device)
        self.log_prob = self.log_prob.to(self.device)
        
        self.firsts_trajs  = self.firsts_trajs.to(self.device)
        
    
    def __repr__(self) -> str:
        return 'PPOReplayBuffer'
    
    def add(self, transition):
        state, action, next_state, reward, done, terminated, value, logprob = transition
        
        # store transition in the buffer
        self.state[self.idx] = torch.as_tensor(state)
        self.action[self.idx] = torch.as_tensor(action)
        self.reward[self.idx] = torch.as_tensor(reward)
        self.next_state[self.idx] = torch.as_tensor(next_state)
        
        self.terminated[self.idx] = torch.as_tensor(terminated)
        
        # self.done[self.idx] = torch.as_tensor(done)
        self.done = torch.as_tensor(done)
        
        self.firsts_trajs[self.idx+1]  = torch.as_tensor(done)
        
        self.value[self.idx] = torch.as_tensor(value)
        self.log_prob[self.idx] = torch.as_tensor(logprob)
        
        # update counters
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        
        
    
    def clear(self):
        self.idx = 0
        self.size = 0
    
    def make_dataset(self):
        '''
        return the flattened dataset. 
        this function makes self.state: [capacity, num_envs, obs_horizion, obs_dim] 
        to a tensor of shape 
            flattend_state:  [capacity x num_envs, obs_horizion, obs_dim]
        and the other tensors are processed in the similar way. 
        this makes the returned batch contain all the independent samples, each of shape (obs_horizon, obs_dim) etc.
        we will then randomly pick states and actions from it. 
        '''
        batch = (
            self.state[:self.size].flatten(0, 1),
            self.action[:self.size].flatten(0, 1),
            self.log_prob[:self.size].flatten(0, 1),
            self.value[:self.size].flatten(0, 1),
            self.advantage[:self.size].flatten(0, 1),
            self.returns[:self.size].flatten(0, 1)
        )
        return batch

    def get_next_values(self, agent, t) -> torch.Tensor:
        """
        Given timestep t and the current agent, obtain or calculate values of t + 1
        If t is the last timestep or is done, return the value of the next state from the agent
        Otherwise, you can directly return the value of the next state from the buffer
       
        We assume that the buffer is full, and vector envs are used.
        return is of size torch.zeros(capacity, num_envs, dtype=torch.int)
        """
        if t == self.capacity - 1:
            next_values = agent.get_value(self.next_state[t]).to(self.next_state.device)
        else:
            next_values  = self.value[t+1]
        return next_values
        # next_values = torch.where(
        #     self.done[t].bool(), 
        #     agent.get_value(self.next_state[t]).to(self.next_state.device),
        #     self.value[t + 1]
        # )
    
    
    def compute_advantages_and_returns(self, agent) -> None:
        """
        Once the buffer is full, calculate all the advantages and returns for each timestep
        returns shape: [capacitity, number of envs]=[1024,16] in our case.
        """
        last_gae_lam = 0
        for t in reversed(range(self.capacity)):
            next_values = self.get_next_values(agent, t)
            # print(f"next_values at t={t}: {next_values.shape}")
            
            delta = self.reward[t] + self.gamma * next_values * (1 - self.terminated[t]) - self.value[t]
            
            # print(f"delta at t={t}: {delta.shape}")
            
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * (1 - self.terminated[t]) # changed from 1-done[t]
            
            # print(f"last_gae_lam at t={t}: {last_gae_lam.shape}")
            
            # Assign and print the advantage
            self.advantage[t] = last_gae_lam
            # print(f"self.advantage[{t}]: {self.advantage[t].shape}")

        # compute returns for the whole buffers
        self.returns = self.advantage + self.value
    
    def normalize_reward(self): 
        # normalize reward with running variance if specified
        # this could cause constant moving between cpu and gpu, should optimize. 
        if self.reward_scale_running:
            reward_trajs_transpose = self.running_reward_scaler(
                reward=self.reward.cpu().numpy().T, 
                first=self.firsts_trajs[:-1].cpu().numpy().T
            )
        self.reward = torch.tensor(reward_trajs_transpose.T, dtype=torch.float32).to(self.device)