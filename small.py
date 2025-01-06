import gym
import mujoco_py

# Initialize the environment
env_name = "Hopper-v2"  # Replace with your desired environment
env = gym.make(env_name)

env.reset()

for _ in range(100000):
    env.render()
    env.step(env.action_space.sample())  # Take a random action

env.close()
