import numpy as np
import cv2
import gym
import os

# Force CPU rendering
os.environ['MUJOCO_GL'] = 'osmesa'

# Create the Hopper-v3 environment
env = gym.make('Hopper-v3')

# Reset the environment to get the initial observation
observation = env.reset()

# Simulation loop
while True:
    # Sample a random action (you can implement your control policy here)
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, done, info = env.step(action)

    # Render to the screen
    env.render(mode='human')  # Use 'human' mode for on-screen rendering

    if done:
        observation = env.reset()  # Reset environment if done

# Clean up
env.close()
