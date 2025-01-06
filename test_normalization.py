
import numpy as np

# Define the path to the .npz file (use the appropriate path format for Linux)
file_path = '/mnt/d/Research/0-Robotics/Grad_Thesis_Diffusion_Robots/1-code/0-dppo-video-mujoco/data/gym/halfcheetah-medium-v2/normalization.npz'

# Load the .npz file
data = np.load(file_path)

# Print the keys (names of arrays) in the .npz file
print("Keys in the .npz file:")
for key in data.keys():
    print(key)
    
# Optionally, print the contents of each array
print("\nContents of the .npz file:")
for key in data.keys():
    print(f"{key}: {data[key]}, {data[key].shape}")
