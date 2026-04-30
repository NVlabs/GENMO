import os

import joblib
import torch

# Load the data
data = joblib.load("phc/output/data/noise_False_0.05_2025-02-06-23_00_31.pkl")
out_dir = "inputs/humanoid/data/traj_im_v1/noise_0"
stats_dir = "inputs/humanoid/data/traj_im_v1/stats"
self_obs_dim = None
# data = joblib.load("phc/output/data/noise_True_0.07_2025-02-07-00_05_21.pkl")
# out_dir = "inputs/humanoid/data/traj_v1/noise_0.07"
# stats_dir = "inputs/humanoid/data/traj_v1/stats"
# self_obs_dim = 358

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)
# Get the sequence names and length
key_names = data["key_names"]
seq_length = len(key_names)

# dump running_mean
running_mean = dict(data["running_mean"])
mean = running_mean["running_mean"][:self_obs_dim].float()
std = running_mean["running_var"][:self_obs_dim].sqrt().float()
torch.save(mean, os.path.join(stats_dir, "mean.pth"))
torch.save(std, os.path.join(stats_dir, "std.pth"))

# Split data into individual sequences
for i in range(seq_length):
    # Create a new dictionary for this sequence
    seq_data = {}

    # Copy data for this sequence index from all relevant keys
    for key in data.keys():
        if key != "running_mean":
            if key == "obs":
                seq_data[key] = data[key][i][..., :self_obs_dim]
            else:
                seq_data[key] = data[key][i]
            if key != "key_names":
                seq_data[key] = torch.tensor(seq_data[key])

    # Create filename from sequence name (sanitize if needed)
    filename = f"{key_names[i]}.pth"
    filepath = os.path.join(out_dir, filename)

    # Save as torch file
    torch.save(seq_data, filepath)
    print(f"Saved sequence {i + 1}/{seq_length}: {filename}")
