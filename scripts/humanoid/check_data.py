import joblib
import torch

data = joblib.load("phc/output/data/noise_False_0.05_2025-02-06-23_00_31.pkl")

print(data.keys())

for key in data.keys():
    print(key)
    if key in ["obs", "humanoid_clean_action", "env_action"]:
        print(data[key][0].shape)
    elif key != "running_mean":
        print(data[key][:20])

split = "train"
fname = f"inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_{split}.pth"
motion_data = torch.load(fname, weights_only=False)

max_len = 0
for key, data in motion_data.items():
    if data["pose"].shape[0] > 300:
        print(key)
    else:
        print(data["pose"].shape, key)
    max_len = max(max_len, data["pose"].shape[0])

print(max_len)
