import os

import torch

# Define the input and output directories
input_dir = "inputs/HumanML3D_SMPL/t5_embeddings_v1"
output_dir = "inputs/HumanML3D_SMPL/t5_embeddings_v1_half"
# input_dir = 'inputs/MotionXpp_ye/t5_embeddings_v1'
# output_dir = 'inputs/MotionXpp_ye/t5_embeddings_v1_half'

fname = "test_text_embed.pth"
os.makedirs(output_dir, exist_ok=True)

# Load the saved dictionary
input_file = os.path.join(input_dir, fname)
# input_file = os.path.join(input_dir, 'test_text_embed.pth')
text_embed_dict = torch.load(input_file)

# Convert the tensors to half precision
text_embed_dict_half = {k: v.half() for k, v in text_embed_dict.items()}

# compute precision difference
for k, v in list(text_embed_dict.items())[:10]:
    diff = torch.abs(v - text_embed_dict_half[k].float()).mean()
    print(f"{k}: {diff}")

# Save the dictionary with half precision tensors
output_file = os.path.join(output_dir, fname)
torch.save(text_embed_dict_half, output_file)

print(f"Converted and saved dictionary to half precision at {output_file}")
