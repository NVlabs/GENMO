import os
import pickle
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# from motiondiff.utils.mdm_modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from torch.utils.data import DataLoader, Dataset

from motiondiff.utils.motionx_modules import (
    MotionDecoderWithDropout,
    MotionEncoderBiGRUCoWithDropout,
    MovementConvEncoderWithDropout,
)

dataset_name = "motionx"

opt = {
    "dataset_name": dataset_name,
    "device": "cuda",
    "dim_word": 300,
    "max_motion_length": 196,
    "dim_motion_hidden": 1024,
    "max_text_len": 20,
    "dim_text_hidden": 512,
    "dim_coemb_hidden": 512,
    "dim_pose": 143 if dataset_name == "motionx" else 263,
    "dim_movement_enc_hidden": 512,
    "dim_movement_latent": 512,
    "checkpoints_dir": ".",
    "unit_length": 4,
}


# movement_enc = MovementConvEncoder(opt['dim_pose'], opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])

# motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
#                                     hidden_size=opt['dim_motion_hidden'],
#                                     output_size=opt['dim_coemb_hidden'],
#                                     device=opt['device'])

# # Initialize the decoder
# motion_decoder = MotionDecoder(
#     latent_dim=opt['dim_coemb_hidden'],
#     hidden_dim=opt['dim_motion_hidden'],
#     output_dim=opt['dim_pose'],
#     unit_length=opt['unit_length']
# ).to(opt['device'])

movement_enc = MovementConvEncoderWithDropout(
    input_dim=opt["dim_pose"],  # Use the full input dimension (143)
    hidden_dim=opt["dim_movement_enc_hidden"],
    latent_dim=opt["dim_movement_latent"],
    dropout_prob=0.3,  # Set dropout probability
).to(opt["device"])

motion_enc = MotionEncoderBiGRUCoWithDropout(
    input_size=opt["dim_movement_latent"],
    hidden_size=opt["dim_motion_hidden"],
    output_size=opt["dim_coemb_hidden"],
    device=opt["device"],
    dropout_prob=0.3,
).to(opt["device"])

motion_decoder = MotionDecoderWithDropout(
    latent_dim=opt["dim_coemb_hidden"],
    hidden_dim=opt["dim_motion_hidden"],
    output_dim=opt["dim_pose"],
    unit_length=opt["unit_length"],
    dropout_prob=0.3,
).to(opt["device"])

movement_enc = movement_enc.to(opt["device"])
motion_enc = motion_enc.to(opt["device"])
motion_decoder = motion_decoder.to(opt["device"])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(movement_enc.parameters())
    + list(motion_enc.parameters())
    + list(motion_decoder.parameters()),
    lr=1e-4,
)


gt_features = pickle.load(open("inputs/motionx_gt_feats.pkl", "rb"))
gt_motions = gt_features["motions"]  # shape: (3837, 196, 143)
gt_texts = gt_features["texts"]


# Define a Dataset class
class MotionDataset(Dataset):
    def __init__(self, motions):
        self.motions = torch.tensor(motions).float().cuda()  # Preload and move to GPU

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx]


batch_size = 64
# Create Dataset and DataLoader
motion_dataset = MotionDataset(gt_motions)
data_loader = DataLoader(motion_dataset, batch_size=batch_size, shuffle=True)

total_num = gt_motions.shape[0]
batch_num = total_num // batch_size

# Training loop
for epoch in range(30):  # Number of epochs
    total_loss = 0.0
    for batch_idx, data in enumerate(data_loader):
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass through encoder
        movements = movement_enc(data)
        m_lens = (
            torch.tensor([data.shape[1]])
            .float()
            .to(opt["device"])
            .repeat(data.shape[0])
        )
        m_lens = (
            (m_lens // opt["unit_length"]).cpu().long()
        )  # Move to CPU and convert to int64
        motion_embedding = motion_enc(movements, m_lens)
        # Forward pass through decoder
        reconstructed_motion = motion_decoder(motion_embedding, m_lens)

        # Compute loss
        loss = criterion(reconstructed_motion, data)
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    # Log progress
    print(f"Epoch {epoch + 1}/{30}, Loss: {total_loss / batch_num:.6f}")

# Save the model weights
save_path = os.path.join("outputs", "motionx_motion_encoder_decoder_weights.pth")
torch.save(
    {
        "movement_enc_state_dict": movement_enc.state_dict(),
        "motion_enc_state_dict": motion_enc.state_dict(),
        "motion_decoder_state_dict": motion_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    save_path,
)

print(f"Model weights saved to {save_path}")

# for batch_idx in range(batch_num):
#     data = gt_motions[batch_idx * batch_size: (batch_idx + 1) * batch_size]
#     movements = movement_enc(data)
#     m_lens = torch.tensor([data.shape[1]]).float().cuda().repeat(data.shape[0])
#     m_lens = m_lens // opt['unit_length']
#     motion_embedding = motion_enc(movements, m_lens)

#     # have got the latent code, then write the decoder
