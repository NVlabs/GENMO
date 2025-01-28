import math
import random

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hmr4d.network.mdm.mdm_denoiser_rope import PositionalEncoding, TimestepEmbedder


class RfModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_x = nn.Linear(3, 128)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
        sequence_time_encoder = PositionalEncoding(128, dropout=0.1)
        self.embed_timestep = TimestepEmbedder(128, sequence_time_encoder)

    def forward(self, x, cond, t):
        emb = self.embed_timestep(t.long()).permute(1, 0, 2).squeeze(1)
        x = torch.cat([x, cond], dim=-1)
        x = self.enc_x(x)
        x = torch.cat([x, emb], dim=-1)
        return self.mlp(x)


class RfDataset(Dataset):
    def __init__(self, length=10000, cond=None, z1=None):
        self.length = length
        self.cond = cond
        self.z1 = z1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        z0 = torch.randn(2)
        # if self.train:
        #     cond = torch.rand(2)
        # else:
        #     cond = torch.rand(2)
        # cond = torch.randn(1)
        cond = self.cond[idx].clone()
        z1 = self.z1[idx].clone()
        z0[1] = z1[1]

        # if cond > 0 and z0 > 0:
        #     z1[0] = z1[0] + 5
        #     z1[1] = z1[1] + 5
        # elif z0[0] < 0 and z0[1] < 0:
        #     z1[0] = z1[0] - 5
        #     z1[1] = z1[1] - 5
        # elif z0[0] > 0 and z0[1] < 0:
        #     z1[0] = z1[0] + 5
        #     z1[1] = z1[1] - 5
        # else:
        #     z1[0] = z1[0] - 5
        #     z1[1] = z1[1] + 5

        return {"z0": z0, "cond": cond, "z1": z1}


if __name__ == "__main__":
    # set seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    train_cond = torch.randn(10000, 1)
    z1 = torch.randn(10000, 2)
    z1[:, :1] = train_cond * 5

    train_cond = torch.load("rf_base_train_cond.pth")
    z1 = torch.load("rf_base_train_z1.pth")

    model = RfModel()
    dataset = RfDataset(10000, cond=train_cond, z1=z1)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda"
    num_epochs = 100
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    sampling_eps = 1e-3
    T = 1

    model.to(device)

    # Training
    epoch = 0
    pbar = tqdm(
        total=num_epochs * len(dataset),
        desc=f"Epoch {epoch}/{num_epochs}",
        dynamic_ncols=True,
    )
    for epoch in range(num_epochs):
        for batch in dataloader:
            gt_z0, cond, z1 = (
                batch["z0"].to(device),
                batch["cond"].to(device),
                batch["z1"].to(device),
            )
            # z0 = torch.randn_like(gt_z0).to(device)
            z0 = torch.zeros_like(gt_z0).to(device)
            t = (
                torch.rand(z0.shape[0], device=z0.device) * T
            )  # * (T - sampling_eps) + sampling_eps
            t = torch.zeros(z0.shape[0], device=z0.device)
            t_expand = t.view(-1, 1).repeat(1, z0.shape[1])

            zt = t_expand * z1 + (1 - t_expand) * z0
            target = z1 - z0
            pred = model(zt, cond, t * 1000)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item()}")
            pbar.update(z1.shape[0])

    # Sampling
    model.eval()
    with torch.no_grad():
        z0 = torch.zeros(1000, 2).to(device)

        x = z0.clone()
        cond1 = torch.ones(z0.shape[0] // 2, 1).to(device)
        cond2 = torch.ones(z0.shape[0] // 2, 1).to(device) * -1

        cond = torch.cat([cond1, cond2], dim=0)
        z1 = model(
            x,
            cond,
            torch.ones(x.shape[0], device=x.device) * (T - sampling_eps) + sampling_eps,
        )

        z1 = z1.cpu().numpy()
        z0 = z0.cpu().numpy()

    with torch.no_grad():
        noise = torch.zeros(100, 2).to(device)

        x = noise.clone()
        # cond1 = torch.ones(noise.shape[0] // 2, 1).to(device)
        # cond2 = torch.ones(noise.shape[0] // 2, 1).to(device) * -1

        # cond = torch.cat([cond1, cond2], dim=0)
        cond = (
            torch.arange(-1, 1, 1.0 / (noise.shape[0] // 2)).to(device).reshape(-1, 1)
        )
        z1 = model(x, cond, torch.zeros(x.shape[0], device=x.device) * 1000)

        pred = z1.cpu().numpy()
        err = ((pred[:, :1] - 5 * cond.cpu().numpy()) ** 2).mean() * 1000
        print(f"Error Regression: {err}")
