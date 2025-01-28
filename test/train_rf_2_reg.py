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
    def __init__(self, z1, z0, cond):
        self.z1 = z1.copy()
        self.z0 = z0.copy()
        self.cond = cond.copy()

    def __len__(self):
        return self.z1.shape[0]

    def __getitem__(self, idx):
        return {"z0": self.z0[idx], "z1": self.z1[idx], "cond": self.cond[idx]}


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model = RfModel()

    rf1_mode = "z1"
    # rf1_mode = 'drift'
    # pred_mode = "z1"  # 'drift'
    pred_mode = "z1_abs"
    # pred_mode = "drift"
    device = "cuda"
    num_epochs = 100
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    sampling_eps = 0.0
    T = 1

    model.load_state_dict(torch.load("rf1.pth", map_location="cpu"))
    model.to(device)

    # generate data

    print("Generating data...")
    sample_N = 100
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(10000, 2).to(device)
        dt = (1.0 - sampling_eps) / sample_N

        x = z0.clone()
        cond = torch.load("rf_base_train_cond.pth").to(device)
        # z1 = torch.load('rf_base_train_z1.pth').to(device)
        for i in range(sample_N):
            num_t = i / sample_N * (T - sampling_eps) + sampling_eps
            vec_t = torch.ones(x.shape[0], device=x.device) * num_t
            if rf1_mode == "z1":
                # pred_z1 = model(x, cond, vec_t*1000)
                # drift = (pred_z1 - x) / (1 - vec_t[:, None])
                drift = model(x, cond, vec_t * 1000)
            elif rf1_mode == "z1_abs":
                pred_z1 = model(x, cond, vec_t * 1000)
                drift = (pred_z1 - x) / (1 - vec_t[:, None])
            else:
                drift = model(x, cond, vec_t * 1000)

            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
            sigma_t = 0
            noise_scale = 1
            pred_sigma = drift + (sigma_t**2) / (
                2 * (noise_scale**2) * ((1.0 - num_t) ** 2)
            ) * (
                0.5 * num_t * (1.0 - num_t) * drift
                - 0.5 * (2.0 - num_t) * x.detach().clone()
            )

            x = (
                x.detach().clone() + drift * dt
            )  # + sigma_t * math.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
        z1 = x.cpu().numpy()
        z0 = z0.cpu().numpy()
        cond = cond.cpu().numpy()
        z1[:, :1] = cond * 5

    dataset = RfDataset(z1, z0, cond)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Training
    epoch = 0
    pbar = tqdm(
        total=num_epochs * len(dataset),
        desc=f"Epoch {epoch}/{num_epochs}",
        dynamic_ncols=True,
    )
    for epoch in range(num_epochs):
        for batch in dataloader:
            z0, z1, cond = (
                batch["z0"].to(device),
                batch["z1"].to(device),
                batch["cond"].to(device),
            )
            t = (
                torch.rand(z0.shape[0], device=z0.device) * (T - sampling_eps)
                + sampling_eps
            )
            t_expand = t.view(-1, 1).repeat(1, z0.shape[1])

            zt = t_expand * z1 + (1 - t_expand) * z0
            target = z1 - z0
            if pred_mode == "drift":
                pred_drift = model(zt, cond, t * 1000)
                # pred_z1 = zt + pred_drift * (1 - t_expand)
                loss = ((pred_drift - target) ** 2).mean()
            elif pred_mode == "z1":
                pred_drift = model(zt, cond, t * 1000)
                pred_z1 = zt + pred_drift * (1 - t_expand)
                loss = ((pred_z1 - z1) ** 2).mean()
            else:
                pred_z1 = model(zt, cond, t * 1000)
                # pred_drift = (pred_z1 - zt) / (1 - t_expand)

                # loss = (((pred_drift - target) * (1 - t_expand)) ** 2).mean()
                loss = ((pred_z1 - z1) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item()}")
            pbar.update(z1.shape[0])

    # Sampling
    sample_N = 1
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(1000, 2).to(device)
        dt = (1.0 - sampling_eps) / sample_N

        x = z0.clone()
        cond1 = torch.ones(z0.shape[0] // 2, 1).to(device)
        cond2 = torch.ones(z0.shape[0] // 2, 1).to(device) * -1

        cond = torch.cat([cond1, cond2], dim=0)
        # cond = torch.load('rf1_cond.pth').to(device)
        intermediate_z = [x.cpu().numpy()]
        for i in range(sample_N):
            num_t = i / sample_N * (T - sampling_eps) + sampling_eps
            vec_t = torch.ones(x.shape[0], device=x.device) * num_t
            if pred_mode == "drift":
                drift = model(x, cond, vec_t * 1000)
            elif pred_mode == "z1":
                drift = model(x, cond, vec_t * 1000)
            else:
                pred_z1 = model(x, cond, vec_t * 1000)
                drift = (pred_z1 - x) / (1 - vec_t[:, None])

            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
            sigma_t = 0
            noise_scale = 1
            pred_sigma = drift + (sigma_t**2) / (
                2 * (noise_scale**2) * ((1.0 - num_t) ** 2)
            ) * (
                0.5 * num_t * (1.0 - num_t) * drift
                - 0.5 * (2.0 - num_t) * x.detach().clone()
            )

            x = (
                x.detach().clone() + drift * dt
            )  # + sigma_t * math.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
            intermediate_z.append(x.cpu().numpy())
        z1 = x.cpu().numpy()
        z0 = z0.cpu().numpy()
        intermediate_z = np.stack(intermediate_z).transpose(1, 0, 2)

    sample_N = 1
    with torch.no_grad():
        noise = torch.zeros(100, 2).to(device)
        dt = (1.0 - sampling_eps) / sample_N

        x = noise.clone()
        # cond1 = torch.ones(noise.shape[0] // 2, 1).to(device)
        # cond2 = torch.ones(noise.shape[0] // 2, 1).to(device) * -1

        # cond = torch.cat([cond1, cond2], dim=0)
        cond = (
            torch.arange(-1, 1, 1.0 / (noise.shape[0] // 2)).to(device).reshape(-1, 1)
        )
        for i in range(sample_N):
            num_t = i / sample_N * (T - sampling_eps) + sampling_eps
            vec_t = torch.ones(x.shape[0], device=x.device) * num_t
            if pred_mode == "drift":
                drift = model(x, cond, vec_t * 1000)
            else:
                pred_z1 = model(x, cond, vec_t * 1000)
                drift = (pred_z1 - x) / (1 - vec_t[:, None])

            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
            sigma_t = 0
            noise_scale = 1
            pred_sigma = drift + (sigma_t**2) / (
                2 * (noise_scale**2) * ((1.0 - num_t) ** 2)
            ) * (
                0.5 * num_t * (1.0 - num_t) * drift
                - 0.5 * (2.0 - num_t) * x.detach().clone()
            )

            x = (
                x.detach().clone() + drift * dt
            )  # + sigma_t * math.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
        pred = x.cpu().numpy()
        err = ((pred[:, :1] - 5 * cond.cpu().numpy()) ** 2).mean() * 1000
        print(f"Error RF2: {err}")

    # Visualization
    import matplotlib.pyplot as plt

    for bid in range(0, len(intermediate_z), 10):
        plt.plot(
            intermediate_z[bid][:, 0],
            intermediate_z[bid][:, 1],
            color="black",
            alpha=0.5,
        )

    plt.scatter(
        z0[z0.shape[0] // 2 :, 0][::10],
        z0[z0.shape[0] // 2 :, 1][::10],
        label="cond2_z0",
        color="red",
    )
    plt.scatter(
        z0[: z0.shape[0] // 2, 0][::10],
        z0[: z0.shape[0] // 2, 1][::10],
        label="cond1_z0",
        color="blue",
    )
    plt.scatter(
        z1[z1.shape[0] // 2 :, 0][::10],
        z1[z1.shape[0] // 2 :, 1][::10],
        label="cond2_z1",
        color="red",
    )
    plt.scatter(
        z1[: z1.shape[0] // 2, 0][::10],
        z1[: z1.shape[0] // 2, 1][::10],
        label="cond1_z1",
        color="blue",
    )
    plt.legend()
    plt.savefig("rf2_reg.png")
    torch.save(model.state_dict(), "rf2_reg.pth")
