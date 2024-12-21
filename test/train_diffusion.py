import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from hmr4d.network.mdm.mdm_denoiser_rope import PositionalEncoding, TimestepEmbedder
from motiondiff.diffusion import gaussian_diffusion as gd
from motiondiff.diffusion.respace import SpacedDiffusion, space_timesteps
from motiondiff.diffusion.resample import create_named_schedule_sampler

import random
from tqdm import tqdm
import math


class MLPModel(nn.Module):
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
    
    def forward(self, x, t, cond):
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

        return {'z0': z0, 'cond': cond, 'z1': z1}

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def create_gaussian_diffusion(training):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.0  # no scaling
    train_timestep_respacing = ""
    test_timestep_respacing = "100"
    noise_schedule = "cosine"
    sigma_small = True
    timestep_respacing = (
        train_timestep_respacing if training else test_timestep_respacing
    )  # ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


if __name__ == "__main__":
    # set seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    pred_mode = "z1"  # 'drift'
    pred_mode = 'z1_abs'
    pred_mode = "drift"
    schedule_sampler_type = "uniform"

    train_cond = torch.randn(10000, 1)
    z1 = torch.randn(10000, 2)
    z1[:, :1] = train_cond * 5  # the first dimension is the condition * 5
    torch.save(train_cond, 'rf_base_train_cond.pth')
    torch.save(z1, 'rf_base_train_z1.pth')

    model = MLPModel()
    diffusion = create_gaussian_diffusion(training=True)
    schedule_sampler = create_named_schedule_sampler(
        schedule_sampler_type, diffusion
    )
    test_diffusion = create_gaussian_diffusion(training=False)
    dataset = RfDataset(10000, cond=train_cond, z1=z1)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = 'cuda'
    num_epochs = 100
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    sampling_eps = 1e-3
    T = 1

    model.to(device)

    # Training
    epoch = 0
    pbar = tqdm(total=num_epochs * len(dataset), desc=f"Epoch {epoch}/{num_epochs}", dynamic_ncols=True)
    for epoch in range(num_epochs):
        for batch in dataloader:
            gt_z0, cond, z1 = batch['z0'].to(device), batch['cond'].to(device), batch['z1'].to(device)
            z0 = torch.randn_like(gt_z0).to(device)
            t = (torch.rand(z0.shape[0], device=z0.device) * 1000).long()
            t_expand = t.view(-1, 1).repeat(1, z0.shape[1])

            zt = diffusion.q_sample(z1.clone(), t, noise=z0).to(device)

            pred_z1 = model(zt, t, cond)
            loss = ((pred_z1 - z1) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item()}")
            pbar.update(z1.shape[0])
    
    # Sampling
    # sample_N = 100
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(1000, 2).to(device)
        indices = list(range(test_diffusion.num_timesteps - 1))[::-1]

        x = z0.clone()
        cond1 = torch.ones(z0.shape[0] // 2, 1).to(device)
        cond2 = torch.ones(z0.shape[0] // 2, 1).to(device) * -1
        cond = torch.cat([cond1, cond2], dim=0)
        intermediate_z = [x.cpu().numpy()]
        for k, i in enumerate(indices):

            t = torch.tensor([i] * z0.shape[0], device=device)

            alpha_bar = _extract_into_tensor(test_diffusion.alphas_cumprod, t, z0.shape)
            alpha_bar_prev = _extract_into_tensor(test_diffusion.alphas_cumprod_prev, t, z0.shape)

            out_orig = test_diffusion.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={'cond': cond.clone()},
                model_output=None
            )
            eps = test_diffusion._predict_eps_from_xstart(x, t, out_orig["pred_xstart"])
            eta = 0.0
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = torch.randn_like(x)
            mean_pred = (
                out_orig["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            x = mean_pred + nonzero_mask * sigma * noise
            intermediate_z.append(x.cpu().numpy())

        z0 = z0.cpu().numpy()
        z1 = x.cpu().numpy()
        intermediate_z = np.stack(intermediate_z).transpose(1, 0, 2)

    with torch.no_grad():
        noise = torch.zeros(100, 2).to(device)

        x = noise.clone()
        # cond1 = torch.ones(noise.shape[0] // 2, 1).to(device)
        # cond2 = torch.ones(noise.shape[0] // 2, 1).to(device) * -1

        # cond = torch.cat([cond1, cond2], dim=0)
        cond = torch.arange(-1, 1, 1.0 / (noise.shape[0] // 2)).to(device).reshape(-1, 1)
        indices = list(range(test_diffusion.num_timesteps - 1))[::-1]

        for k, i in enumerate(indices):
            vec_t = torch.tensor([i] * noise.shape[0], device=device)

            alpha_bar = _extract_into_tensor(test_diffusion.alphas_cumprod, vec_t, noise.shape)
            alpha_bar_prev = _extract_into_tensor(test_diffusion.alphas_cumprod_prev, vec_t, noise.shape)

            out_orig = test_diffusion.p_mean_variance(
                model,
                x,
                vec_t,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={'cond': cond},
                model_output=None
            )
            eps = test_diffusion._predict_eps_from_xstart(x, vec_t, out_orig["pred_xstart"])
            eta = 0.0
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = torch.randn_like(x)
            mean_pred = (
                out_orig["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (vec_t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            x = mean_pred + nonzero_mask * sigma * noise

        pred = x.cpu().numpy()
        err = ((pred[:, :1] -  5 * cond.cpu().numpy()) ** 2).mean() * 1000
        print(f'Error DM: {err}')

    # Visualization
    import matplotlib.pyplot as plt
    for bid in range(0, len(intermediate_z), 10):
        plt.plot(intermediate_z[bid][:, 0], intermediate_z[bid][:, 1], color='black', alpha=0.5)

    plt.scatter(z0[z0.shape[0] // 2:, 0][::10], z0[z0.shape[0] // 2:, 1][::10], label='cond2_z0', color='red')
    plt.scatter(z0[:z0.shape[0] // 2, 0][::10], z0[:z0.shape[0] // 2, 1][::10], label='cond1_z0', color='blue')
    plt.scatter(z1[z1.shape[0] // 2:, 0][::10], z1[z1.shape[0] // 2:, 1][::10], label='cond2_z1', color='red')
    plt.scatter(z1[:z1.shape[0] // 2, 0][::10], z1[:z1.shape[0] // 2, 1][::10], label='cond1_z1', color='blue')
    plt.legend()
    plt.savefig('dm.png')
    torch.save(model.state_dict(), 'dm.pth')
    torch.save(cond, 'dm_cond.pth')
