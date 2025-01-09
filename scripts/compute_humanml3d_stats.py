import os
import torch
import numpy as np
from hmr4d.dataset.pure_motion.humanml3d import Humanml3dDataset
from transformers import T5Tokenizer, T5EncoderModel
from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from motiondiff.utils.torch_utils import tensor_to

torch.autograd.set_grad_enabled(False)

dataset = Humanml3dDataset(cam_augmentation="v11")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)
encoder = EnDecoder(stats_name='DEFAULT_01', encode_type='humanml3d').cuda()


count = 0
mean = 0
M2 = 0

for data in dataloader:
    data = tensor_to(data, 'cuda')
    x = encoder.encode(data)
    x = x.reshape(-1, x.shape[-1])
    # Update count, mean, and M2 for the new data
    count += x.shape[0]
    delta = x - mean
    mean += delta.sum(dim=0, keepdim=True) / count
    delta2 = x - mean
    M2 += torch.sum(delta * delta2, dim=0, keepdim=True)
    print(count, len(dataset))

std = torch.sqrt(M2 / count)
std[torch.isnan(std)] = 1e-3
std[std < 1e-3] = 1e-3

# Save the computed statistics
mean = mean[0].cpu().numpy()
std = std[0].cpu().numpy()
np.set_printoptions(precision=4, suppress=True)
np.save(os.path.join('out', 'mean.npy'), mean)
np.save(os.path.join('out', 'std.npy'), std)

# Print the arrays
mean_str = np.array2string(mean, separator=', ')
std_str = np.array2string(std, separator=', ')
print('mean:', mean_str)
print('std:', std_str)
