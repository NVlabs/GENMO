import sys 
import os 
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle 
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hmr4d.dataset.motionx.motionx import MotionXDataset
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from motiondiff.utils.torch_utils import tensor_to

encoder = EnDecoder(stats_name='DEFAULT_01', encode_type='humanml3d').cuda()

test_dataset = MotionXDataset(split='aligned', version='vlocal', max_motion_frames=196, motion_start_mode='fixed')
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

texts = []
motions = []

for data in tqdm(dataloader):
    data = tensor_to(data, 'cuda')
    x = encoder.encode_humanml3d(data)
    motions.append(x)
    texts.append(data['caption'])
  
motions = torch.cat(motions, dim=0).cpu().numpy()

result_dict = {'motions': motions, 'texts': texts}

with open('inputs/motionx_all_gt_feats.pkl', 'wb') as f:
    pickle.dump(result_dict, f)

breakpoint()


