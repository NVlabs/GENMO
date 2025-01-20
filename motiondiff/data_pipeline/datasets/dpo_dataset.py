import numpy as np
import torch
from torch.utils import data
from motiondiff.data_pipeline.tensors import collate


# an adapter to our collate func
def dpo_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    win_batch = [{
        'inp': torch.tensor(b[2]).float(), # [seqlen, J] -> [J, 1, seqlen]
        'target': 0,
        'text': b[0], #b[0]['caption']
        'lengths': b[1],
    } for b in batch]
    win_batch = collate(win_batch)
    lose_batch = [{
        'inp': torch.tensor(b[5]).float(), # [seqlen, J] -> [J, 1, seqlen]
        'target': 0,
        'text': b[3], #b[0]['caption']
        'lengths': b[4],
    } for b in batch]
    lose_batch = collate(lose_batch)
    return win_batch[0], win_batch[1], lose_batch[0], lose_batch[1]


class DPODataset(data.Dataset):
    def __init__(self, split, num_frames, win_key='left', lose_key='right', datapath='data/dpo/syn_data_v1'):
        self.win_key = win_key
        self.lose_key = lose_key
        self.win_motion_file = f'{datapath}/{split}_{win_key}_samples.pt'
        self.lose_motion_file = f'{datapath}/{split}_{lose_key}_samples.pt'
        self.win_motions = torch.load(self.win_motion_file).cpu().numpy()
        self.lose_motions = torch.load(self.lose_motion_file).cpu().numpy()
        return

    def __len__(self):
        return len(self.win_motions)

    def __getitem__(self, item):
        win_caption = f'Walking'
        lose_caption = f'Walking'
        win_motion = self.win_motions[item]
        lose_motion = self.lose_motions[item]
        win_m_length = win_motion.shape[-1]
        lose_m_length = lose_motion.shape[-1]
        return win_caption, win_m_length, win_motion, lose_caption, lose_m_length, lose_motion