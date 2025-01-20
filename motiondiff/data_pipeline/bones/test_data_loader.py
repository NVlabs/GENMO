import numpy as np
import os
from tqdm import tqdm
import json
from torch.utils import data
from torch.utils.data import DataLoader
import time
import pdb


data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full341'
# data_dir = '/mnt/nvr_torontoai_humanmotionfm/datasets/bones_full341'

test_data_dir = os.path.join(data_dir, 'new_joint_vecs')
test_manifest_file = os.path.join(data_dir, 'test_data_loader_manifest_v1.json')
test_data_file = os.path.join(data_dir, 'test_data_loader_features_all_v1.npy')
# test_manifest_file = os.path.join('dataset/bones_full341', 'test_data_loader_manifest_v1.json')
# test_data_file = os.path.join('dataset/bones_full341', 'test_data_loader_features_all_v1.npy')


# Create stitched file
def create_stitched_file(num_test_files):
    posed_joints_all = []
    base = 0
    manifest = []
    for file in tqdm(sorted(os.listdir(test_data_dir))[:num_test_files]):
        joints = np.load(os.path.join(test_data_dir, file))
        posed_joints_all += [joints]
        manifest += [{'base': base, 'length': joints.shape[0]}]
    posed_joints_all = np.concatenate(posed_joints_all, axis=0)
    json.dump(manifest, open(test_manifest_file, 'w'))
    np.save(test_data_file, posed_joints_all)


class DistributeDataset(data.Dataset):
    def __init__(self, num_frames, num_test_files):
        self.file_list = sorted(os.listdir(test_data_dir))[:num_test_files]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(test_data_dir, self.file_list[idx]))
        if data.shape[0] >= self.num_frames:
            data = data[:self.num_frames]
        else:
            data = np.concatenate([data, np.zeros((self.num_frames - data.shape[0], data.shape[1]), dtype=np.float32)], axis=0)
        return data

class SingleDataset(data.Dataset):
    def __init__(self, num_frames):
        self.manifest = json.load(open(test_manifest_file, 'r'))
        self.data_all = np.load(test_data_file, mmap_mode='r')
        self.num_frames = num_frames

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        data = self.data_all[item['base']: item['base'] + item['length']]
        if data.shape[0] >= self.num_frames:
            data = data[:self.num_frames].copy()
        else:
            data = np.concatenate([data, np.zeros((self.num_frames - data.shape[0], data.shape[1]), dtype=np.float32)], axis=0) 
        return data

if __name__ == '__main__':

    num_test_files = 350000
    num_frames = 196

    create_stitched_file(num_test_files)

    num_epochs = 100
    batch_size = 512
    for num_workers in [4, 8, 32, 64]:

        dataset = DistributeDataset(num_frames, num_test_files)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )

        start_time = time.time()
        num_steps = 0
        for e in tqdm(range(num_epochs)):
            for batch_idx, batch in enumerate(data_loader):
                num_steps += 1
                pass
        time_dis = time.time() - start_time
        time_dis_per_step = time_dis / num_steps


        dataset = SingleDataset(num_frames)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )

        start_time = time.time()
        num_steps = 0
        for e in tqdm(range(num_epochs)):
            for batch_idx, batch in enumerate(data_loader):
                num_steps += 1
                pass
        time_single = time.time() - start_time
        time_single_per_step = time_single / num_steps

        print(f"Running {num_epochs} epochs with batch_size={batch_size} and num_workers={num_workers} takes {time_dis_per_step:.2f}s per step when loading separately, takes {time_single_per_step:.2f}s per step when loading from single file")