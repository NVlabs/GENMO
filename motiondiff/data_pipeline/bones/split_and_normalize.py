import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from motiondiff.data_pipeline.datasets.bones_dataset import BonesDataset

seed = 7
split_version = 'v5'
stats_version = 'v5_local_height'
out_split_folder = f'assets/bones/splits/{split_version}'
out_stats_folder = f'assets/bones/stats/{stats_version}'
skip_index = np.load('assets/bones/skip_idx/v14/skip_index_geq_2artifacts.npy')
split_ratio = 0.9
split_by_content = True
motion_feature_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full353_v3.0/new_joint_vecs'
meta_file = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full353_v3.0/meta_240619_v014_002.csv'
# motion_feature_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full353_v1.1/new_joint_vecs'
# meta_file = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full353_v1.1/meta_240416_v3.csv'

np.random.seed(seed)

os.makedirs(out_split_folder, exist_ok=True)
os.makedirs(out_stats_folder, exist_ok=True)

bone_dataset_len = 350000

""" Split """
train_index_file = os.path.join(out_split_folder, 'train_index.npy')
test_index_file = os.path.join(out_split_folder, 'test_index.npy')
if os.path.exists(train_index_file) and os.path.exists(test_index_file):
    print(f'loaded index files from {out_split_folder}')
    train_index = np.load(train_index_file)
    test_index = np.load(test_index_file)
else:
    if split_by_content:
        # meta_file = 'out/meta1.csv'
        meta = pd.read_csv(meta_file)
        meta = meta.drop(skip_index)
        # Separate NaN content
        nan_content_df = meta[meta['content'].isna()]
        non_nan_content_df = meta[meta['content'].isna() == False]
        # Group by content
        grouped = non_nan_content_df.groupby('content')
        groups = [group for _, group in grouped]
        np.random.shuffle(groups)
        cumulative_count = np.cumsum([len(group) for group in groups])
        total_rows = non_nan_content_df.shape[0]
        split_row_index = np.where(cumulative_count >= total_rows * split_ratio)[0][0]
        train_groups = groups[:split_row_index + 1]
        test_groups = groups[split_row_index + 1:]
        # Concatenate groups back to DataFrames
        train_df = pd.concat(train_groups + [nan_content_df])
        test_df = pd.concat(test_groups)
        train_index = np.asarray(train_df.index)
        test_index = np.asarray(test_df.index)
    else:
        all_index = np.arange(bone_dataset_len)
        if skip_index is not None:
            all_index = np.setdiff1d(all_index, skip_index)
        total_len = len(all_index)
        np.random.shuffle(all_index)
        train_index = all_index[:int(total_len * split_ratio)]
        test_index = all_index[int(total_len * split_ratio):]
    train_index = np.sort(train_index)
    test_index = np.sort(test_index)
    np.save(os.path.join(out_split_folder, 'train_index.npy'), train_index)
    np.save(os.path.join(out_split_folder, 'test_index.npy'), test_index)
    print(f'index files saved to {out_split_folder}')
    
""" Normalize """    
dataset = BonesDataset('train', num_frames=-1, split_file_pattern=f'{out_split_folder}/%s_index.npy', meta_file=meta_file, stats_folder=out_stats_folder, motion_feature_dir=motion_feature_dir, normalize_motion=False)
# for i in range(3):
#     text, motion, m_length = dataset[i]
#     print(motion.shape, text)

# Initialize variables to compute the mean and std incrementally
count = 0
mean = 0
M2 = 0
# Loop over the dataset and update the mean and std incrementally
for i in tqdm(range(len(dataset))):
    text, motion, m_length = dataset[i]
    # Update count, mean, and M2 for the new data
    count += motion.shape[0]
    delta = motion - mean
    mean += delta.sum(axis=0) / count
    delta2 = motion - mean
    M2 += np.sum(delta * delta2, axis=0)

# Finalize the mean and standard deviation
std = np.sqrt(M2 / count)
std[np.isnan(std)] = 1e-3
std[std < 1e-3] = 1e-3
# Save the computed statistics
np.save(os.path.join(out_stats_folder, 'mean.npy'), mean)
np.save(os.path.join(out_stats_folder, 'std.npy'), std)
print('mean:', mean)
print('std:', std)

