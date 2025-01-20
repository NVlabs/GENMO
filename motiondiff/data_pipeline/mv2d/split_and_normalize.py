import numpy as np
import os
import sys
import pandas as pd
import pickle
from tqdm import tqdm
sys.path.append('./')

seed = 7
regen = False
np.random.seed(seed)
split_version = 'v1'
stats_version = 'v1'
out_split_folder = f'assets/mv2d/splits/{split_version}'
out_stats_folder = f'assets/mv2d/stats/{stats_version}'
skip_index = np.load('assets/bones/skip_idx/v14/skip_index_geq_2artifacts.npy')
split_by_content = False
smpl_dir = '/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_to_smpl/bones_to_smpl_v14.7/smpl'
meta_file = '/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw_v14/metadata_240527_v014.csv'
meta = pd.read_csv(meta_file)
bvh_files = meta['move_bvh_path'].values
motion_names = [x[4:].replace('.bvh', '') for x in bvh_files]
os.makedirs(out_split_folder, exist_ok=True)
# os.makedirs(out_stats_folder, exist_ok=True)

bone_dataset_len = 350000

existing_smpl_ind_file = f'{out_split_folder}/existing_smpl_ind.npy'
filtered_smpl_ind_file = f'{out_split_folder}/filtered_smpl_ind.npy'
if not regen and os.path.exists(existing_smpl_ind_file) and os.path.exists(filtered_smpl_ind_file):
    existing_smpl_ind = np.load(existing_smpl_ind_file)
    filtered_smpl_ind = np.load(filtered_smpl_ind_file)
else:
    existing_smpl_ind = []
    filtered_smpl_ind = []
    for i, name in enumerate(motion_names):
        if i in skip_index:
            continue
        fname = os.path.join(smpl_dir, f'{name}.npz')
        if os.path.exists(fname):
            try:
                loss = np.load(fname)['losses']
                existing_smpl_ind.append(i)
                if loss.mean() < 0.4:
                    filtered_smpl_ind.append(i)
                    print(i, len(existing_smpl_ind), len(filtered_smpl_ind))
            except KeyboardInterrupt:
                exit()
            except:
                print(f'Error in {fname}')
                pass
    np.save(f'{out_split_folder}/existing_smpl_ind.npy', np.array(existing_smpl_ind))
    np.save(f'{out_split_folder}/filtered_smpl_ind.npy', np.array(filtered_smpl_ind))
# exit()


""" Split """
num_test_motions = 5000
train_index_file = os.path.join(out_split_folder, 'train_index.npy')
test_index_file = os.path.join(out_split_folder, 'test_index.npy')
all_index = filtered_smpl_ind.copy()
total_len = len(all_index)
np.random.shuffle(all_index)
train_index = all_index[:-num_test_motions]
test_index = all_index[-num_test_motions:]
train_index = np.sort(train_index)
test_index = np.sort(test_index)
np.save(os.path.join(out_split_folder, 'train_index.npy'), train_index)
np.save(os.path.join(out_split_folder, 'test_index.npy'), test_index)
print(f'index files saved to {out_split_folder}')

rand_keys = ['angle_y', 'radius']
randn_keys = ['shape', 'elevation']
rng_dict = {}
for key in rand_keys:
    rng_dict[key] = np.random.rand(len(filtered_smpl_ind))
for key in randn_keys:
    if key == 'shape':
        rng_dict[key] = np.random.randn(len(filtered_smpl_ind), 10)
    else:
        rng_dict[key] = np.random.randn(len(filtered_smpl_ind))
pickle.dump(rng_dict, open(f'{out_split_folder}/rng_dict.pkl', 'wb'))
print(f'rng_dict saved to {out_split_folder}')
