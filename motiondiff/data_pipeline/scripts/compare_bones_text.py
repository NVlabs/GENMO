import os
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--initial_csv_path', type=str, default='dataset/bones/bones_full_raw/Metadata_240416_v3.csv')
parser.add_argument('--new_csv_path', type=str, default='dataset/bones/bones_full_raw_v14/metadata_240527_v014.csv')
parser.add_argument('--log_path', type=str, default='dataset/bones/bones_full353_v2.0/meta_change_log_240527_v014.txt')
args = parser.parse_args()


if __name__ == '__main__':

    initial_csv = pd.read_csv(args.initial_csv_path)
    new_csv = pd.read_csv(args.new_csv_path)

    initial_bvh_map = {path.split('/')[-1]: i for i, path in enumerate(initial_csv.move_bvh_path)}

    initial_bvh_paths = [path.split('/')[-1] for i, path in enumerate(initial_csv.move_bvh_path)]
    new_bvh_paths = [path.split('/')[-1] for i, path in enumerate(new_csv.move_bvh_path)]

    num_updates = 0
    num_not_found = 0
    f_change = open(args.log_path, 'w')
    for i, bvh_path in enumerate(tqdm(new_bvh_paths)):
        j = initial_bvh_map.get(bvh_path, None)
        if j is not None:
            for col in ['content_natural_desc_1', 'content_natural_desc_2', 'content_natural_desc_3', 'content_technical_description', 'content_short_description']:
                initial_val = initial_csv[col][j]
                new_val = new_csv[col][i]

                # if isinstance(initial_val, str) and initial_val[:-1] == new_val: continue
                # if isinstance(initial_val, str) and initial_val == new_val[:-1]: continue

                if initial_val != new_val:
                    num_updates += 1
                    f_change.write(f'{bvh_path} --- [{col}]:\n{initial_val}\n{new_val}\n\n')
        else:
            num_not_found += 1

    f_change.close()
    print(f"Find {num_updates} text updates, {num_not_found} bvh files not found in old meta")