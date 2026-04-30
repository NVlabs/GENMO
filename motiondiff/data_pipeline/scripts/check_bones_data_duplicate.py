import hashlib
import os
import pdb
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw"
# data_dir = 'dataset/bones_full_raw'
csv_path = os.path.join(data_dir, "Metadata - 350 000 moves.csv")

csv = pd.read_csv(csv_path)

hash_dict = {}
duplicate_cnt = 0
f = open("bones_duplicate_bvh_file.txt", "w")
for path in tqdm(csv.move_bvh_path):
    if not os.path.exists(os.path.join(data_dir, path)):
        continue

    file = open(os.path.join(data_dir, path), "rb").read()
    h = hashlib.md5(file).hexdigest()

    if h in hash_dict:
        duplicate_cnt += 1
        print(f"Find duplicated bvh file: {hash_dict[h]} and {path}")
        f.write(f"{hash_dict[h]}, {path}\n")
    else:
        hash_dict[h] = path
f.close()
