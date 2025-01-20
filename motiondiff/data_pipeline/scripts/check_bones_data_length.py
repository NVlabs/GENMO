import sys
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints'
# data_dir = 'dataset/bones_full_joints'

length_dict_path = 'bones_data_length.json'
if not os.path.exists(length_dict_path):
    length_dict = {}
    for sub in tqdm(os.listdir(data_dir)):
        sub_dir = os.path.join(data_dir, sub)
        for path in tqdm(os.listdir(os.path.join(sub_dir, 'posed_joints'))):
            full_path = os.path.join(sub_dir, 'posed_joints', path)
            joints = np.load(full_path)
            length_dict[path] = joints.shape[0]
    json.dump(length_dict, open('', 'w'))
else:
    length_dict = json.load(open(length_dict_path, 'r'))

plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(length_dict)), sorted(np.array(list(length_dict.values())) / 120), s = 1)
plt.xlabel('Clips')
plt.ylabel('Length (second)')
plt.title('Bones clip length distribution')
plt.savefig('bones_data_length.png')

bins = [120, 240, 600, 1200, 1800, 3600, 7200, 14400, 30000]
cnt = [0 for i in range(len(bins))]

for k, v in length_dict.items():
    for i, b in enumerate(bins):
        if v <= b:
            cnt[i] += 1
            break
assert sum(cnt) == len(length_dict)
print(cnt)

print('Min', min(length_dict.values()))
print('Max', max(length_dict.values()))
print('Total', sum(length_dict.values()))