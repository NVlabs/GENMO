import os
import sys
sys.path.append('./')
import argparse
import glob
import pickle
import torch
import lmdb
import numpy as np
from tqdm import tqdm

from motiondiff.utils.tools import import_type_from_str
from motiondiff.utils.config import create_config


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='data_aug/HumanML3D_gpt_aug/part1.pkl')
parser.add_argument('-o', '--out_lmdb', default='data_aug/T5_embed/HumanML3D_gpt_aug_part1_debug')
parser.add_argument('-c', '--cfg', default='mdm_t5_enc_cat_len50_aug_amp')
parser.add_argument('-bs', '--batch_size', type=int, default=256)
args = parser.parse_args()


text_mapping = pickle.load(open(args.input_file, 'rb'))
all_texts = sum(list(text_mapping.values()), [])
print(len(all_texts))

torch.set_grad_enabled(False)
cfg = create_config(args.cfg, tmp=True, training=False)
model = import_type_from_str(cfg.model.type)(cfg)
model.cuda()

num_data = 0
env = lmdb.open(args.out_lmdb, map_size=int(1e11))
with env.begin(write=True) as txn:
    for i in tqdm(range(0, len(all_texts), args.batch_size)):
        text_chunk = all_texts[i:min(i+args.batch_size, len(all_texts))]
        embedding = model.denoiser.encode_text(text_chunk)
        embedding = embedding.cpu().numpy()
        for v, embed in zip(text_chunk, embedding):
            # print(v, len(v), len(v.encode()))
            if len(v.encode()) >= 511:
                print('bad text:', v)
                continue
            txn.put(v.encode(), embed)
        num_data += embedding.shape[0]
        # if num_data >= 300:
        #     break

print(num_data)
# Close the environment
env.close()

# ## test reading from lmdb
# env = lmdb.open(args.out_lmdb, readonly=True)
# with env.begin() as txn:
#     # Retrieve individual NumPy arrays
#     embed = np.frombuffer(txn.get(all_texts[0].encode()), dtype=np.float32).reshape(-1, 1024)
#     print("embed shape:", embed.shape)
# # Close the environment
# env.close()

# # embed1 = model.denoiser.encode_text([all_texts[0]]).cpu().numpy()
# diff = embed1 - embed
# print(np.linalg.norm(diff))