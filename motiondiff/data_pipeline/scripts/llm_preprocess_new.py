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
parser.add_argument('-o', '--out_dir', default='data_aug/T5_embed/part1_key')
# parser.add_argument('-o', '--out_dir', default='data_aug/T5_embed/part1_value')
parser.add_argument('-c', '--cfg', default='mdm_t5_enc_cat_len50_aug_amp')
parser.add_argument('-bs', '--batch_size', type=int, default=512)
parser.add_argument('-sp', '--split_size', type=int, default=10000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

text_mapping = pickle.load(open(args.input_file, 'rb'))
all_texts = list(text_mapping.keys())
# all_texts = sum(list(text_mapping.values()), [])
print('total num of texts:', len(all_texts))

torch.set_grad_enabled(False)
cfg = create_config(args.cfg, tmp=True, training=False)
model = import_type_from_str(cfg.model.type)(cfg)
model.cuda()


num_text_splits = int(np.ceil(len(all_texts) / args.split_size))
for k in range(num_text_splits):
    print(f'processing text split {k}')
    split_texts = all_texts[k*args.split_size:min((k+1)*args.split_size, len(all_texts))]
    print(f'split {k} num texts: {len(split_texts)}')

    split_embeddings = {}
    for i in tqdm(range(0, len(split_texts), args.batch_size)):
        print(f'split {k} processing batch {i}')
        text_chunk = split_texts[i:min(i+args.batch_size, len(split_texts))]
        embedding = model.denoiser.encode_text(text_chunk)
        embedding = embedding.cpu().numpy()
        for v, embed in zip(text_chunk, embedding):
            split_embeddings[v] = embed
            # print(v, embed.shape)

    print(f'split {k} num embeddings: {len(split_embeddings)}')

    split_pickle_file = os.path.join(args.out_dir, f'split_embeddings_{k}.pkl')
    # Save the list of embeddings as a pickle file
    with open(split_pickle_file, 'wb') as pickle_file:
        pickle.dump(split_embeddings, pickle_file)
