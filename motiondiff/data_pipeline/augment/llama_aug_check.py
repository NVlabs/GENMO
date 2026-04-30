import os
import sys

sys.path.append("./")
import argparse
import time

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=350000)
args = parser.parse_args()

out_dir = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_aug_texts/v1"
meta_file = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.1/meta_240416_v3.csv"
# meta_file = 'out/meta1.csv'
meta = pd.read_csv(meta_file)


def read_text_augment_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


for i in range(args.start, args.end):
    t0 = time.time()

    row = meta.iloc[i]
    texts = [
        str(row["natural_desc_1"]),
        str(row["natural_desc_2"]),
        str(row["natural_desc_3"]),
        str(row["technical_description"]),
        str(row["short_description"]),
    ]

    for k, text in enumerate(texts):
        if len(text) <= 2 or text == "nan":
            continue
        out_path = f"{out_dir}/{i:06d}-{k}.txt"
        if not os.path.exists(out_path):
            print(f"[Non-Exists] {out_path} {text}")
            continue

        aug_texts = read_text_augment_file(out_path)
        if len(aug_texts) != 13:
            print(f"[Error] {out_path} {text}")
            print("num_aug:", len(aug_texts))
            print(aug_texts)
            os.remove(out_path)
            continue

        # if text != aug_texts[0]:
        #     print(f'[Text-Error] {out_path} {text}')
        #     print(aug_texts)
        #     continue

    if i % 10000 == 0:
        print(f"{i}/({args.start}, {args.end}) processed, time: {time.time() - t0:.2f}")
