import os
import sys

sys.path.append("./")
import argparse
import glob
import pickle

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from motiondiff.utils.config import create_config
from motiondiff.utils.tools import import_type_from_str

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input_file", default="data_aug/HumanML3D_gpt_aug/part1.pkl"
)
parser.add_argument("-w", "--word", default="luggage")
args = parser.parse_args()


text_mapping = pickle.load(open(args.input_file, "rb"))
orig_texts = list(text_mapping.keys())

find_texts = [text for text in orig_texts if args.word in text]
print(find_texts)
print(len(find_texts))
