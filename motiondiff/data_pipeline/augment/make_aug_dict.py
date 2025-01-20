import os
import sys
sys.path.append('./')
import argparse
import glob
import pickle
from tqdm import tqdm
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir_query', nargs='+', default=['dataset/HumanML3D_gpt_aug/part1'])
parser.add_argument('-o', '--out_file', default='dataset/HumanML3D_gpt_aug/part1.pkl')
args = parser.parse_args()

txt_files = []
for query in args.dir_query:
	txt_files += sorted(glob.glob(f'{query}/**.txt', recursive=True))

text_mapping = {}
text_counter = defaultdict(int)

# text_mapping = pickle.load(open(args.out_file, 'rb'))
# print(len(text_mapping))

for f in tqdm(txt_files):
    # read lines from file
	with open(f, 'r') as f:
		lines = f.readlines()
		lines = [x.strip() for x in lines]
		text_mapping[lines[0]] = lines
		text_counter[lines[0]] += 1
		# print(lines[0], text_mapping[lines[0]])

for k, v in text_counter.items():
	if v > 1:
		print(k, v)

pickle.dump(text_mapping, open(args.out_file, 'wb'))