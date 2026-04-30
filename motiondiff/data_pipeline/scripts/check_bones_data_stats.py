import os
from collections import OrderedDict
from typing import DefaultDict

import pandas as pd
from tqdm import tqdm

# data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
data_dir = "dataset/bones_full_raw"
# csv_path = os.path.join(data_dir, 'Metadata - 350 000 moves.csv')
csv_path = os.path.join(data_dir, "Metadata_240416_v3.csv")

csv = pd.read_csv(csv_path)

actor_dict = DefaultDict(int)
action_dict = DefaultDict(int)
style_dict = DefaultDict(int)
content_dict = DefaultDict(int)

f_actor = open("bones_actor_stats.txt", "w")
f_action = open("bones_action_stats.txt", "w")
f_style = open("bones_style_stats.txt", "w")
f_content = open("bones_content_stats.txt", "w")

for actor in tqdm(csv.actor_uid):
    actor_dict[str(actor)] += 1
    # if not isinstance(actor, str):
    #     print(actor)
for action in tqdm(csv.content_type_of_movement):
    action_dict[str(action)] += 1
for style in tqdm(csv.content_uniform_style):
    style_dict[str(style)] += 1
for content in tqdm(csv.content_name):
    content_dict[str(content)] += 1

for k, v in sorted(actor_dict.items(), key=lambda x: x[1]):
    f_actor.write(f"{k}: {v}\n")
for k, v in sorted(action_dict.items(), key=lambda x: x[1]):
    f_action.write(f"{k}: {v}\n")
for k, v in sorted(style_dict.items(), key=lambda x: x[1]):
    f_style.write(f"{k}: {v}\n")
for k, v in sorted(content_dict.items(), key=lambda x: x[1]):
    f_content.write(f"{k}: {v}\n")

f_actor.close()
f_action.close()
f_style.close()
f_content.close()

# Check missing texts
f_texts = open("bones_missing_texts.txt", "w")
missing_texts = DefaultDict(int)
for col in [
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_technical_description",
    "content_short_description",
]:
    for i in range(len(csv.move_bvh_path)):
        if not isinstance(csv[col][i], str):
            missing_texts[col] += 1
            f_texts.write(f"{csv.move_bvh_path[i]} missing {col}\n")
print(missing_texts)
f_texts.close()
