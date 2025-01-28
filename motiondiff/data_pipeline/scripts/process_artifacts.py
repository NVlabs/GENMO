"""
Go through check_artifacts.csv (output of test_bones_bvh_check_artifacts.py)
and process to better column format. Also save list of indices into the original
metadata file for which sequences contain artifacts.
"""

import csv

import numpy as np

# input_artifacts = './out/bones_check_artifacts/check_artifacts.csv'
# output_artifacts = './out/bones_check_artifacts/check_artifacts_processed.csv'
input_artifacts = "./out/bones_check_artifacts_v14/check_artifacts.csv"
output_artifacts = "./out/bones_check_artifacts_v14/check_artifacts_processed.csv"

# meta_path = '../bones_data/foundation_data/Metadata - 350 000 moves.csv'
# skip_all_out_path = './out/bones_check_artifacts/skip_index_all.npy'
# skip_three_out_path = './out/bones_check_artifacts/skip_index_geq_3artifacts.npy'
# skip_two_out_path = './out/bones_check_artifacts/skip_index_geq_2artifacts.npy'
meta_path = "../bones_data/foundation_data/v014_retarget/metadata_240527_v014.csv"
skip_all_out_path = "./out/bones_check_artifacts_v14/skip_index_all.npy"
skip_three_out_path = "./out/bones_check_artifacts_v14/skip_index_geq_3artifacts.npy"
skip_two_out_path = "./out/bones_check_artifacts_v14/skip_index_geq_2artifacts.npy"

added_skips = [187418]  # problematic files

# print(np.load(skip_out_path))


def extract_numbers(text):
    toks = text.split("(")
    artifact_name = toks[0]
    num = toks[1].split(")")[0]
    return [artifact_name, num]


# get indices for each bvh name
name2idx = dict()
print("Loading metadata...")
with open(meta_path, "r") as f:
    reader = csv.DictReader(f)
    for ri, row in enumerate(reader):
        bvh_path = row["move_bvh_path"]
        name2idx[bvh_path] = ri

skip_inds = dict()  # idx -> num_artifacts
skip_files = set()
with (
    open(input_artifacts, "r") as f_in,
    open(output_artifacts, "w", newline="") as f_out,
):
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    for row in reader:
        file_path = row[0].strip('"')
        if file_path in skip_files:
            # duplicate
            continue

        new_cols = list(map(extract_numbers, row[1:]))

        row_out = []
        for col in new_cols:
            row_out += col

        # Write the reformatted row to the output CSV file
        writer.writerow([file_path, *row_out])

        num_artifacts = len(row_out[::2])
        skip_files.add(file_path)
        skip_inds[name2idx[file_path]] = num_artifacts
        # also need to skip  mirrored version
        mirror_path = file_path[:-4] + "_M.bvh"
        skip_inds[name2idx[mirror_path]] = num_artifacts

skip_all_idx = list(skip_inds.keys()) + added_skips
print(f"{len(skip_all_idx)} total motions to skip, saving...")
np.save(skip_all_out_path, np.array(skip_all_idx, dtype=int))

skip_two_idx = [
    i for i, num_artifacts in skip_inds.items() if num_artifacts >= 2
] + added_skips
print(f"{len(skip_two_idx)} motions >= 2 artifacts to skip, saving...")
np.save(skip_two_out_path, np.array(skip_two_idx, dtype=int))

skip_three_idx = [
    i for i, num_artifacts in skip_inds.items() if num_artifacts >= 3
] + added_skips
print(f"{len(skip_three_idx)} motions >= 3 artifacts to skip, saving...")
np.save(skip_three_out_path, np.array(skip_three_idx, dtype=int))
