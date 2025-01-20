import pandas as pd
import os
from tqdm import tqdm 

data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
# data_dir = 'dataset/bones_full_raw'
csv_path = os.path.join(data_dir, 'Metadata_240416_v3.csv')

csv = pd.read_csv(csv_path)
bvh_path_list = csv.move_bvh_path
fbx_path_list = csv.move_fbx_path
mov_path_list = csv.refcam_path

# Check BVH files
missing_cnt = 0
duplicate_cnt = 0
duplicate_set = set()
wrong_folder_cnt = 0
not_bvh_cnt = 0
f_missing = open('bones_missing_bvh.txt', 'w')
f_duplicate = open('bones_duplicate_bvh.txt', 'w')
f_wrong_folder = open('bones_wrong_folder_bvh.txt', 'w')
f_not_bvh = open('bones_not_bvh.txt', 'w')
for path in tqdm(bvh_path_list):
    if not 'P15' in path: continue
    if not os.path.exists(os.path.join(data_dir, path)):
        missing_cnt += 1
        f_missing.write(path + '\n')
    if path.split("/")[2] in duplicate_set:
        duplicate_cnt += 1
        f_duplicate.write(path + '\n')
    duplicate_set.add(path.split("/")[2])
    if 'BVH' not in path:
        wrong_folder_cnt += 1
        f_wrong_folder.write(path + '\n')
    if path[-3:] != 'bvh':
        not_bvh_cnt += 1
        f_not_bvh.write(path + '\n')
f_missing.close()
f_duplicate.close()
f_wrong_folder.close()
f_not_bvh.close()
print(f"{missing_cnt} bvh files missing, {duplicate_cnt} duplicate bvh files, {wrong_folder_cnt} files not in BVH sub-folders, {not_bvh_cnt} files not ended with .bvh,  ")
exit()


# Check BFX files
missing_cnt = 0
duplicate_cnt = 0
duplicate_set = set()
wrong_folder_cnt = 0
not_fbx_cnt = 0
f_missing = open('bones_missing_fbx.txt', 'w')
f_duplicate = open('bones_duplicate_fbx.txt', 'w')
f_wrong_folder = open('bones_wrong_folder_fbx.txt', 'w')
f_not_fbx = open('bones_not_fbx.txt', 'w')
for path in tqdm(fbx_path_list):
    if not os.path.exists(os.path.join(data_dir, path)):
        missing_cnt += 1
        f_missing.write(path + '\n')
    if path.split("/")[2] in duplicate_set:
        duplicate_cnt += 1
        f_duplicate.write(path + '\n')
    duplicate_set.add(path.split("/")[2])
    if 'FBX' not in path:
        wrong_folder_cnt += 1
        f_wrong_folder.write(path + '\n')
    if path[-3:] != 'fbx':
        not_fbx_cnt += 1
        f_not_fbx.write(path + '\n')
f_missing.close()
f_duplicate.close()
f_wrong_folder.close()
f_not_fbx.close()
print(f"{missing_cnt} fbx files missing, {duplicate_cnt} duplicate fbx files, {wrong_folder_cnt} files not in FBX sub-folders, {not_fbx_cnt} files not ended with .fbx,  ")


    
# Check MOV files
missing_mov_cnt = 0
f_missing_mov = open('bones_missing_mov.txt', 'w')
for path in tqdm(mov_path_list):
    if not os.path.exists(os.path.join(data_dir, path)):
        missing_mov_cnt += 1
        f_missing_mov.write(path + '\n')
f_missing_mov.close()
print(f"{missing_mov_cnt} video files missing")


# Check misplaced files
bvh_cnt = 0
wrong_bvh_cnt = 0
fbx_cnt = 0
wrong_fbx_cnt = 0
mov_cnt = 0
wrong_mov_cnt = 0
f_wrong_bvh = open('bones_non_bvh_under_BVH.txt', 'w')
f_wrong_fbx = open('bones_non_fbx_under_FBX.txt', 'w')
f_wrong_mov = open('bones_non_mov_under_REFCAMS.txt', 'w')
for sub in tqdm(os.listdir(data_dir)):
    if sub[0] != 'P': continue
    sub_dir = os.path.join(data_dir, sub)
    for path in os.listdir(os.path.join(sub_dir, 'BVH')):
        if path[-3:] == 'bvh':
            bvh_cnt += 1
        else:
            wrong_bvh_cnt += 1
            f_wrong_bvh.write(path + '\n')
    for path in os.listdir(os.path.join(sub_dir, 'FBX')):
        if path[-3:] == 'fbx':
            fbx_cnt += 1
        else:
            wrong_fbx_cnt += 1
            f_wrong_fbx.write(path + '\n')
    for path in os.listdir(os.path.join(sub_dir, 'REFCAMS')):
        if path[-3:] == 'mov' or path[-3:] == 'mp4':
            mov_cnt += 1
        else:
            wrong_mov_cnt += 1
            f_wrong_mov.write(path + '\n')
print(f"{bvh_cnt} BVH files, {wrong_bvh_cnt} non BVH files under BVH folders")
print(f"{fbx_cnt} FBX files, {wrong_fbx_cnt} non FBX files under FBX folders")
print(f"{mov_cnt} MOV files, {wrong_mov_cnt} non MOV files under REFCAMS folders")
f_wrong_bvh.close()
f_wrong_fbx.close()
f_wrong_mov.close()
