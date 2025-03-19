import glob
import os
import subprocess

input_dirs = [
    # "out/v1-200-text-500-v2",
    # "out/v1-200-text-600-v2",
    # "out/v1-300-text-500-v2",
    # "out/v1-300-text-600-v2",
    "out/v1-new"
]

# filter_fnames = [
#     '003-P2_24_outdoor_long_walk-P4_37_outdoor_run_circle-the_body_climbs_up_something_and_turns_back_to_',
#     '006-P3_29_outdoor_stairs_up-P4_37_outdoor_run_circle-a_person_lunges_forwards_on_their_right_leg__st.pth',
# ]

# for input_dir in input_dirs:
#     vid_files = sorted(glob.glob(f"{input_dir}/*.pth"))
#     for vid_file in vid_files:
#         fname = os.path.basename(vid_file)
#         # if fname in filter_fnames:
#         subprocess.run(["python3", "scripts/vis_vtv.py", "--file_name", vid_file])


# vid_files = [
#     'out/v1-200-new/006-P2_24_outdoor_long_walk-P4_37_outdoor_run_circle-raising_hands_above_head_and_stretching..pth',
#     'out/v1-200-new/006-P3_29_outdoor_stairs_up-P4_37_outdoor_run_circle-a_person_lunges_forwards_on_their_right_leg__st.pth',
# ]

# for vid_file in vid_files:
#     fname = os.path.basename(vid_file)
#     # if fname in filter_fnames:
#     # subprocess.run(["python", "scripts/vis_vtv.py", "--file_name", vid_file])
#     subprocess.run(["/workspace/isaaclab/_isaac_sim/python.sh", "scripts/vis_vtv.py", "--file_name", vid_file])


vid_files = [
    "out/v1-new/001-P2_24_outdoor_long_walk-P4_37_outdoor_run_circle-a_person_lunges_forwards_on_their_left_leg__sta.pth",
]

for vid_file in vid_files:
    fname = os.path.basename(vid_file)
    # if fname in filter_fnames:
    # subprocess.run(["python", "scripts/vis_vtv.py", "--file_name", vid_file])
    subprocess.run(
        [
            "/workspace/isaaclab/_isaac_sim/python.sh",
            "scripts/vis_vtv_key.py",
            "--file_name",
            vid_file,
        ]
    )
