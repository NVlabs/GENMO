"""
Process the GRAB data to be in the form aligend with other datasets
The code is adapted from Jinkun's private code
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

GRAB_PATH = "/lustre/fsw/portfolios/nvr/users/jinkunc/datasets/GRAB_V00"
SAVE_PATH = "/lustre/fsw/portfolios/nvr/users/jinkunc/datasets/GRAB_motiondiff"
STAT_PATH = "assets/hand/stats"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(STAT_PATH, exist_ok=True)

splits = ["train", "val", "test"]


"""
    Following the MANO repo: https://github.com/otaheri/MANO/blob/master/mano/joints_info.py
    we add the tip joints into the original MANO joints by selecting from vertices
"""
TIP_IDS = {
    "mano": {
        "thumb": 744,
        "index": 320,
        "middle": 443,
        "ring": 554,
        "pinky": 671,
    }
}

tip_ids = TIP_IDS["mano"]
tip_ids = torch.from_numpy(np.array(list(tip_ids.values())))

# The order and name for the 21 joints
JOINT_NAMES = [
    "wrist",  # 0
    "index1",  # 1
    "index2",
    "index3",
    "middle1",  # 4
    "middle2",
    "middle3",
    "pinky1",  # 7
    "pinky2",
    "pinky3",
    "ring1",  # 10
    "ring2",
    "ring3",
    "thumb1",  # 13
    "thumb2",
    "thumb3",
    "thumb_tip",  # 16
    "index_tip",  # 17
    "middle_tip",  # 18
    "ring_tip",  # 19
    "pinky_tip",  # 20
    # 'index_tip', # 16
    # 'middle_tip', # 17
    # 'pinky_tip', # 18
    # 'ring_tip', # 19
    # 'thumb_tip', # 20
]

colors = {
    "pink": [1.00, 0.75, 0.80],
    "skin": [0.96, 0.75, 0.69],
    "purple": [0.63, 0.13, 0.94],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0],
    "brown": [1.00, 0.25, 0.25],
    "blue": [0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "orange": [1.00, 0.65, 0.00],
    "grey": [0.75, 0.75, 0.75],
    "black": [0.0, 0.0, 0.0],
}

LINKS = [
    [0, 1],
    [1, 2],
    [2, 3],  # index finger
    [0, 4],
    [4, 5],
    [5, 6],  # middle finger
    [0, 7],
    [7, 8],
    [8, 9],  # pinky finger
    [0, 10],
    [10, 11],
    [11, 12],  # ring finger
    [0, 13],
    [13, 14],
    [14, 15],  # thumb finger
    [15, 16],
    [3, 17],
    [6, 18],
    [9, 20],
    [12, 19],
]  # tips
# [3,16],[6,17],[9,18],[12,20],[15,19]]


body_model_path = "assets/body_models"


def viz_hand(joints, ax, color="red"):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="r")
    for i in range(joints.shape[1]):
        # ax.text(lhand_all_joints[0, i, 0], lhand_all_joints[0, i, 1], lhand_all_joints[0, i, 2], JOINT_NAMES[i])
        # ax.text(joints[i, 0], joints[i, 1], joints[i, 2], str(i))
        pass

    for link in LINKS:
        first = link[0]
        second = link[1]
        ax.plot(
            [joints[first, 0], joints[second, 0]],
            [joints[first, 1], joints[second, 1]],
            [joints[first, 2], joints[second, 2]],
            c="b",
        )
    return ax


def load(datasets):
    loaded = {}
    for d in datasets:
        k = os.path.basename(d).split("_")[0]
        loaded[k] = torch.load(d)
    return loaded


viz_mesh_idx = np.random.choice(778, 500)


def process_split(dataset_dir, split):
    split_save_path = os.path.join(SAVE_PATH, split)
    os.makedirs(split_save_path, exist_ok=True)
    frame_slicer = f"{dataset_dir}/{split}_frame_nums.txt"
    episodes = []
    seqnames = []
    total_frames = 0
    for line in open(frame_slicer, "r").readlines():
        seq_name, n_frames = line.strip().split()
        seqnames.append(seq_name)
        episodes.append(int(n_frames))

    datasets = glob.glob(f"{dataset_dir}/{split}/*.pt")
    features = load(datasets)

    lhand_feature = torch.cat(
        [
            features["lhand"]["global_orient"],
            features["lhand"]["transl"],
            features["lhand"]["fullpose"],
        ],
        dim=1,
    )
    rhand_feature = torch.cat(
        [
            features["rhand"]["global_orient"],
            features["rhand"]["transl"],
            features["rhand"]["fullpose"],
        ],
        dim=1,
    )
    hand_feature = torch.cat([lhand_feature, rhand_feature], dim=1)

    all_feature = []

    for epi_idx in range(len(episodes)):
        episode = episodes[epi_idx]
        seqname = seqnames[epi_idx]
        eps_feature = hand_feature[total_frames : total_frames + episode].numpy()

        rhand_m = smplx.create(
            model_path=body_model_path,
            model_type="mano",
            flat_hand_mean=True,
            # gender=gender,
            # num_pca_comps=n_comps,
            use_pca=False,
            # v_template=rhand_vtemp,
            batch_size=eps_feature.shape[0],
        )

        lhand_m = smplx.create(
            model_path=body_model_path,
            model_type="mano",
            flat_hand_mean=True,
            # gender=gender,
            # num_pca_comps=n_comps,
            use_pca=False,
            # v_template=lhand_vtemp,
            is_rhand=False,
            batch_size=eps_feature.shape[0],
        )

        lhand_mano = lhand_m(
            global_orient=features["lhand"]["global_orient"][
                total_frames : total_frames + episode
            ].float(),
            hand_pose=features["lhand"]["fullpose"][
                total_frames : total_frames + episode
            ].float(),
            transl=features["lhand"]["transl"][
                total_frames : total_frames + episode
            ].float(),
        )

        rhand_mano = rhand_m(
            global_orient=features["rhand"]["global_orient"][
                total_frames : total_frames + episode
            ].float(),
            hand_pose=features["rhand"]["fullpose"][
                total_frames : total_frames + episode
            ].float(),
            transl=features["rhand"]["transl"][
                total_frames : total_frames + episode
            ].float(),
        )

        lhand_joints = lhand_mano.joints.detach().cpu().numpy()
        rhand_joints = rhand_mano.joints.detach().cpu().numpy()

        lhand_vertices = lhand_mano.vertices.detach().cpu()
        lhand_tip_joints = torch.index_select(lhand_vertices, 1, tip_ids).numpy()
        rhand_vertices = rhand_mano.vertices.detach().cpu()
        rhand_tip_joints = torch.index_select(rhand_vertices, 1, tip_ids).numpy()

        lhand_all_joints = np.concatenate([lhand_joints, lhand_tip_joints], axis=1)
        rhand_all_joints = np.concatenate([rhand_joints, rhand_tip_joints], axis=1)

        joint_features = np.concatenate(
            [lhand_all_joints, rhand_all_joints], axis=1
        ).reshape(episode, -1)
        all_eps_feature = np.concatenate([eps_feature, joint_features], axis=1)

        all_feature.append(all_eps_feature.copy())

        ############ visualization #############
        # lhand_all_joints[:, 0, :] = lhand_all_joints[:, 0, :] - lhand_all_joints[0, 0, :] # relative to the first frame
        # rhand_all_joints[:, 0, :] = rhand_all_joints[:, 0, :] - rhand_all_joints[0, 0, :]
        # lhand_all_joints[:, 1:, :] = lhand_all_joints[:, 1:, :] - lhand_all_joints[:, 0:1, :] # relative to the wrist pos
        # rhand_all_joints[:, 1:, :] = rhand_all_joints[:, 1:, :] - rhand_all_joints[:, 0:1, :]

        # lhand_all_joints[:, 0, :] = 0
        # save_dir = 'vis_debug_order_mesh+joint'

        # lhand_vertices = lhand_mano.vertices.detach().cpu() - lhand_all_joints[:, 0:1]
        # lhand_vertices[:, :, 0] += 0.5

        # os.makedirs(f'{save_dir}/{epi_idx}', exist_ok=True)
        # lhand_all_joints[:, 1:] = lhand_all_joints[:, 1:] - lhand_all_joints[:, 0:1]
        # lhand_all_joints[:, 0, :] = 0
        # x_min = lhand_all_joints[:, :, 0].min()
        # x_max = lhand_all_joints[:, :, 0].max() + 0.5
        # y_min = lhand_all_joints[:, :, 1].min()
        # y_max = lhand_all_joints[:, :, 1].max()
        # z_min = lhand_all_joints[:, :, 2].min()
        # z_max = lhand_all_joints[:, :, 2].max()

        # for i in range(min(lhand_all_joints.shape[0], 196)):
        #     fig = plt.figure()
        #     plt.axis('off')
        #     ax = fig.add_subplot(111, projection='3d')
        #     # ax.get_xaxis().set_ticks([])
        #     # ax.get_yaxis().set_ticks([])
        #     # ax.get_zaxis().set_ticks([])
        #     # if 50 < i and i < 100:
        #     #     lhand_all_joints[i, 1:, :] = lhand_all_joints[i, 1:, :] + np.random.random(size=lhand_all_joints[i, 1:, :].shape) * 0.001
        #     #     ax = viz_hand(lhand_all_joints[i], ax)
        #     # else:

        #     ax.set_xlim3d(x_min - 0.01 * (x_max-x_min), x_max + 0.01 * (x_max-x_min))
        #     ax.set_ylim3d(y_min - 0.01 * (y_max-y_min), y_max + 0.01 * (y_max-y_min))
        #     ax.set_zlim3d(z_min - 0.01 * (z_max - z_min), z_max + 0.01 * (z_max - z_min))

        #     verts = lhand_vertices[i][viz_mesh_idx]
        #     ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='green')

        #     ax = viz_hand(lhand_all_joints[i], ax)

        #     # plt.xlim(-1,1)
        #     # plt.ylim(-1,1)
        #     # plt.zlim(-1,1)
        #     plt.savefig('%s/%d/%04d.png' % (save_dir, epi_idx, i))
        #     print("Saving %s/%d/%04d.png" % (save_dir, epi_idx, i))
        ############ End: visualization #############

        total_frames += episode
        # np.save(os.path.join(split_save_path, '%s.npy' % seqname), all_eps_feature)
        print("saved: ", os.path.join(split_save_path, "%s.npy" % seqname))

    if split == "train":
        # we save the stats from the training split
        all_feature = np.concatenate(all_feature, axis=0)
        mean = all_feature.mean(axis=0)
        std = all_feature.std(axis=0)

        # np.save(os.path.join(STAT_PATH, 'mean.npy'), mean)
        # np.save(os.path.join(STAT_PATH, 'std.npy'), std)
    # preprocess_frame_num = features['lhand']['fullpose'].shape[0]
    # assert total_frames == preprocess_frame_num, f"frame number mismatch: {total_frames} vs {preprocess_frame_num}"
    print(f"processed {split} split")


for split in splits:
    process_split(GRAB_PATH, split)
