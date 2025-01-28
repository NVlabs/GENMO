import os
from copy import copy

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils import data

from motiondiff.data_pipeline.tensors import collate
from motiondiff.utils.tools import wandb_run_exists


# an adapter to our collate func
def bones_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[1].T)
            .float()
            .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "target": 0,
            "text": b[0],  # b[0]['caption']
            "lengths": b[2],
        }
        for b in batch
    ]
    return collate(adapted_batch)


def read_text_augment_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


class BonesDataset(data.Dataset):
    def __init__(
        self,
        split,
        num_frames,
        meta_file="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.0/meta_240416_v3.csv",
        # meta_file='out/meta.csv',   # TODO
        motion_feature_dir="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.0/new_joint_vecs",
        text_augment_dir=None,
        augment_text=False,
        aug_text_ind_range=None,
        aug_text_prob=1.0,
        use_natural_desc=True,
        use_short_desc=True,
        use_technical_desc=False,
        split_file_pattern="assets/bones/splits/v1/%s_index.npy",
        stats_folder="assets/bones/stats/v1",
        normalize_motion=True,
        skip_idx_file=None,
        name=None,  # for compatibility with other datasets
    ):
        self.meta = pd.read_csv(meta_file)
        self.split = split
        self.normalize_motion = normalize_motion
        if split == "all":
            self.index = np.arange(len(self.meta))
        else:
            self.index = np.load(split_file_pattern % split)
        # remove data indices with artifacts
        self.skip_idx = (
            set(np.load(skip_idx_file).tolist()) if skip_idx_file is not None else set()
        )
        self.index = np.array([i for i in self.index if i not in self.skip_idx])

        if self.normalize_motion:
            self.mean = np.load(os.path.join(stats_folder, "mean.npy"))
            self.std = np.load(os.path.join(stats_folder, "std.npy"))
        self.num_frames = num_frames
        self.motion_feature_dir = motion_feature_dir
        self.text_augment_dir = text_augment_dir
        self.augment_text = augment_text
        self.aug_text_ind_range = aug_text_ind_range
        self.aug_text_prob = aug_text_prob
        self.use_natural_desc = use_natural_desc
        self.use_short_desc = use_short_desc
        self.use_technical_desc = use_technical_desc
        assert self.use_natural_desc or self.use_short_desc or self.use_technical_desc
        self.motion_paths = self.meta["feature_path"].values
        if self.use_natural_desc:
            self.natural_desc = [
                self.meta[f"natural_desc_{i}"].values for i in range(1, 4)
            ]
        if self.use_short_desc:
            self.short_desc = self.meta[f"short_description"].values
        if self.use_technical_desc:
            self.technical_desc = self.meta[f"technical_description"].values
        return

    def __len__(self):
        return len(self.index)

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def __getitem__(self, idx):
        item = self.index[idx]
        motion_path = os.path.join(self.motion_feature_dir, self.motion_paths[item])
        motion = np.load(motion_path)
        if self.normalize_motion:
            motion = self.normalize(motion)

        text_list = []
        if self.use_natural_desc:
            text_list += [(self.natural_desc[i][item], i) for i in range(3)]
        if self.use_technical_desc:
            text_list.append((self.technical_desc[item], 3))
        if self.use_short_desc:
            text_list.append((self.short_desc[item], 4))

        text, text_sub_ind = text_list[np.random.choice(len(text_list))]
        if type(text) not in [str, np.str_]:
            text = ""
        if (
            self.augment_text
            and text_sub_ind != 3
            and text != ""
            and np.random.rand() < self.aug_text_prob
        ):  # don't have aug for technical descriptions
            try:
                text_augment_path = os.path.join(
                    self.text_augment_dir, f"{item:06d}-{text_sub_ind}.txt"
                )
                if os.path.exists(text_augment_path):
                    aug_texts = read_text_augment_file(text_augment_path)
                    if self.aug_text_ind_range is not None:
                        aug_texts = aug_texts[
                            self.aug_text_ind_range[0] : min(
                                self.aug_text_ind_range[1], len(aug_texts)
                            )
                        ]
                    text = np.random.choice(
                        aug_texts
                    )  # augmented files also include the original text annotation
                    if text[-1] == "." and np.random.rand() < 0.5:
                        # random drop of period at the end
                        text = text[:-1]
                    if np.random.rand() < 0.5:
                        # randomly remove capitalization
                        text = text.lower()
            except Exception as e:
                print(f"Error in text augmentation: {e}")
                print(
                    f"item: {item}, text_sub_ind: {text_sub_ind}, {text_augment_path}"
                )
                if wandb_run_exists():
                    wandb.alert(
                        title=f"[{item}-{text_sub_ind}]",
                        text=f"[{item}-{text_sub_ind}] {text_augment_path}\n" + str(e),
                        level=wandb.AlertLevel.ERROR,
                    )
                # raise

        if self.num_frames == -1:  # no truncation or padding
            m_length = motion.shape[0]
            return text, motion, m_length

        if motion.shape[0] >= self.num_frames:
            m_length = self.num_frames
            idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[idx : idx + self.num_frames]
        else:
            m_length = motion.shape[0]
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.num_frames - motion.shape[0], motion.shape[1])),
                ],
                axis=0,
            )

        return text, motion, m_length


class BonesDatasetMix(data.Dataset):
    """
    The Bones dataset with one or more extra generated datasets mixed in with the specified ratio.
    Note: due to sample_ratio, data will always be randomly returned from this dataset whether or not
            shuffle is true. This should not be used for test-time.
    """

    def __init__(self, mix_paths, sample_ratio, **bones_kwargs):
        # load bones data
        self.datasets = [BonesDataset(**bones_kwargs)]
        # load other datasets to mix in
        for mix_base_path in mix_paths:
            mix_config = copy(bones_kwargs)
            mix_config["meta_file"] = os.path.join(mix_base_path, "meta.csv")
            mix_config["motion_feature_dir"] = os.path.join(mix_base_path, "joint_vecs")
            mix_config["augment_text"] = False
            mix_config["split"] = "all"
            mix_config["skip_idx_file"] = None
            mix_config["use_short_desc"] = (
                True  # only use the text that was the input to the model
            )
            mix_config["use_natural_desc"] = False
            mix_config["use_technical_desc"] = False

            mix_dataset = BonesDataset(**mix_config)
            self.datasets.append(mix_dataset)

        self.dataset_lens = [len(cur_data) for cur_data in self.datasets]
        self.len = sum(self.dataset_lens)
        self.sample_ratio = np.array(sample_ratio) / sum(sample_ratio)
        assert len(self.sample_ratio) == (len(mix_paths) + 1)
        self.sample_thresh = np.cumsum(self.sample_ratio)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # choose which dataset to return from based on sample_ratio, then sample random datapoint
        ratio = float(idx) / self.len
        dataset_idx = np.argmax(
            ratio < self.sample_thresh
        )  # returns first index with 1.0
        samp_idx = np.random.randint(
            0, self.dataset_lens[dataset_idx]
        )  # randomly sample from corresponding dataset
        return self.datasets[dataset_idx].__getitem__(samp_idx)
