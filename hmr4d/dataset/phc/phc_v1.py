import glob
from pathlib import Path
from time import time

import numpy as np
import torch

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.pure_motion.utils import *
from hmr4d.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from hmr4d.utils.pylogger import Log


class PHCDataset(ImgfeatMotionDatasetBase):
    """Dataset for loading PHC dumped data."""

    def __init__(
        self,
        data_path,  # Path to the directory containing episode files
        motion_frames=120,
        num_repeat=1,
    ):
        self.data_path = Path(data_path)
        self.motion_frames = motion_frames
        self.num_repeat = num_repeat
        self.rng = np.random
        super().__init__()

    def reload_data(self):
        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        Log.info(f"[PHC] Loading metadata from {self.data_path}")
        tic = time()

        # Get all episode files
        self.episode_files = sorted(glob.glob(str(self.data_path / "*.pth")))
        # Filter out metadata.p file
        self.episode_files = [f for f in self.episode_files if "metadata" not in f]

        if len(self.episode_files) == 0:
            raise ValueError(f"No episode files found in {self.data_path}")

        # Load metadata if available
        self.metadata = {}
        metadata_path = self.data_path / "metadata_0.p"
        if metadata_path.exists():
            self.metadata = torch.load(metadata_path)

        # Load first episode to get data structure
        self.sample_data = torch.load(self.episode_files[0])

        Log.info(
            f"[PHC] Found {len(self.episode_files)} episodes. Elapsed: {time() - tic:.2f}s. Metadata: {self.metadata}"
        )

    def _get_idx2meta(self):
        # Each episode file becomes an index
        self.idx2meta = list(range(len(self.episode_files))) * self.num_repeat

        # We'll check validity when loading individual episodes
        Log.info(f"[PHC] {len(self.idx2meta)} sequences.")

    def _load_data(self, idx):
        # Load the episode data from the corresponding file
        eid = self.idx2meta[idx]
        episode_data = torch.load(
            self.episode_files[eid], map_location="cpu", weights_only=False
        )

        # Get sequence length
        raw_length = episode_data["self_obs"].shape[0]
        tgt_len = self.motion_frames

        if tgt_len <= raw_length:
            start = self.rng.randint(0, raw_length - tgt_len + 1)
            end = start + tgt_len
        else:  # interpolation will use all possible frames
            start = 0
            end = raw_length

        length = end - start

        data = {
            "self_obs": episode_data["self_obs"][start:end],
            "rgb_obs": episode_data["rgb_obs"][start:end],
            "contact_forces_obs": episode_data["contact_forces_obs"][start:end],
            "clean_actions": episode_data["clean_actions"][start:end],
            "env_actions": episode_data["env_actions"][start:end],
            "z_actions": episode_data["z_actions"][start:end],
            "died_buf": episode_data["died_buf"][start:end],
            "timedout_buf": episode_data["timedout_buf"][start:end],
        }
        if "text_label_ids" in episode_data:
            data["text_label_ids"] = episode_data["text_label_ids"][start:end]

        if length < tgt_len:
            data = pad_data(data, tgt_len)

        if "text_embed" in episode_data:
            data["text_embed"] = episode_data["text_embed"]
        if "multi_text_embed" in episode_data:
            data["multi_text_embed"] = episode_data["multi_text_embed"]

        data["length"] = length
        return data

    def _process_data(self, data, idx):
        eid = self.idx2meta[idx]
        length = data["length"]
        max_len = self.motion_frames

        return_data = {
            "meta": {
                "data_name": "phc",
                "idx": idx,
                "eid": eid,
                "has_humanoid_data": True,
            },
            "length": length,
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "humanoid": get_valid_mask(max_len, length),
            },
            # Additional PHC-specific data
            "humanoid_obs": data["self_obs"],
            "humanoid_rgb_obs": data["rgb_obs"],
            "humanoid_contact_force": data["contact_forces_obs"],
            "humanoid_clean_action": data["clean_actions"],
            "humanoid_env_action": data["env_actions"],
            "humanoid_z_action": data["z_actions"],
            "humanoid_died_buf": data["died_buf"],
            "humanoid_timedout_buf": data["timedout_buf"],
        }
        if "text_embed" in data:
            return_data["text_embed"] = data["text_embed"]
        if "multi_text_embed" in data:
            return_data["multi_text_embed"] = data["multi_text_embed"]
            return_data["text_label_ids"] = data["text_label_ids"]
            return_data["meta"]["multi_text_label"] = True
        return_data["humanoid_contact_force_exists"] = torch.ones(
            *data["contact_forces_obs"].shape[:1]
        ).bool()
        return_data["humanoid_contact_force_exists"][length:] = False

        return return_data
