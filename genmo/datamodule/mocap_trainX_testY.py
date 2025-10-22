import resource
from functools import partial

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from numpy.random import choice
from omegaconf import DictConfig
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Subset, default_collate

from genmo.utils.pylogger import Log

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate_fn(batch, mode, collate_cfg=None):
    """Handle meta and Add batch size to the return dict
    Args:
        batch: list of dict, each dict is a data point
        collate_cfg: configuration for collation
    """
    # Assume all keys in the batch are the same
    return_dict = {"B": len(batch)}

    length = collate_cfg.max_motion_frames
    if mode in ["val", "test"] and "K_fullimg" in batch[0]:
        length = batch[0]["K_fullimg"].shape[0]

    # Get a superset of all keys from all batch items
    mandatory_keys = [
        "has_text",
        # "has_audio",
        # "has_music",
        "caption",
        "text_embed",
        "music_embed",
        "music_array",
        "music_fps",
        "music_beats",
        "audio_array",
        "audio_fps",
        "use_det_kp",
    ]
    keys = set(mandatory_keys)
    for item in batch:
        keys.update(item.keys())
    keys = sorted(keys)

    for k in keys:
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        elif k == "multi_text_embed":
            # Get max length across batch
            max_len = max(d[k].shape[0] for d in batch if k in d)
            padded_tensors = []
            for d in batch:
                if k in d:
                    padded = torch.cat(
                        [
                            d[k],
                            torch.zeros(max_len - d[k].shape[0], *d[k].shape[1:]).to(
                                d[k]
                            ),
                        ],
                        dim=0,
                    )
                    padded_tensors.append(padded)
                else:
                    # Handle case where key is missing in this batch item
                    continue
            if padded_tensors:
                return_dict[k] = default_collate(padded_tensors)
        else:
            vals = []
            for d in batch:
                if k not in d:
                    if k in collate_cfg.default_feature_val:
                        val = collate_cfg.default_feature_val[k]
                    elif k in collate_cfg.default_frame_feature_dim:
                        length_multiplier = (
                            collate_cfg.default_seq_feature_length_multiplier.get(k, 1)
                        )
                        val = torch.zeros(
                            length * length_multiplier,
                            *collate_cfg.default_frame_feature_dim[k],
                            dtype=eval(
                                collate_cfg.default_feature_type.get(k, "torch.float32")
                            ),
                        )
                    elif k in collate_cfg.default_seq_feature_dim:
                        val = torch.zeros(
                            *collate_cfg.default_seq_feature_dim[k],
                            dtype=eval(
                                collate_cfg.default_feature_type.get(k, "torch.float32")
                            ),
                        )
                    else:
                        raise ValueError(f"Key {k} not found in collate_cfg")
                    vals.append(val)
                else:
                    vals.append(d[k])
            return_dict[k] = default_collate(vals)

    return return_dict


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_opts: DictConfig,
        loader_opts: DictConfig,
        limit_each_trainset=None,
        train_subset_ratio=None,
        train_2d_only=False,
        collate_cfg: DictConfig = None,
    ):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset
        self.train_subset_ratio = train_subset_ratio
        self.train_2d_only = train_2d_only
        self.collate_cfg = collate_cfg
        # Train uses concat dataset
        if "train" in dataset_opts:
            assert "train" in self.loader_opts, "train not in loader_opts"
            split_opts = dataset_opts.get("train")
            assert isinstance(split_opts, DictConfig), (
                "split_opts should be a dict for each dataset"
            )
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                if self.limit_each_trainset:
                    dataset_i = Subset(
                        dataset_i, choice(len(dataset_i), self.limit_each_trainset)
                    )
                if self.train_subset_ratio is not None:
                    dataset_i = Subset(
                        dataset_i,
                        choice(
                            len(dataset_i),
                            int(len(dataset_i) * self.train_subset_ratio),
                        ),
                    )
                dataset.append(dataset_i)
                Log.info(
                    f"[Train Dataset][{idx + 1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}"
                )
            dataset = ConcatDataset(dataset)
            self.trainset = dataset
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(dataset)}")
            Log.info("")

        # Val and Test use sequential dataset
        for split in ("val", "test"):
            if split not in dataset_opts:
                continue
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = dataset_opts.get(split)
            assert isinstance(split_opts, DictConfig), (
                "split_opts should be a dict for each dataset"
            )
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset.append(instantiate(v))
                dataset_type = "Val Dataset" if split == "val" else "Test Dataset"
                Log.info(
                    f"[{dataset_type}][{idx + 1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}"
                )
            setattr(self, f"{split}sets", dataset)
            Log.info("")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=partial(
                    collate_fn, mode="train", collate_cfg=self.collate_cfg
                ),
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True
                        and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=partial(
                            collate_fn, mode="val", collate_cfg=self.collate_cfg
                        ),
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=partial(
                            collate_fn, mode="test", collate_cfg=self.collate_cfg
                        ),
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()
