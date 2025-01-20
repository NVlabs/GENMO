import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from motiondiff.utils.tools import build_object_from_dict


class MultiDataset(Dataset):
    def __init__(self, dataset_kwargs, name, **common_kwargs):
        """
        Initialize the MultiDataset with a dictionary of datasets and their kwargs.
        
        Args:
            dataset_kwargs (dict): A dictionary where keys are dataset classes and values are dictionaries of kwargs.
        """
        self.datasets = []
        for dataset_name, kwargs in dataset_kwargs.items():
            print(kwargs)
            self.datasets.append(build_object_from_dict(kwargs, **common_kwargs))
        
        # Calculate the cumulative lengths of all datasets
        self.lengths = np.array([len(dataset) for dataset in self.datasets])
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Use np.digitize to find the appropriate dataset
        dataset_idx = np.digitize(idx, self.cumulative_lengths)
        if dataset_idx > 0:
            dataset_offset = self.cumulative_lengths[dataset_idx - 1]
            idx -= dataset_offset
        
        # Get the item from the appropriate dataset
        return self.datasets[dataset_idx][idx]

