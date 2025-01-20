"""
    Dataset to load video data for the human/hand pose estimation
"""
import numpy as np
import torch
from torch.utils import data
from motiondiff.data_pipeline.tensors import collate
from motiondiff.utils.tools import wandb_run_exists
import wandb
import pandas as pd
import os
from copy import copy
import random
from torchvision.transforms import transforms
from pycocotools.coco import COCO