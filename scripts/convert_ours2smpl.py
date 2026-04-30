import os
import sys

from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch

from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

encoder = EnDecoder(stats_name="DEFAULT_01", encode_type="humanml3d").cuda()

# data_path = 'outputs/humanml3d_feats_gt/feats_test.pt'
data_path = (
    "outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats.pt"
)
raw_data = torch.load(data_path).cuda()


smpl_data = encoder.decode_humanml3d(raw_data)

save_path = (
    "outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats_smpl.pt"
)
torch.save(smpl_data, save_path)
