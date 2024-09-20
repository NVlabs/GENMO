import yaml
import os
import glob
import numpy as np
from omegaconf import OmegaConf


# TODO: move to config
SMPL_DATA_PATH = "./body_models/smpl"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10


def create_config(cfg_name, tmp=False, training=True, ngpus=1, nodes=1, results_root_dir=None):
    cfg_path = 'motiondiff/cfg/**/%s.yml' % cfg_name
    files = glob.glob(cfg_path, recursive=True)
    assert(len(files) == 1)
    cfg = OmegaConf.load(files[0])

    cfg_id = f'{cfg_name}_gpu{ngpus}_node{nodes}'
    cfg['name'] = cfg_name
    cfg['id'] = cfg_id
    cfg['training'] = training

    # default values
    if results_root_dir is not None:
        cfg['results_root_dir'] = results_root_dir
    cfg['results_root_dir'] = os.path.expanduser(cfg['results_root_dir'])
    cfg_root_dir = '/tmp/motiondiff' if tmp else cfg['results_root_dir']
    cfg_root_dir = os.path.expanduser(cfg_root_dir)
    cfg['cfg_dir'] = f'{cfg_root_dir}/{cfg_id}'
    os.makedirs(cfg['cfg_dir'], exist_ok=True)
    return cfg

def create_guide_config(cfg_id, guide_cfg_id, ui=False):
    cfg_dir = 'guidance_ui' if ui else 'guidance'
    cfg_path = 'motiondiff/cfg/%s/%s/**/%s.yml' % (cfg_dir, cfg_id, guide_cfg_id)
    files = glob.glob(cfg_path, recursive=True)
    assert len(files) == 1, f'{len(files)} files were matched to the given config! Must be exactly 1.'
    cfg = OmegaConf.load(files[0])

    return cfg, files[0]