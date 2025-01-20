from torch.utils.data import DataLoader
from motiondiff.data_pipeline.tensors import collate as all_collate
from motiondiff.data_pipeline.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from motiondiff.data_pipeline.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from motiondiff.data_pipeline.humanml.data.dataset import KIT
        return KIT
    elif name == 'dpo':
        from motiondiff.data_pipeline.datasets.dpo_dataset import DPODataset
        return DPODataset
    elif name == 'bones':
        from motiondiff.data_pipeline.datasets.bones_dataset import BonesDataset
        return BonesDataset
    elif name == 'bones2d':
        from motiondiff.data_pipeline.datasets.bones_2d_dataset import Bones2DDataset
        return Bones2DDataset
    elif name == 'bones':
        from motiondiff.data_pipeline.datasets.bones_dataset import BonesDataset
        return BonesDataset
    elif name == 'bonesmix':
        from motiondiff.data_pipeline.datasets.bones_dataset import BonesDatasetMix
        return BonesDatasetMix
    elif name == 'multi':
        from motiondiff.data_pipeline.datasets.multi_dataset import MultiDataset
        return MultiDataset
    elif name == 'hand3d':
        from motiondiff.data_pipeline.datasets.hand_dataset import HandDataset 
        # from motiondiff.data_pipeline.datasets.hand_dataset import LocoHandDataset as Dataset
        return HandDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from motiondiff.data_pipeline.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    elif name == 'dpo':
        from motiondiff.data_pipeline.datasets.dpo_dataset import dpo_collate
        return dpo_collate
    elif name in ["humanml", "kit"]:
        return t2m_collate
    elif name in {'bones', 'bonesmix'}:
        from motiondiff.data_pipeline.datasets.bones_dataset import bones_collate
        return bones_collate
    elif name in {'bones2d'}:
        from motiondiff.data_pipeline.datasets.bones_2d_dataset import bones_collate
        return bones_collate
    elif name == 'kp2d':
        from motiondiff.data_pipeline.datasets.kp2d_dataset import kp2d_collate
        return kp2d_collate
    elif name == 'hand3d':
        from motiondiff.data_pipeline.datasets.hand_dataset import hand_collate 
        return hand_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', debug=False, rng=None, data_cfg=None):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, debug=debug, rng=rng)
    elif name == 'dpo':
        dataset = DATA(split=split, num_frames=num_frames, win_key=data_cfg.win_key, lose_key=data_cfg.lose_key)
    elif name in {'bones', 'bonesmix', 'bones2d'}:
        dataset = DATA(**data_cfg)
    elif name in {'hand3d'}:
        dataset = DATA(**data_cfg)
    elif name == 'multi':
        dataset = DATA(**data_cfg)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', debug=False, shuffle=False, drop_last=True, data_cfg=None, num_workers=8):
    dataset = get_dataset(name, num_frames, split, hml_mode, debug, data_cfg=data_cfg)
    
    collate_type = data_cfg.get('collate_type', name)
    collate = get_collate_fn(collate_type, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train') or shuffle,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    )
    print("#######################################", len(loader))
    print(name, num_frames, split, hml_mode)

    return loader