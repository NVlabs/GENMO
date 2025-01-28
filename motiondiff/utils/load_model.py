import torch

from motiondiff.utils.config import create_config
from motiondiff.utils.tools import (
    find_last_version,
    get_checkpoint_path,
    import_type_from_str,
    load_ema_weights_from_checkpoint,
)


def get_motiondiff_model(cfg, args, cp="0500000", version=3, test=False):
    cfg = create_config(cfg, training=False, ngpus=args.ngpus, nodes=args.nodes)

    guidance = 2.5
    disable_ema = False
    info_dict = {"html_keys": {}}

    if not test:
        timesteps = "1000"
        cfg.model.diffusion.test_timestep_respacing = timesteps
    cfg.model.diffusion.guidance_param = guidance

    model_cls = import_type_from_str(cfg.model.type)

    version = find_last_version(cfg.cfg_dir) if version is None else version
    info_dict["version"] = version
    checkpoint_dir = f"{cfg.cfg_dir}/version_{version}/checkpoints"
    model_cp = get_checkpoint_path(checkpoint_dir, cp)

    print(f"loading checkpoint {model_cp}")
    if disable_ema or not cfg.train.get("use_ema", False) or test:
        print(f"loading model without EMA from {checkpoint_dir}")
        model = model_cls.load_from_checkpoint(
            model_cp, cfg=cfg, is_inference=True, preload_checkpoint=False, strict=False
        )
        checkpoint = None
    else:
        print(f"loading model with EMA from {checkpoint_dir}")
        model = model_cls(cfg, is_inference=True, preload_checkpoint=False)
        checkpoint = torch.load(model_cp, map_location="cpu")
        load_ema_weights_from_checkpoint(model, checkpoint)

    model.eval()

    return model, cfg
