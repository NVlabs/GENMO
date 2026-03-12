# FAQ / Troubleshooting

**Hydra config errors** — Ensure you are using the correct experiment config name (`gem_smpl_regression` or `gem_smpl`). Run `python scripts/train.py --help` to see available options.

**Body model** — This repo uses SMPL (body only). For full-body estimation with [SOMA](https://github.com/NVlabs/SOMA-X) including hands and face, see [GEM-X](https://github.com/NVlabs/GEM-X).

**Hardware requirements** — NVIDIA A100 or newer recommended. Requires CUDA 12.4+. Inference uses approximately 16 GB of GPU memory.

**W&B logging** — Disable with `use_wandb=false` appended to any training or evaluation command.

**Multi-GPU training** — Add `pl_trainer.devices=N` for DDP training across N GPUs.

**Missing datasets** — Only evaluation datasets are needed to run `task=test`. Training datasets are only required if you intend to train from scratch.
