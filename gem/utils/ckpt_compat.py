# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict


def remap_legacy_state_dict(state_dict):
    """
    Remap known legacy checkpoint keys to current keyspace.

    Returns:
        (new_state_dict, report)
        report: {"renamed": int, "stripped_model_prefix": bool}
    """
    keys = list(state_dict.keys())
    strip_model_prefix = len(keys) > 0 and all(k.startswith("model.") for k in keys)

    prefix_rules = [
        ("pipeline.denoiser3d.endecoder.", "pipeline.endecoder."),
    ]
    dual_write_denoiser_prefixes = (
        "learned_pos_params",
        "learned_pos_linear.",
        "embed_noisyobs.",
        "cliffcam_embedder.",
        "imgseq_embedder.",
        "cam_angvel_embedder.",
        "cond_exists_embedder.",
    )

    remapped = OrderedDict()
    renamed = 0
    for key, value in state_dict.items():
        new_key = key[6:] if strip_model_prefix else key
        for old_prefix, new_prefix in prefix_rules:
            if new_key.startswith(old_prefix):
                new_key = f"{new_prefix}{new_key[len(old_prefix):]}"
                renamed += 1
                break
        remapped[new_key] = value
        if any(new_key.startswith(prefix) for prefix in dual_write_denoiser_prefixes):
            dual_key = f"pipeline.denoiser3d.denoiser.{new_key}"
            if dual_key not in remapped:
                remapped[dual_key] = value
                renamed += 1

    return remapped, {
        "renamed": renamed,
        "stripped_model_prefix": strip_model_prefix,
    }
