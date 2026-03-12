# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def get_pred_body_params_incam(outputs: dict) -> dict | None:
    return outputs.get("pred_body_params_incam")


def get_pred_body_params_global(outputs: dict) -> dict | None:
    return outputs.get("pred_body_params_global")


def get_intermediate_pred_body_params_global(outputs: dict) -> list[dict] | None:
    return outputs.get("intermediate_pred_body_params_global")


def set_pred_body_params_incam(outputs: dict, params: dict) -> None:
    outputs["pred_body_params_incam"] = params


def set_pred_body_params_global(outputs: dict, params: dict) -> None:
    outputs["pred_body_params_global"] = params


def set_intermediate_pred_body_params_global(outputs: dict, params_list: list[dict]) -> None:
    outputs["intermediate_pred_body_params_global"] = params_list
