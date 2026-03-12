# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from gem.utils.vis.o3d_render import Settings, create_meshes


def simple_render_mesh(render_dict):
    """Render a camera-space mesh, blank background."""
    width, height, focal_length = render_dict["whf"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]

    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.8, 0.8, 0.8])

    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int32)).long()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    renderer.scene.camera.set_projection(
        K.double().numpy(), 0.01, 100.0, float(width), float(height)
    )
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    outputs = []
    for i in tqdm(range(len(verts)), desc="Rendering"):
        mesh = create_meshes(verts[i], faces, color)
        mesh_name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())  # (H, W, 3) uint8
        outputs.append(rendered)
    outputs = np.stack(outputs, axis=0)
    return outputs


def simple_render_mesh_background(render_dict, VI=50, colors=None):
    """Render a camera-space mesh onto background frames."""
    if colors is None:
        colors = [0.8, 0.8, 0.8]
    K = render_dict["K"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]
    background = render_dict["background"]
    N_frames = len(verts)
    if len(background.shape) == 3:
        background = [background] * N_frames
    height, width = background[0].shape[:2]

    if isinstance(K, torch.Tensor):
        K_np = K.cpu().double().numpy().reshape(3, 3)
    else:
        K_np = np.array(K, dtype=np.float64).reshape(3, 3)

    color_tensor = torch.tensor(colors) if not isinstance(colors, torch.Tensor) else colors
    if color_tensor.max() > 1:
        color_tensor = color_tensor.float() / 255.0

    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int32)).long()

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    renderer.scene.camera.set_projection(K_np, 0.01, 100.0, float(width), float(height))
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    outputs = []
    for i in tqdm(range(len(verts)), desc="Rendering"):
        mesh = create_meshes(verts[i], faces, color_tensor)
        mesh_name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())  # (H, W, 3) uint8
        depth = np.asarray(renderer.render_to_depth_image())  # (H, W) float32
        mask = (depth < 1.0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.0)
        alpha = mask[..., np.newaxis]
        bg = background[i] if isinstance(background, list) else background
        composite = rendered.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
        outputs.append(composite.clip(0, 255).astype(np.uint8))
    outputs = np.stack(outputs, axis=0)
    return outputs
