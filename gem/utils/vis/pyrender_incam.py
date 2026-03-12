# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal pyrender-based mesh-on-image compositor.

Minimal pyrender-based mesh renderer for overlaying SMPL meshes on images.
  Renderer(focal_length, faces).__call__(vertices, cam_t, image, camera_center=None)
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # must precede first pyrender import

import numpy as np
import pyrender
import trimesh


def _create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


class Renderer:
    def __init__(self, focal_length, faces=None):
        self.focal_length = focal_length
        self.faces = faces

    def __call__(
        self,
        vertices: np.ndarray,  # (V, 3)
        cam_t: np.ndarray,  # (3,)
        image: np.ndarray,  # (H, W, 3) uint8
        mesh_base_color=(1.0, 1.0, 0.9),
        camera_center=None,
    ) -> np.ndarray:  # (H, W, 3) uint8
        image_f = image.astype(np.float32) / 255.0
        h, w = image_f.shape[:2]
        if camera_center is None:
            camera_center = [w / 2.0, h / 2.0]

        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(mesh_base_color[2], mesh_base_color[1], mesh_base_color[0], 1.0),
        )
        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)
        for node in _create_raymond_lights():
            scene.add_node(node)

        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()

        color_f = color.astype(np.float32) / 255.0
        if color_f.shape[2] == 4:
            alpha = color_f[:, :, 3:4]
            out = color_f[:, :, :3] * alpha + (1.0 - alpha) * image_f
        else:
            # RGBA not supported by this backend; composite via depth mask
            mask = (depth > 0)[..., np.newaxis]
            out = np.where(mask, color_f, image_f)
        return (out * 255).clip(0, 255).astype(np.uint8)
