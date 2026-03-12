# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from einops import einsum

from gem.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay

CRF = 23  # 17 is lossless, every +6 halves the mp4 size
color_sequences = [
    "Yellow",
    "Green",
    "Teal",
    "Red",
    "Blue",
    "Purple",
    "Orange",
    "Pink",
    "Brown",
    "Gray",
    "Black",
    "White",
]
color_rgb = (
    np.array(
        [
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [0, 0, 255],
            [255, 0, 255],
            [255, 165, 0],
            [255, 20, 147],
            [165, 42, 42],
            [169, 169, 169],
            [0, 0, 0],
            [255, 255, 255],
        ]
    )
    / 255.0
)


def move_to_start_point_face_z(verts, J_regressor):
    "XZ to origin, Start from the ground, Face-Z"
    # position
    verts = verts.clone()  # (L, V, 3)
    offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
    offset[1] = verts[:, :, [1]].min()
    verts = verts - offset
    # face direction
    T_ay2ayfz = compute_T_ayfz2ay(
        einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True
    )
    verts = apply_T_on_points(verts, T_ay2ayfz)
    return verts


def convert_image_to_mesh(img, offset, R_c2w):
    import open3d as o3d

    img = np.asarray(img)

    # Instead of backprojecting, just convert img to an actual 3D plane with Z=0

    # Create 3D vertex for each pixel location
    xvalues = np.arange(img.shape[1])
    yvalues = np.arange(img.shape[0])[::-1].copy()
    x_loc, y_loc = np.meshgrid(xvalues, yvalues)
    z_loc = np.zeros_like(x_loc)

    # Scale down before making 3D vertices
    x_loc = x_loc / xvalues.shape[0] * 1.5
    y_loc = (
        y_loc / xvalues.shape[0] * 1.5
    )  # Keep aspect ratio same by dividing with same denominator. Now image width is 1 meter in 3d.

    vertices = np.stack((x_loc, y_loc, z_loc), axis=2).reshape(-1, 3)
    vertices = np.matmul(R_c2w, vertices.T).T
    vertices = vertices + offset[None]

    vertex_colors = img.reshape(-1, 3) / 255.0

    # Create triangles between each pair of neighboring vertices
    # Connect positions (i,j), (i+1,j) and (i,j+1) to make one triangle and (i, j+1), (i+1,j) and (i+1,j+1) to make
    # another triangle.
    # Pixel (i,j) is in vertices array at location i + j*xvalues.shape[0]

    vertex_positions = np.arange(xvalues.size * yvalues.size)
    # Reshape into 2D grid and discard last row and column
    vertex_positions = vertex_positions.reshape(yvalues.size, xvalues.size)[:-1, :-1].flatten()

    # Now create triangles (keep vertices in anticlockwise order when making triangles)
    top_triangles = np.vstack(
        (vertex_positions + 1, vertex_positions, vertex_positions + xvalues.shape[0])
    ).transpose(1, 0)
    vertex_positions = np.arange(xvalues.size * yvalues.size)
    vertex_positions = vertex_positions.reshape(yvalues.size, xvalues.size)[1:, 1:].flatten()
    bottom_triangles = np.vstack(
        (vertex_positions - 1, vertex_positions, vertex_positions - xvalues.shape[0])
    ).transpose(1, 0)
    triangles = np.vstack((top_triangles, bottom_triangles))

    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    """
    Flip the y and z axis according to opencv to opengl transformation.
    See - https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform
    """
    # mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return mesh
