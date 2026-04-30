import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from o3d_materials import Settings
from tqdm import tqdm


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(faces.cpu().numpy()),
    )
    mesh.compute_vertex_normals()
    if len(colors.shape) == 1:
        colors = colors[None, :].repeat(len(verts), 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    # texture = np.zeros((faces.shape[0], 1, 3), dtype=np.uint8)
    # # for i, color in enumerate(face_colors.cpu().numpy()):
    # #     texture[i, 0, :] = (colors[:3] * 255).astype(np.uint8)
    # texture[:, 0, :] = colors.cpu().numpy()

    # # Assign the texture to the mesh
    # mesh.textures = [o3d.geometry.Image(texture)]

    # # Set triangle_uvs to map each face to the corresponding color
    # uvs = np.zeros((faces.shape[0] * 3, 2), dtype=np.float32)
    # for i in range(faces.shape[0]):
    #     uvs[i * 3 + 0] = [0, 0]
    #     uvs[i * 3 + 1] = [1, 0]
    #     uvs[i * 3 + 2] = [0, 1]
    # mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    return mesh


def compute_look_at(cam_R, cam_T):
    # Convert cam_R and cam_T to numpy arrays if they are not already
    cam_R = cam_R.cpu().numpy() if isinstance(cam_R, torch.Tensor) else cam_R
    cam_T = cam_T.cpu().numpy() if isinstance(cam_T, torch.Tensor) else cam_T

    # Camera position is the translation vector
    camera_position = cam_T

    # Forward vector in camera space
    forward_vector = np.array([0, 0, -1])

    # Transform the forward vector to world space to get the target
    camera_target = cam_R @ forward_vector + camera_position

    # Up vector in camera space
    up_vector = np.array([0, 1, 0])

    # Transform the up vector to world space
    world_up_vector = cam_R @ up_vector

    return camera_position, camera_target, world_up_vector


def get_global_cameras_static(
    verts,
    beta=4.0,
    cam_height_degree=30,
    target_center_height=1.0,
    use_long_axis=False,
    vec_rot=45,
    device="cuda",
):
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(0)

    # a 45 degree vec from longest axis
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4
    # target_scale = max(target_scale, 1.0) * beta
    target_scale = target_scale * beta

    position = target_center + vec * target_scale
    position[1] = (
        target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height
    )

    # Compute camera rotation and translation
    # positions = position.unsqueeze(0).repeat(L, 1)
    # target_centers = target_center.unsqueeze(0).repeat(L, 1)
    target_center[1] = target_center_height
    # rotation = look_at_rotation(positions, target_centers).mT
    # translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    # lights = PointLights(device=device, location=[position.tolist()])
    # return rotation, translation, lights
    up = torch.tensor([0, 1, 0])
    return position, target_center, up


if __name__ == "__main__":
    data = torch.load("tmp.pth")

    verts_list = data["verts_list"]
    colors = data["colors"]
    cameras = data["cameras"]
    lights = data["lights"]
    faces = data["faces"]
    ground_geometry = data["ground_geometry"]
    cam_R = cameras.R
    cam_T = cameras.T
    gv, gf, gc = ground_geometry

    with open(os.path.join(os.path.dirname(__file__), "smpl_key.json"), "r") as f:
        data_format = json.load(f)

    POSE_COLOR = {}
    for i, key in enumerate(data_format["color_order"]):
        if len(data_format["color_order"][key]) > 0:
            POSE_COLOR[key] = np.array(data_format["color_order"][key]) / 255
        else:
            POSE_COLOR[key] = plt.get_cmap(data_format["color_map"])((i * 2 + 1) % 20)[
                :3
            ]

    # colors = np.stack([np.array(color) for color in list(POSE_COLOR.values())]).astype(np.float32)
    # colors = torch.from_numpy(colors).to(verts_list[0].device)
    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(
        verts_list[0].device
    )
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(
        verts_list[0].device
    )
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(
        verts_list[0].device
    )

    mat_settings = Settings()
    # Set up OffscreenRenderer
    width, height = 640 * 4, 480 * 4  # Set the desired resolution
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    position, target, up = get_global_cameras_static(
        # verts_list[0].cpu(),
        verts_list[:, 0].cpu().clone(),
        beta=1.5,
        cam_height_degree=20,
        target_center_height=1.0,
    )
    # Set camera parameters
    camera = renderer.scene.camera
    # cam_R = cam_R[0]
    # cam_T = cam_T[0]
    # cam_T = -cam_R.T @ cam_T
    # # cam_R = cam_R.T
    # camera_position, camera_target, world_up_vector = compute_look_at(
    #     cam_R, cam_T
    # )
    camera.look_at(target[:, None], position[:, None], up[:, None])
    # camera.look_at(position[:, None], target[:, None], up[:, None])

    mesh = create_meshes(gv, gf, gc[..., :3])
    renderer.scene.add_geometry(
        "mesh_ground", mesh, o3d.visualization.rendering.MaterialRecord()
    )
    # mat_box = o3d.visualization.rendering.MaterialRecord()
    # mat_box.base_color = [0.9, 0.9, 0.9, 1.0]
    # mat_box.shader = "defaultLit"
    trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]
    # mat_box = trans_mat_box
    # mat_box.shader = "defaultLitSSR"
    # mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    # mat_box.base_roughness = 0.0
    # mat_box.base_reflectance = 0.0
    # mat_box.base_clearcoat = 1.0
    # mat_box.thickness = 1.0
    # mat_box.transmission = 1.0
    # mat_box.absorption_distance = 10
    # mat_box.absorption_color = [0.5, 0.5, 0.5]
    # visualize colors
    # for i, color in enumerate(colors):
    #     mesh = create_meshes(verts_list[100, 0], faces[0], color)
    #     renderer.scene.add_geometry(f"mesh_color_{i}", mesh, lit_mat_box)
    #     key = list(POSE_COLOR.keys())[i]
    #     print(key, color)
    #     image = renderer.render_to_image()
    #     o3d.io.write_image(f"out/mesh_{key}.png", image)
    # interpolate colors from green to purple
    colors = color_purple[None, :].repeat(len(verts_list), 1)
    colors[:, 0] = torch.linspace(color_green[0], color_purple[0], len(verts_list))
    colors[:, 1] = torch.linspace(color_green[1], color_purple[1], len(verts_list))
    colors[:, 2] = torch.linspace(color_green[2], color_purple[2], len(verts_list))

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(
        color_green[0], color_light_purple[0], len(verts_list)
    )
    colors_trans[:, 1] = torch.linspace(
        color_green[1], color_light_purple[1], len(verts_list)
    )
    colors_trans[:, 2] = torch.linspace(
        color_green[2], color_light_purple[2], len(verts_list)
    )

    for t, verts in tqdm(enumerate(verts_list)):
        if t % 20 != 0:
            continue
        N, V, _ = verts.shape
        verts = list(torch.unbind(verts_list[t], dim=0))  # + [gv]
        faces_list = list(torch.unbind(faces, dim=0))  # + [gf]
        # colors_list = list(torch.unbind(colors, dim=0))# + [gc[..., :3]]
        for i in range(N):
            mesh = create_meshes(
                verts[i], faces_list[i], colors[t] if i == N - 1 else colors_trans[t]
            )
            renderer.scene.add_geometry(
                f"mesh_{i}_{t}", mesh, lit_mat_box if i == N - 1 else trans_mat_box
            )

        # camera.look_at([0, 0, 0], [0, 0, 1], [0, -1, 0])
        # camera.set_projection(60.0, width / height, 0.1, 1000.0)

        # o3d.io.write_triangle_mesh(f"out/mesh_{t}_{i}.obj", mesh)
    image = renderer.render_to_image()
    o3d.io.write_image(f"out/mesh_{t}.png", image)
    import ipdb

    ipdb.set_trace()
