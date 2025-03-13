import os
import sys

import torch

sys.path.append("./")

# from o3d_materials import Settings
import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from einops import einsum, rearrange
from tqdm import tqdm

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer
from hmr4d.utils.vis.o3d_render import Settings, create_meshes, get_ground
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


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


if __name__ == "__main__":
    # data = torch.load("tmp.pth")
    file_name = "out/motions/motion1-2"
    global_video_path = f"out/demo/motion1-2.mp4"
    # file_name = "out/motions-1/motion1-1"
    # global_video_path = f"out/demo/motion1-1.mp4"

    smplx = make_smplx("supermotion").to("cuda")
    smpl_model = {
        "male": make_smplx("smpl", gender="male"),
        "female": make_smplx("smpl", gender="female"),
    }
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to("cuda")
    faces_smpl = smpl_model["male"].faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(
        "cuda"
    )
    data = torch.load(f"{file_name}.pt")

    pred_smpl_params_global = data["pred_smpl_params_global"]
    time_pred_ay_smpl_out_list = smplx(**pred_smpl_params_global)
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in time_pred_ay_smpl_out_list.vertices]
    )

    verts_glob_list = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob_list = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")
    length = verts_glob_list.shape[0]

    # render_length = min(length, 800)
    render_length = length
    verts_glob_list = verts_glob_list[:render_length]  # (T, V, 3)
    joints_glob_list = joints_glob_list[:render_length]  # (T, J, 3)

    device = verts_glob_list.device
    # global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    writer = get_writer(global_video_path, fps=30, crf=CRF)

    # length, width, height = get_video_lwh(global_video_path)
    width, height = 640 * 3, 480 * 3
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[:, 0], verts_glob_list
    )
    ground_geometry = get_ground(scale * 5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8

    T, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.cpu().clone(),
        beta=2.5,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    # colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    # colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    # colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)
    colors[:, 0] = color_purple[0]
    colors[:, 1] = color_purple[1]
    colors[:, 2] = color_purple[2]

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = torch.from_numpy(faces_smpl.astype("int")).to(device)

    # colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    camera = renderer.scene.camera
    camera.look_at(target[:, None], position[:, None], up[:, None])

    gv, gf, gc = ground_geometry
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    renderer.scene.add_geometry(
        "mesh_ground", ground_mesh, o3d.visualization.rendering.MaterialRecord()
    )
    # faces_list = list(torch.unbind(faces, dim=0))  # + [gf]

    T_c2w = camera.get_view_matrix()
    R_w2c = T_c2w[:3, :3].T
    for t, verts in tqdm(enumerate(verts_glob_list)):
        verts = verts_glob_list[t]  # + [gv]
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.base_color = [0.9, 0.9, 0.9, 0.3 + t * 0.7 / T]
        mat.shader = Settings.Transparency
        # mat.opacity = (i + 1) / N
        mat.thickness = 1.0
        mat.transmission = 1.0
        mat.absorption_distance = 10
        mat.absorption_color = [0.5, 0.5, 0.5]

        mesh = create_meshes(verts, faces, colors[t])
        if t > 0:
            renderer.scene.remove_geometry(f"mesh_{t - 1}")
        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

        img = renderer.render_to_image()
        # import ipdb; ipdb.set_trace()
        # o3d.io.write_image("out/tmp.png", img)
        # img = cv2.imread("out/tmp.png")
        writer.write_frame(np.array(img))
    writer.close()
    # os.makedirs(
    #     os.path.dirname(f"out/{vis_type}_scene_time/{fname}.png"), exist_ok=True
    # )
    # # cv2.imwrite(f"out/{vis_type}_scene_time/{fname}.png", img)
