import os

import numpy as np
import torch
import wandb
from einops import einsum, rearrange
import cv2

from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from motiondiff.utils.tools import wandb_run_exists
from motiondiff.utils.vis_scenepic import ScenepicVisualizer

sp_visualizer = ScenepicVisualizer("inputs/smpl_data", device='cuda')

CRF = 23  # 17 is lossless, every +6 halves the mp4 size
color_sequences = ['Yellow', 'Green', 'Teal', 'Red', 'Blue', 'Purple', 'Orange', 'Pink', 'Brown', 'Gray', 'Black', 'White']
color_rgb = np.array([[255, 255, 0], [0, 255, 0], [0, 255, 255], [255, 0, 0], [0, 0, 255], [255, 0, 255], [255, 165, 0], [255, 20, 147], [165, 42, 42], [169, 169, 169], [0, 0, 0], [255, 255, 255]]) / 255.0


def move_to_start_point_face_z(verts, J_regressor):
    "XZ to origin, Start from the ground, Face-Z"
    # position
    verts = verts.clone()  # (L, V, 3)
    offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
    offset[1] = verts[:, :, [1]].min()
    verts = verts - offset
    # face direction
    T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
    verts = apply_T_on_points(verts, T_ay2ayfz)
    return verts


def visualize_smpl_scene(vis_type, index, vid, j3d, gt_j3d, transform_mode=None):
    if transform_mode == 'global':
        global_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        if gt_j3d is not None:
            gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
    elif transform_mode == 'local':
        global_rot = axis_angle_to_matrix(torch.tensor([-np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        j3d[..., 2] += 0.8
        if gt_j3d is not None:
            gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
            gt_j3d[..., 2] += 0.8
    smpl_seq = {
        'pred': {
            'joints_pos': j3d,
        },
    }
    if gt_j3d is not None:
        smpl_seq['gt'] = {
            'joints_pos': gt_j3d,
        }
    vid_ = vid.replace("/", "_")
    fname = f'{index:03d}-{vid_}'
    if len(fname) > 100:
        fname = fname[:100]
    html_file = f"out/{vis_type}/{fname}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)
    
    if wandb_run_exists():
        # pl_module.logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)}, step=pl_module.global_step)
        return {f'{vis_type}/{fname}': wandb.Html(html_file)}
    return {}   


def visualize_intermediate_smpl_scene(vis_type, index, vid, j3d_list, gt_j3d, transform_mode=None):
    if transform_mode == 'global':
        global_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0, 0])).cuda()
        j3d_list = [(global_rot @ j3d.transpose(1, 2)).transpose(1, 2) for j3d in j3d_list]
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
    elif transform_mode == 'local':
        global_rot = axis_angle_to_matrix(torch.tensor([-np.pi / 2, 0, 0])).cuda()
        j3d_list = [(global_rot @ j3d.transpose(1, 2)).transpose(1, 2) for j3d in j3d_list]
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
        for i in range(len(j3d_list)):
            j3d_list[i][..., 2] = j3d_list[i][..., 2] + 0.8
        gt_j3d[..., 2] += 0.8
    smpl_seq = {}
    for i, j3d in enumerate(j3d_list):
        smpl_seq[f'pred_{i}'] = {
            'joints_pos': j3d,
        }

    vid_ = vid.replace("/", "_")
    fname = f'{index:03d}-{vid_}'
    html_file = f"out/{vis_type}/{fname}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)
    
    if wandb_run_exists():
        # pl_module.logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)}, step=pl_module.global_step)
        return {f'{vis_type}/{fname}': wandb.Html(html_file)}
    return {}


def visualize_smplmesh_scene(
    vis_type, index, vid, pred_ay_verts, gt_ay_verts, J_regressor, faces_smpl
):

    verts_glob = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    length = gt_ay_verts.shape[0]
    render_length = min(length, 200)
    verts_glob = verts_glob[:render_length]
    joints_glob = joints_glob[:render_length]
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    # length, width, height = get_video_lwh(global_video_path)
    width, height = 512, 512
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in range(render_length):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]], color[None], cameras, global_lights
        )
        writer.write_frame(img)
    writer.close()



def visualize_intermediate_smplmesh_scene(
    vis_type, index, vid, pred_ay_verts_list, gt_ay_verts, J_regressor, faces_smpl
):

    verts_glob_list = [move_to_start_point_face_z(pred_ay_verts, J_regressor) for pred_ay_verts in pred_ay_verts_list]
    joints_glob_list = [einsum(J_regressor, verts_glob, "j v, l v i -> l j i") for verts_glob in verts_glob_list]
    length = gt_ay_verts.shape[0]
    render_length = min(length, 200)
    verts_glob_list = [verts_glob[:render_length] for verts_glob in verts_glob_list]
    joints_glob_list = [joints_glob[:render_length] for joints_glob in joints_glob_list]
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob_list[0].cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    # length, width, height = get_video_lwh(global_video_path)
    width, height = 512, 512
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob_list[0][:, 0], verts_glob_list[0])
    renderer.set_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8
    colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in range(render_length):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        verts = torch.cat([verts_glob[[i]] for verts_glob in verts_glob_list], dim=0)
        img = renderer.render_with_ground(
            verts, colors, cameras, global_lights, opacity=float(i/render_length)
        )
        writer.write_frame(img)
    writer.close()


def visualize_intermediate_smplmesh_scene_img(
    vis_type, index, vid, pred_ay_verts_list, gt_ay_verts, J_regressor, faces_smpl
):
    import open3d as o3d
    from hmr4d.utils.vis.o3d_render import get_ground, create_meshes, Settings


    verts_glob_list = [move_to_start_point_face_z(pred_ay_verts, J_regressor) for pred_ay_verts in pred_ay_verts_list]
    joints_glob_list = [einsum(J_regressor, verts_glob, "j v, l v i -> l j i") for verts_glob in verts_glob_list]
    length = verts_glob_list[0].shape[0]
    render_length = min(length, 200)
    verts_glob_list = [verts_glob[:render_length] for verts_glob in verts_glob_list]    # (N, T, V, 3)
    joints_glob_list = [joints_glob[:render_length] for joints_glob in joints_glob_list]    # (N, T, J, 3)

    device = verts_glob_list[0].device

    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    # length, width, height = get_video_lwh(global_video_path)
    width, height = 640 * 4, 480 * 4
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob_list[0][:, 0], verts_glob_list[0])
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8
    colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)

    verts_glob_list = torch.stack(verts_glob_list, dim=0).transpose(1, 0)  # (T, N, V, 3)
    T, N, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.reshape(-1, V, 3).cpu().clone(),
        beta=1.5,
        cam_height_degree=20,
        target_center_height=1.0,
    )
    camera = renderer.scene.camera
    camera.look_at(target[:, None], position[:, None], up[:, None])

    gv, gf, gc = ground_geometry
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    renderer.scene.add_geometry(
        "mesh_ground", ground_mesh, o3d.visualization.rendering.MaterialRecord()
    )

    trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = torch.from_numpy(faces_smpl.astype('int')).unsqueeze(0).to(device).expand(N, -1, -1)

    # colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)

    for i in range(N):
        faces_list = list(torch.unbind(faces, dim=0))  # + [gf]
        for t, verts in enumerate(verts_glob_list):
            if t % 20 != 0:
                continue
            verts = list(torch.unbind(verts_glob_list[t], dim=0))  # + [gv]
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.base_color = [0.9, 0.9, 0.9, (i + 1) / N]
            mat.shader = Settings.Transparency
            # mat.opacity = (i + 1) / N
            mat.thickness = 1.0
            mat.transmission = 1.0
            mat.absorption_distance = 10
            mat.absorption_color = [0.5, 0.5, 0.5]

            # mesh = create_meshes(verts[i], faces_list[i], colors[t] if i == 0 else colors_trans[t])
            mesh = create_meshes(verts[i], faces_list[i], colors[t])
            renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)
            # renderer.scene.add_geometry(
            #     f"mesh_{i}_{t}", mesh, lit_mat_box if i == 0 else trans_mat_box
            # )

    img = renderer.render_to_image()
    os.makedirs(os.path.dirname(f"out/{vis_type}_time/{fname}.png"), exist_ok=True)
    # cv2.imwrite(f"out/{vis_type}_time/{fname}.png", img)
    o3d.io.write_image(f"out/{vis_type}_time/{fname}.png", img)
