import os

import cv2
import numpy as np
import torch
import wandb
from einops import einsum
from moviepy.editor import VideoFileClip

from genmo.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from genmo.utils.rotation_conversions import axis_angle_to_matrix
from genmo.utils.tools import wandb_run_exists
from genmo.utils.video_io_utils import get_writer
from genmo.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)
from genmo.utils.vis.vis_scenepic import ScenepicVisualizer
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import create_camera_sensor

sp_visualizer = ScenepicVisualizer("inputs/checkpoints/body_models/smpl", device="cuda")

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


def visualize_smpl_scene(
    vis_type, index, vid, j3d, gt_j3d, transform_mode=None, keyframes=None
):
    if transform_mode == "global":
        global_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        if gt_j3d is not None:
            gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
    elif transform_mode == "local":
        global_rot = axis_angle_to_matrix(torch.tensor([-np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        j3d[..., 2] += 0.8
        if gt_j3d is not None:
            gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
            gt_j3d[..., 2] += 0.8
    smpl_seq = {
        "pred": {
            "joints_pos": j3d,
        },
    }
    if keyframes is not None:
        smpl_seq["pred"]["keyframe_idx"] = keyframes.tolist()
    if gt_j3d is not None:
        smpl_seq["gt"] = {
            "joints_pos": gt_j3d,
        }
    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    if len(fname) > 100:
        fname = fname[:100]
    html_file = f"out/{vis_type}/{fname}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)

    if wandb_run_exists():
        # pl_module.logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)}, step=pl_module.global_step)
        return {f"{vis_type}/{fname}": wandb.Html(html_file)}
    return {}


def visualize_intermediate_smpl_scene(
    vis_type, index, vid, j3d_list, gt_j3d, transform_mode=None
):
    if transform_mode == "global":
        global_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0, 0])).cuda()
        j3d_list = [
            (global_rot @ j3d.transpose(1, 2)).transpose(1, 2) for j3d in j3d_list
        ]
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
    elif transform_mode == "local":
        global_rot = axis_angle_to_matrix(torch.tensor([-np.pi / 2, 0, 0])).cuda()
        j3d_list = [
            (global_rot @ j3d.transpose(1, 2)).transpose(1, 2) for j3d in j3d_list
        ]
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
        for i in range(len(j3d_list)):
            j3d_list[i][..., 2] = j3d_list[i][..., 2] + 0.8
        gt_j3d[..., 2] += 0.8
    smpl_seq = {}
    for i, j3d in enumerate(j3d_list):
        smpl_seq[f"pred_{i}"] = {
            "joints_pos": j3d,
        }

    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    html_file = f"out/{vis_type}/{fname}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)

    if wandb_run_exists():
        # pl_module.logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)}, step=pl_module.global_step)
        return {f"{vis_type}/{fname}": wandb.Html(html_file)}
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
    verts_glob_list = [
        move_to_start_point_face_z(pred_ay_verts, J_regressor)
        for pred_ay_verts in pred_ay_verts_list
    ]
    joints_glob_list = [
        einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
        for verts_glob in verts_glob_list
    ]
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
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[0][:, 0], verts_glob_list[0]
    )
    renderer.set_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8
    colors = torch.stack(
        [
            torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda()
            for i in range(len(verts_glob_list))
        ],
        dim=0,
    )

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in range(render_length):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        verts = torch.cat([verts_glob[[i]] for verts_glob in verts_glob_list], dim=0)
        img = renderer.render_with_ground(
            verts, colors, cameras, global_lights, opacity=float(i / render_length)
        )
        writer.write_frame(img)
    writer.close()


def visualize_intermediate_smplmesh_scene_img(
    vis_type, index, vid, pred_ay_verts_list, gt_ay_verts, J_regressor, faces_smpl
):
    import open3d as o3d

    from genmo.utils.vis.o3d_render import Settings, create_meshes, get_ground

    verts_glob_list = [
        move_to_start_point_face_z(pred_ay_verts, J_regressor)
        for pred_ay_verts in pred_ay_verts_list
    ]
    joints_glob_list = [
        einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
        for verts_glob in verts_glob_list
    ]
    length = verts_glob_list[0].shape[0]
    render_length = min(length, 200)
    verts_glob_list = [
        verts_glob[:render_length] for verts_glob in verts_glob_list
    ]  # (N, T, V, 3)
    joints_glob_list = [
        joints_glob[:render_length] for joints_glob in joints_glob_list
    ]  # (N, T, J, 3)

    device = verts_glob_list[0].device

    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    # length, width, height = get_video_lwh(global_video_path)
    width, height = 640 * 4, 480 * 4
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    # mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[0][:, 0], verts_glob_list[0]
    )
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8
    colors = torch.stack(
        [
            torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda()
            for i in range(len(verts_glob_list))
        ],
        dim=0,
    )

    verts_glob_list = torch.stack(verts_glob_list, dim=0).transpose(
        1, 0
    )  # (T, N, V, 3)
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

    # trans_mat_box = mat_settings._materials[Settings.Transparency]
    # lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = (
        torch.from_numpy(faces_smpl.astype("int"))
        .unsqueeze(0)
        .to(device)
        .expand(N, -1, -1)
    )

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


def visualize_smpl_scene_mp4(
    vis_type,
    index,
    vid,
    v3d,
    gt_j3d,
    faces_smpl,
    J_regressor,
    transform_mode=None,
    keyframes=None,
    audio_clip=None,
):
    verts_glob = move_to_start_point_face_z(v3d, J_regressor)

    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
    )
    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    if len(fname) > 100:
        fname = fname[:100]
    video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    length = verts_glob.shape[0]
    # length, width, height = get_video_lwh(video_path)
    width, height = 512, 512
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(verts_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(video_path, fps=30, crf=CRF)
    for i in range(length):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]], color[None], cameras, global_lights
        )
        writer.write_frame(img)
    writer.close()

    if audio_clip is not None:
        video_clip = VideoFileClip(video_path)
        video_with_audio = video_clip.set_audio(audio_clip)
        # video_clip.write_videofile(video_path, audio=True)
        video_with_audio.write_videofile(
            video_path.replace(".mp4", "_audio.mp4"), codec="libx264", audio_codec="aac"
        )

        # os.makedirs(os.path.dirname(f"out/{vis_type}_audio/{fname}.mp3"), exist_ok=True)
        # audio_clip.write_audiofile(f"out/{vis_type}_audio/{fname}.mp3")

    if wandb_run_exists():
        # pl_module.logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)}, step=pl_module.global_step)
        return {f"{vis_type}/{fname}": wandb.Video(video_path)}
    return {}


def visualize_smplmesh_scene_img(
    vis_type,
    index,
    vid,
    pred_ay_verts,
    gt_ay_verts,
    J_regressor,
    faces_smpl,
    vid2=None,
    start_ind=300,
    end_ind=550,
):
    import open3d as o3d

    from genmo.utils.vis.o3d_render import Settings, create_meshes, get_ground

    verts_glob_list = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob_list = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")
    length = verts_glob_list.shape[0]

    render_length = min(length, 1600)
    render_length = length
    verts_glob_list = verts_glob_list[:render_length]  # (T, V, 3)
    joints_glob_list = joints_glob_list[:render_length]  # (T, J, 3)

    device = verts_glob_list.device
    if vid2 is None:
        vid2 = vid
    vid_ = vid.replace("/", "_")
    fname = f"{index:03d}-{vid_}"
    global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    # length, width, height = get_video_lwh(global_video_path)
    width, height = 640 * 4, 480 * 4
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens
    # start_ind = 300
    # end_ind =550

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    all_colors = torch.load("colors.pth")
    color_ids = [0, 1, 2, 3, 4, 11]
    import ipdb

    ipdb.set_trace()
    print(all_colors[color_ids], color_purple)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[:, 0], verts_glob_list
    )
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8

    T, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.cpu().clone(),
        beta=1.2,
        cam_height_degree=15,
        target_center_height=1.0,
        vec_rot=-90,
    )
    camera = renderer.scene.camera
    camera.look_at(target[:, None], position[:, None], up[:, None])

    gv, gf, gc = ground_geometry
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.base_color = [0.9, 0.9, 0.9, 1]
    # ground_mat.shader = "defaultUnlit"
    # ground_mesh.paint_uniform_color([1.0, 1.0, 1.0])
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_mat)

    white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # light_position = np.array([10.0, 1.0, 0.0], dtype=np.float32)

    renderer.scene.scene.add_directional_light(
        "light", white_color, np.array([0, -0.5, -1]), 1e5, True
    )
    # renderer.scene.scene.add_point_light("light", white_color, light_position, 1e5, 1e2, True)

    # trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = torch.from_numpy(faces_smpl.astype("int")).to(device)

    # colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)

    # faces_list = list(torch.unbind(faces, dim=0))  # + [gf]
    T_c2w = camera.get_view_matrix()
    R_w2c = T_c2w[:3, :3].T
    for t, verts in enumerate(verts_glob_list):
        # if start_ind <= t <= end_ind and t % 60 != 0:
        #     continue
        # if (t % 30 == 0 and (t < start_ind or t > end_ind)) or (t in [290, 330, 450, 480, 510, 540, 600, 660]):
        if (t % 30 == 0 and (t <= start_ind or t >= end_ind)) or (t % 30 == 0):
            if t >= length - 600 and not (t % 30 == 0):
                continue
            t_text = t + 600 - length
            if 0 <= t_text < 30 or 120 < t_text <= 240:
                continue

            verts = verts_glob_list[t]  # + [gv]
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.base_color = [0.9, 0.9, 0.9, 0.3 + t * 0.7 / T]
            mat.shader = Settings.Transparency
            # mat.opacity = (i + 1) / N
            mat.thickness = 1.0
            mat.transmission = 1.0
            mat.absorption_distance = 10
            mat.absorption_color = [0.5, 0.5, 0.5]

            # mesh = create_meshes(verts, faces, colors[t])
            if t < 240:
                set_color = all_colors[color_ids[0]]
            elif t < 500:
                set_color = all_colors[color_ids[1]]
            elif t < 800:
                set_color = all_colors[color_ids[2]]
            elif t < 1050:
                set_color = all_colors[color_ids[3]]
            elif t < 1200:
                set_color = all_colors[color_ids[4]]
            else:
                set_color = all_colors[color_ids[5]]
                set_color = color_purple.cpu().numpy()

            set_color = torch.from_numpy(set_color).float().to(device)
            mesh = create_meshes(verts, faces, set_color)
            renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)

        # if (t % 120 == 0 and (t < start_ind or t > end_ind)) or (t in [210, 630]):
        if (t % 120 == 0 and (t < start_ind or t > end_ind)) and False:
            if t >= length - 600:
                continue
            if t > end_ind:
                pid = vid2[:2]
                sid = vid2[3:]
            else:
                pid = vid[:2]
                sid = vid[3:]
            img_path = f"/mnt/dhd/body-pose-dataset/EMDB/{pid}/{sid}/images/{t:05d}.jpg"
            img = cv2.imread(img_path)
            offset = verts.mean(dim=0).cpu().numpy()
            offset[1] += 2

            img_h, img_w = img.shape[:2]
            # img = cv2.resize(img, (img_w // 8, img_h // 8))
            img_mesh = convert_image_to_mesh(img[:, :, ::-1], offset, R_w2c)
            # rotate the image mesh to face the camera

            # img_mesh = o3d.geometry.TriangleMesh.create_box(
            #     width=1, height=0.1, depth=1
            # )
            # img_mesh.compute_vertex_normals()
            img_mat = o3d.visualization.rendering.MaterialRecord()
            # img_mat.base_color = [0.9, 0.9, 0.9, 1]
            img_mat.shader = "defaultLit"
            # img_mat.albedo_img = o3d.geometry.Image(img)
            # img_mesh.paint_uniform_color([1, 1, 1])
            # img_mat.texture = o3d.geometry.Image(img)
            renderer.scene.add_geometry(f"img_mesh_{t}", img_mesh, img_mat)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

    img = renderer.render_to_image()
    os.makedirs(
        os.path.dirname(f"out/{vis_type}_scene_time/{fname}.png"), exist_ok=True
    )
    # cv2.imwrite(f"out/{vis_type}_scene_time/{fname}.png", img)
    o3d.io.write_image(f"out/{vis_type}_scene_time/{fname}.png", img)


def visualize_smplmesh_scene_video(
    vis_type,
    index,
    vid,
    pred_ay_verts,
    gt_ay_verts,
    J_regressor,
    faces_smpl,
    vid2=None,
    start_ind=300,
    end_ind=550,
    caption=None,
):
    import open3d as o3d
    from tqdm import tqdm

    from genmo.utils.vis.o3d_render import Settings, create_meshes, get_ground

    verts_glob_list = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob_list = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")
    length = verts_glob_list.shape[0]

    render_length = min(length, 800)
    verts_glob_list = verts_glob_list[:render_length]  # (T, V, 3)
    joints_glob_list = joints_glob_list[:render_length]  # (T, J, 3)

    if vid2 is None:
        vid2 = vid
    device = verts_glob_list.device
    vid_ = vid.replace("/", "_")
    vid2_ = vid2.replace("/", "_")
    fname = f"{index:03d}-{vid_}-{vid2_}"
    if caption is not None:
        fname = (
            f"{fname}-{caption.replace(' ', '_').replace('/', '_').replace(',', '_')}"
        )
        fname = fname[:100]
    # global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    global_video_path = f"out/v1-t-v2/{fname}.mp4"
    if os.path.exists(global_video_path):
        rand_idx = torch.randint(0, 1000000, (1,)).item()
        global_video_path = global_video_path.replace(".mp4", f"_{rand_idx}.mp4")
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    writer = get_writer(global_video_path, fps=30, crf=CRF)

    # length, width, height = get_video_lwh(global_video_path)
    width, height = 640 * 4, 480 * 4
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[:, 0], verts_glob_list
    )
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8

    T, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.cpu().clone(),
        beta=1.3,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

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
            if t - 1 < start_ind or t - 1 > end_ind:
                renderer.scene.remove_geometry(f"img_mesh_{t - 1}")
        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

        if t < start_ind or t > end_ind:
            if t > end_ind:
                pid = vid2[:2]
                sid = vid2[3:]
            else:
                pid = vid[:2]
                sid = vid[3:]
            img_path = f"/mnt/dhd/body-pose-dataset/EMDB/{pid}/{sid}/images/{t:05d}.jpg"
            input_img = cv2.imread(img_path)
            offset = verts.mean(dim=0).cpu().numpy()
            offset[1] += 1.5

            img_h, img_w = input_img.shape[:2]
            input_img = cv2.resize(input_img, (img_w // 4, img_h // 4))
            img_mesh = convert_image_to_mesh(input_img[:, :, ::-1], offset, R_w2c)
            renderer.scene.add_geometry(
                f"img_mesh_{t}", img_mesh, mat_settings._materials[Settings.LIT]
            )
        img = renderer.render_to_image()
        # import ipdb; ipdb.set_trace()
        # o3d.io.write_image("out/tmp.png", img)
        # img = cv2.imread("out/tmp.png")
        writer.write_frame(np.array(img))
    writer.close()
    return global_video_path
    # os.makedirs(
    #     os.path.dirname(f"out/{vis_type}_scene_time/{fname}.png"), exist_ok=True
    # )
    # # cv2.imwrite(f"out/{vis_type}_scene_time/{fname}.png", img)


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
    vertex_positions = vertex_positions.reshape(yvalues.size, xvalues.size)[
        :-1, :-1
    ].flatten()

    # Now create triangles (keep vertices in anticlockwise order when making triangles)
    top_triangles = np.vstack(
        ((vertex_positions + 1, vertex_positions, vertex_positions + xvalues.shape[0]))
    ).transpose(1, 0)
    vertex_positions = np.arange(xvalues.size * yvalues.size)
    vertex_positions = vertex_positions.reshape(yvalues.size, xvalues.size)[
        1:, 1:
    ].flatten()
    bottom_triangles = np.vstack(
        ((vertex_positions - 1, vertex_positions, vertex_positions - xvalues.shape[0]))
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
