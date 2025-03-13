import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
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


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file_name", type=str, required=True)
    # arg_parser.add_argument("--input_dir", type=str, required=True)
    args = arg_parser.parse_args()

    file_name = args.file_name
    input_dir = os.path.dirname(file_name)
    print(file_name)
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
    data = torch.load(file_name)

    pred_smpl_params_global = data["pred_smpl_params_global"]
    time_pred_ay_smpl_out_list = smplx(**pred_smpl_params_global)
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in time_pred_ay_smpl_out_list.vertices]
    )
    vid1 = data["vid1"]
    vid2 = data["vid2"]
    start_ind = data["start_ind"]
    end_ind = data["end_ind"]
    if "multicond_args" in data:
        end_ind = max(
            data["multicond_args"]["video2"]["start_ind"],
            data["multicond_args"]["music1"]["end_ind"],
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
    global_video_path = file_name.replace(".pth", ".mp4").replace(
        input_dir, f"{input_dir}_video"
    )
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
        beta=1.8,
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
    # renderer.scene.add_geometry(
    #     "mesh_ground", ground_mesh, o3d.visualization.rendering.MaterialRecord()
    # )
    # faces_list = list(torch.unbind(faces, dim=0))  # + [gf]

    key_frame_inds = [420]
    key_frame_colors = torch.tensor([1.0, 0.0, 0.0])
    for t in key_frame_inds:
        verts = verts_glob_list[t]  # + [gv]
        mesh = create_meshes(verts, faces, key_frame_colors)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.base_color = key_frame_colors.tolist() + [
            0.8
        ]  # 0.2 alpha for 0.8 transparency
        mat.shader = Settings.Transparency
        mat.thickness = 1.0
        mat.transmission = 1.0
        mat.absorption_distance = 10
        mat.absorption_color = [0.5, 0.5, 0.5]
        renderer.scene.add_geometry(f"keyframe_mesh_{t}", mesh, mat)

    renderer.scene.set_background(
        [0, 0, 0, 0]
    )  # RGBA where alpha=0 means fully transparent
    img = renderer.render_to_image()

    # Convert black background to transparent
    img_np = np.array(img)
    rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = img_np
    # Create alpha channel - make black pixels transparent
    is_black = (img_np[..., 0] < 30) & (img_np[..., 1] < 30) & (img_np[..., 2] < 30)
    rgba[..., 3] = np.where(is_black, 0, 255)

    # Save as PNG with transparency
    cv2.imwrite("out/tmp.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    exit()

    T_c2w = camera.get_view_matrix()
    R_w2c = T_c2w[:3, :3].T
    for t, verts in tqdm(enumerate(verts_glob_list[:])):
        verts = verts_glob_list[t]  # + [gv]
        mat = o3d.visualization.rendering.MaterialRecord()
        # mat.base_color = [0.9, 0.9, 0.9, 0.3 + t * 0.7 / T]
        mat.base_color = color_purple.tolist() + [0.3]
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

        if t < start_ind or t > end_ind:
            if t > end_ind:
                pid = vid2[:2]
                sid = vid2[3:]
            else:
                pid = vid1[:2]
                sid = vid1[3:]
            img_path = f"/mnt/dhd/body-pose-dataset/EMDB/{pid}/{sid}/images/{t:05d}.jpg"
            input_img = cv2.imread(img_path)
            offset = verts.mean(dim=0).cpu().numpy()
            offset[1] += 0.8
            offset[0] -= 2.0

            img_h, img_w = input_img.shape[:2]
            input_img = cv2.resize(input_img, (img_w // 32, img_h // 32))
            img_mesh = convert_image_to_mesh(input_img[:, :, ::-1], offset, R_w2c)
            renderer.scene.add_geometry(
                f"img_mesh_{t}", img_mesh, mat_settings._materials[Settings.LIT]
            )
        img = renderer.render_to_image()

        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

        img = renderer.render_to_image()
        # import ipdb; ipdb.set_trace()
        # o3d.io.write_image("out/tmp.png", img)
        # output_dir = os.path.dirname(global_video_path)
        # o3d.io.write_image(os.path.join(output_dir, f"tmp_{t:05d}.png"), img)
        # img = cv2.imread("out/tmp.png")
        writer.write_frame(np.array(img))
    writer.close()
    # os.makedirs(
    #     os.path.dirname(f"out/{vis_type}_scene_time/{fname}.png"), exist_ok=True
    # )
    # # cv2.imwrite(f"out/{vis_type}_scene_time/{fname}.png", img)
