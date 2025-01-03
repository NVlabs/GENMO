import glob
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from pycocotools.coco import COCO
from pytorch3d.renderer import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from tqdm import tqdm
import json

from hmr4d.utils.smplx_utils import make_smplx


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the given frame.

    Args:
    frame (numpy.ndarray): The input frame on which to draw the bounding box.
    bbox (tuple): The bounding box coordinates (x, y, w, h).
    color (tuple): The color of the bounding box (default is green).
    thickness (int): The thickness of the bounding box lines (default is 2).

    Returns:
    numpy.ndarray: The frame with the bounding box drawn.
    """
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    return frame


def draw_keypoints(frame, keypoints, color=(0, 255, 0), thickness=2):
    for keypoint in keypoints:
        x, y, _ = keypoint
        cv2.circle(frame, (int(x), int(y)), 2, color, thickness)
    return frame


def render_mesh(img, mesh, face, cam_param, img_white_bg=None, deg=0):
    # mesh
    cur_mesh = mesh.copy()
    mesh = trimesh.Trimesh(mesh, face)

    if deg != 0:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(deg), [0, 1, 0], point=np.mean(cur_mesh, axis=0)
        )
        mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    # baseColorFactor = (1.0, 1.0, 0.9, 1.0)   # graw color
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )

    # Set other material properties for appearance
    # material.baseColorFactor = [0.25, 0.4, 0.65, 1.0]      # gray
    # material.baseColorFactor = [0.3, 0.3, 0.3, 1.0]      # silver
    material.baseColorFactor = [1.0, 1.0, 0.9, 1.0]  # white
    material.metallicFactor = 0.2
    material.roughnessFactor = 0.7

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, "mesh")

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(
        fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1]
    )
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0
    )

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
    rgb, depth = renderer.render(scene, flags=flags)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img


def render_mesh_to_image(smpl_vertices, smpl_faces, cameras, image, camera_params):

    raster_settings = RasterizationSettings(image_size=[image.shape[0], image.shape[1]],
                                            blur_radius=0,
                                            faces_per_pixel=1,
                                            bin_size=0,
                                            max_faces_per_bin=1000,
                                            perspective_correct=True)
    smpl_vertices = smpl_vertices.detach().cpu().numpy()

    verts = torch.tensor([smpl_vertices], dtype=torch.float32, device=device)
    faces = torch.from_numpy(smpl_faces.astype(np.int32)).to(device)
    faces = faces[None, :, :]
    verts_rgb = torch.ones_like(verts)
    verts_rgb[:, :, 2] = verts_rgb[:, :, 2] * 0
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras,
                                                      raster_settings=raster_settings),
                            shader=HardPhongShader(device=device, cameras=cameras))
    images = renderer(
        meshes_world=Meshes(verts=verts.to(device), faces=faces.to(device), textures=textures))

    images = images.cpu().numpy().squeeze() * 255
    images_rgb = images[:, :, :3].astype(np.uint8)
    mask = images_rgb.max(axis=2) < 255  
    images_rgb_resized = cv2.resize(images_rgb, (image.shape[1], image.shape[0]))
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
    image[mask_resized] = images_rgb_resized[mask_resized]
    # render with color vertex 
    # camera_matrix = np.array(camera_params['matrix'])
    # camera_matrix = torch.tensor([camera_matrix], dtype=torch.float32, device=device)
    # rvec = camera_params['R']
    # tvec = camera_params['T']
    # cameraMatrix = np.array(camera_params['matrix'])
    # distCoeffs = numpy.zeros([4, 1])
    # points = verts.cpu().numpy()
    # points2d = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
    # points2d = points2d[0]
    # points2d = points2d.astype(int)
    # for i in range(points2d.shape[0]):
    #     cv2.circle(image, tuple(points2d[i][0]), 1, (0, 255, 255))
    return image


if __name__ == "__main__":
    device = "cuda:0"
    all_videos = glob.glob("/mnt/disk3/motion-x++/video/*/*.mp4")
    smplx_layer = make_smplx(type="smplx-motionx").to(device)

    for video_file in tqdm(all_videos):
        subset = video_file.split("/")[-2]
        if subset in ["fitness", "idea400_light"]:
            continue
        file_name = video_file.split("/")[-1].split(".mp4")[0]
        # video_file = f"/mnt/disk3/motion-x++/video/{subset}/{file_name}.mp4"
        local_motion_file = f"/mnt/disk3/motion-x++/motion/mesh_recovery/local_motion/{subset}/{file_name}.json"
        global_motion_file = f"/mnt/disk3/motion-x++/motion/mesh_recovery/global_motion/{subset}/{file_name}.json"
        keypoint_file = f"/mnt/disk3/motion-x++/motion/keypoints/{subset}/{file_name}.json"
        semmantic_text_file = f"/mnt/disk3/motion-x++/text/semantic_label/{subset}/{file_name}.txt"
        whole_desc_file = f"/mnt/disk3/motion-x++/text/wholebody_pose_description/{subset}/{file_name}.json"
        assert os.path.exists(local_motion_file), (local_motion_file, video_file)
        assert os.path.exists(video_file), (video_file, video_file)
        assert os.path.exists(keypoint_file), (keypoint_file, video_file)
        assert os.path.exists(semmantic_text_file), (semmantic_text_file, video_file)

        db_local = COCO(local_motion_file)
        db_global = COCO(global_motion_file)
        db_keypoint = COCO(keypoint_file)
        with open(semmantic_text_file, "r") as f:
            s_text = f.read()
        with open(whole_desc_file, "r") as f:
            w_text = json.load(f)

        cap = cv2.VideoCapture(video_file)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = f"out/vis_motionxpp/{subset}/{file_name}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        with torch.no_grad():
            for aid in tqdm(db_global.anns.keys()):
                ret, frame = cap.read()
                ann = db_local.anns[aid]
                ann_global = db_global.anns[aid]
                ann_keypoint = db_keypoint.anns[aid]

                R = np.array(ann_global["cam_params"]["cam_R"])
                T = np.array(ann_global["cam_params"]["cam_T"])
                R_vec = cv2.Rodrigues(R)[0]
                if (R_vec == 0).all():
                    continue
                root_pose = torch.FloatTensor(np.array(ann["smplx_params"]['root_pose'])).unsqueeze(0).to(device)
                body_pose = torch.FloatTensor(np.array(ann["smplx_params"]['body_pose'])).unsqueeze(0).to(device)
                lhand_pose = torch.FloatTensor(np.array(ann["smplx_params"]['lhand_pose'])).unsqueeze(0).to(device)
                rhand_pose = torch.FloatTensor(np.array(ann["smplx_params"]['rhand_pose'])).unsqueeze(0).to(device)
                jaw_pose = torch.FloatTensor(np.array(ann["smplx_params"]['jaw_pose'])).unsqueeze(0).to(device)
                shape = torch.FloatTensor(np.array(ann["smplx_params"]['shape'])).unsqueeze(0).to(device)
                expr = torch.FloatTensor(np.array(ann["smplx_params"]['expr'])).unsqueeze(0).to(device)
                cam_trans = torch.FloatTensor(np.array(ann["smplx_params"]['trans'])).unsqueeze(0).to(device)

                output = smplx_layer(betas=shape, 
                                    body_pose=body_pose, 
                                    global_orient=root_pose, 
                                    right_hand_pose=rhand_pose,
                                    left_hand_pose=lhand_pose, 
                                    transl=cam_trans,
                                    jaw_pose=torch.zeros([1, 3]).to(device),
                                    leye_pose=torch.zeros([1, 3]).to(device),
                                    reye_pose=torch.zeros([1, 3]).to(device), 
                                    expression=expr)
                joints = output.joints

                vertices = output.vertices
                # mesh_cam = vertices + cam_trans[:, None, :] #(bs 10475 3)
                mesh_cam = vertices #(bs 10475 3)
                bbox = ann["bbox"]

                input_body_shape = (256, 192)
                output_hm_shape = (16, 16, 12)
                focal = (5000, 5000)
                princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)
                focal = list(focal)
                princpt = list(princpt)

                focal[0] = focal[0] / input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / input_body_shape[0] * bbox[3] + bbox[1]

                # intrins = ann_global["cam_params"]["intrins"]  # [1500.0, 1500.0, 960.0, 540.0]
                cameras_vis = np.array(
                    [[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]]
                )
                cameras_vis = torch.from_numpy(cameras_vis)[None, :, :].to(device).float()
                camera_params_vis = {"matrix": cameras_vis, "R": cv2.Rodrigues(np.eye(3))[0], "T": np.zeros(3)}
                cameras_vis = cameras_from_opencv_projection(
                    R=torch.eye(3)[None, :, :].to(device).float(),
                    tvec=torch.zeros([1, 3]).to(device).float(),
                    camera_matrix=cameras_vis,
                    image_size=torch.tensor([height, width])[None].to(device).float(),
                )

                img = draw_bbox(frame, bbox)
                # mesh_cam = mesh_cam.cpu().numpy()[0]

                # img = render_mesh(img, mesh_cam, smplx_layer.faces, {"focal": focal, "princpt": princpt})
                # processed_frame = render_mesh_to_image(mesh_cam[0], smplx_layer.faces, cameras_vis, frame.copy(), camera_params_vis)
                processed_frame = draw_keypoints(
                    frame.copy(), ann_keypoint["body_kpts"]
                )
                intrins_global = ann_global["cam_params"]["intrins"] # [1500.0, 1500.0, 960.0, 540.0]
                camera_matrix_global = np.array([[intrins_global[0], 0, intrins_global[2]],[0, intrins_global[1], intrins_global[3]],[0,0,1]])

                R = np.array(ann_global["cam_params"]["cam_R"])
                T = np.array(ann_global["cam_params"]["cam_T"])
                R_vec = cv2.Rodrigues(R)[0]
                if (R_vec == 0).all():
                    continue
                T_vec = np.array(ann_global["cam_params"]["cam_T"])

                camera_params_global = {"matrix": camera_matrix_global, "R": R_vec, "T": T_vec}
                R = torch.from_numpy(R).to(device=device, dtype=torch.float32)[None, :, :]
                T = torch.from_numpy(T).to(device=device, dtype=torch.float32)[None, :]
                camera_matrix_global = torch.from_numpy(camera_matrix_global).to(device=device, dtype=torch.float32)[None, :, :]
                image_size = torch.tensor([frame.shape[0], frame.shape[1]], device=device).unsqueeze(0)
                cameras_global = cameras_from_opencv_projection(R, T, camera_matrix_global, image_size)

                global_orient_w = np.array(ann_global["smplx_params"]["root_orient"])
                transl_w = np.array(ann_global["smplx_params"]["trans"])
                global_orient_w = torch.from_numpy(global_orient_w).to(device=device, dtype=torch.float32).unsqueeze(0)
                transl_w = torch.from_numpy(transl_w).to(device=device, dtype=torch.float32).unsqueeze(0)

                output_w = smplx_layer(
                    betas=torch.zeros_like(shape),
                    body_pose=body_pose,
                    global_orient=global_orient_w,
                    right_hand_pose=rhand_pose,
                    left_hand_pose=lhand_pose,
                    transl=transl_w,
                    jaw_pose=torch.zeros([1, 3]).to(device),
                    leye_pose=torch.zeros([1, 3]).to(device),
                    reye_pose=torch.zeros([1, 3]).to(device),
                    expression=expr,
                )
                joints_w = output_w.joints

                vertices = output_w.vertices
                mesh_cam_w = vertices #(bs 10475 3)
                processed_frame_w = render_mesh_to_image(vertices[0], smplx_layer.faces, cameras_global, frame.copy(), camera_params_global)

                img = np.concatenate([processed_frame, processed_frame_w], axis=1)
                # img = processed_frame.astype(np.uint8)
                out.write(img)

            cap.release()
            out.release()
            import ipdb; ipdb.set_trace()
