import glob
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import contextlib
import json

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

from hmr4d.utils.smplx_utils import make_smplx
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from motiondiff.utils.vis_scenepic import ScenepicVisualizer

trans_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])


device = torch.device("cuda")
sp_visualizer = ScenepicVisualizer(
    "/home/jiefengl/git/physdiff_megm/data/smpl_data", device=device
)
smpl_dict = sp_visualizer.smpl_dict


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


def render_mesh(img, mesh, face, cam_param, img_white_bg=None, deg=0):
    # mesh
    cur_mesh = mesh.copy()
    mesh = trimesh.Trimesh(mesh, face)

    if (deg != 0):
        rot = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0], point=np.mean(cur_mesh, axis=0))
        mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    # baseColorFactor = (1.0, 1.0, 0.9, 1.0)   # graw color
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    # Set other material properties for appearance
    #material.baseColorFactor = [0.25, 0.4, 0.65, 1.0]      # gray
    #material.baseColorFactor = [0.3, 0.3, 0.3, 1.0]      # silver
    material.baseColorFactor = [1.0, 1.0, 0.9, 1.0]     # white
    material.metallicFactor = 0.2
    material.roughnessFactor = 0.7

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

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
    flags = (pyrender.RenderFlags.RGBA |
             pyrender.RenderFlags.SKIP_CULL_FACES)
    rgb, depth = renderer.render(scene, flags=flags)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img


def get_local_camint(bbox):
    input_body_shape = (256, 192)
    # output_hm_shape = (16, 16, 12)
    focal = (5000, 5000)
    princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)
    focal = list(focal)
    princpt = list(princpt)

    focal[0] = focal[0] / input_body_shape[1] * bbox[2]
    focal[1] = focal[1] / input_body_shape[0] * bbox[3]
    princpt[0] = princpt[0] / input_body_shape[1] * bbox[2] + bbox[0]
    princpt[1] = princpt[1] / input_body_shape[0] * bbox[3] + bbox[1]

    return focal + princpt


if __name__ == "__main__":
    device = "cuda:0"
    all_videos = glob.glob("/mnt/disk3/motion-x++/video/*/*.mp4")
    smplx_layer = make_smplx(type="smplx-motionx").to(device)

    motionx_db = {}
    motionx_db_old = torch.load("/mnt/disk3/motion-x++/motionxpp_smplhpose.pth")

    for video_file in tqdm(all_videos):
        subset = video_file.split("/")[-2]
        # if subset in ["fitness", "idea400_light"]:
        if subset in ["fitness"]:
            continue
        if subset == "idea400_light":
            subset = "idea400"
        file_name = video_file.split("/")[-1].split(".mp4")[0]
        vid_name = f"{subset}_{file_name}"
        # video_file = f"/mnt/disk3/motion-x++/video/{subset}/{file_name}.mp4"
        local_motion_file = f"/mnt/disk3/motion-x++/motion/mesh_recovery/local_motion/{subset}/{file_name}.json"
        global_motion_file = f"/mnt/disk3/motion-x++/motion/mesh_recovery/global_motion/{subset}/{file_name}.json"
        keypoint_file = f"/mnt/disk3/motion-x++/motion/keypoints/{subset}/{file_name}.json"
        semmantic_text_file = f"/mnt/disk3/motion-x++/text/semantic_label/{subset}/{file_name}.txt"
        whole_desc_file = f"/mnt/disk3/motion-x++/text/wholebody_pose_description/{subset}/{file_name}.json"
        assert os.path.exists(local_motion_file), (local_motion_file, video_file)
        # assert os.path.exists(global_motion_file), (global_motion_file, video_file)
        assert os.path.exists(video_file), (video_file, video_file)
        assert os.path.exists(keypoint_file), (keypoint_file, video_file)
        assert os.path.exists(semmantic_text_file), (semmantic_text_file, video_file)
        if vid_name not in motionx_db_old:
            continue

        # if not os.path.exists(global_motion_file):
        #     print(f"skip {global_motion_file}")
        #     continue
        with contextlib.redirect_stdout(None):
            db_local = COCO(local_motion_file)
            # db_global = COCO(global_motion_file)

            try:
                db_global = COCO(global_motion_file)
            except:
                print(f"skip {global_motion_file}")
                continue
            try:
                db_keypoint = COCO(keypoint_file)
            except:
                print(f"skip {keypoint_file}")
                continue
            with open(semmantic_text_file, "r") as f:
                s_text = f.read()
            if os.path.exists(whole_desc_file):
                with open(whole_desc_file, "r") as f:
                    db_w_text = json.load(f)
            else:
                db_w_text = None

        cap = cv2.VideoCapture(video_file)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)

        # output_path = f"out/vis_motionxpp/{subset}/{file_name}.mp4"
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        with torch.no_grad():
            all_images = []
            pose_lst = []
            global_orient_w_lst = []
            transl_w_lst = []
            global_orient_c_lst = []
            transl_c_lst = []
            beta_lst = []
            lhand_lst = []
            rhand_lst = []
            expr_lst = []
            w_text_lst = []
            cam_R_lst = []
            cam_T_lst = []
            body_kpts_lst = []
            bbox_lst = []
            vis = False
            for i, aid in tqdm(enumerate(db_global.anns.keys())):
                if vis:
                    ret, frame = cap.read()
                    # all_images.append(frame.copy())
                ann = db_local.anns[aid]
                ann_global = db_global.anns[aid]
                ann_keypoint = db_keypoint.anns[aid]
                if db_w_text is not None:
                    w_text = db_w_text[str(i)]
                    w_text_lst.append(w_text)

                smpl_layer = smpl_dict['neutral']
                bbox = ann["bbox"]
                bbox_lst.append(bbox)
                body_kpts = np.array(ann_keypoint["body_kpts"])
                body_kpts_lst.append(body_kpts)
                # local camera intrinsics
                intrins_global = ann_global["cam_params"]["intrins"] # [1500.0, 1500.0, 960.0, 540.0]
                orig_intrins_local = get_local_camint(bbox)
                f_scale = intrins_global[0] / orig_intrins_local[0]

                # adjust local camera intrinsics and translation
                pose_body = np.array(ann_global["smplx_params"]["pose_body"])
                global_orient_w = np.array(ann_global["smplx_params"]["root_orient"])
                transl_w = np.array(ann_global["smplx_params"]["trans"])
                left_hand_pose = np.array(ann_global["smplx_params"]["pose_hand"][:45])
                right_hand_pose = np.array(ann_global["smplx_params"]["pose_hand"][45:])
                expr = np.array(ann_global["smplx_params"]["face_expr"][:10])
                beta = np.ones_like(expr)

                global_orient_c = np.array(ann["smplx_params"]["root_pose"])
                transl_c = np.array(ann["smplx_params"]["trans"])
                transl_c[..., 2] = transl_c[..., 2] * f_scale

                if vis:
                    focal = (intrins_global[0], intrins_global[1])
                    princpt = (intrins_global[2], intrins_global[3])
                    output_c = smplx_layer(
                        betas=torch.from_numpy(beta).to(device=device, dtype=torch.float32).unsqueeze(0),
                        body_pose=torch.from_numpy(pose_body).to(device=device, dtype=torch.float32).unsqueeze(0),
                        global_orient=torch.from_numpy(global_orient_c).to(device=device, dtype=torch.float32).unsqueeze(0),
                        right_hand_pose=torch.from_numpy(right_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0),
                        left_hand_pose=torch.from_numpy(left_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0),
                        transl=torch.from_numpy(transl_c).to(device=device, dtype=torch.float32).unsqueeze(0),
                        jaw_pose=torch.zeros([1, 3]).to(device),
                        leye_pose=torch.zeros([1, 3]).to(device),
                        reye_pose=torch.zeros([1, 3]).to(device),
                        expression=torch.from_numpy(expr).to(device=device, dtype=torch.float32).unsqueeze(0),
                    )

                    mesh_cam_c = output_c.vertices.detach().cpu().numpy()[0]
                    frame_loc = render_mesh(frame.copy(), mesh_cam_c, smplx_layer.faces, {'focal': focal, 'princpt': princpt})

                R_w2c = np.array(ann_global["cam_params"]["cam_R"])
                t_w2c = np.array(ann_global["cam_params"]["cam_T"])
                T_w2c = np.eye(4)
                T_w2c[:3, :3] = R_w2c
                T_w2c[:3, 3] = t_w2c

                # T_c2w = np.linalg.inv(T_w2c)

                # transform the global motion
                transl_w = (trans_matrix @ transl_w[..., None])[..., 0]
                global_orient_w_mat = axis_angle_to_matrix(torch.from_numpy(global_orient_w).float())
                global_orient_w_mat = torch.from_numpy(trans_matrix).float() @ global_orient_w_mat
                global_orient_w = matrix_to_axis_angle(global_orient_w_mat).numpy()

                # pose_pad = np.concatenate([pose_body, np.zeros_like(pose_body[:6])], axis=0)
                # smpl_output = smpl_layer(
                #     betas=torch.from_numpy(beta).to(device).float().unsqueeze(0),
                #     global_orient=torch.from_numpy(global_orient_w).to(device).float().unsqueeze(0),
                #     body_pose=torch.from_numpy(pose_pad).to(device).float().unsqueeze(0),
                #     transl=torch.from_numpy(transl_w).to(device).float().unsqueeze(0),
                #     orig_joints=True,
                #     pose2rot=True,
                # )
                # j3d = smpl_output.joints.detach()
                # # put the person on the ground by -min(z)
                # ground_z = j3d[..., 2].flatten(-2).min(dim=-1)[0].cpu().numpy()  # (B,)  Minimum z value
                # offset_xy = transl_w.copy()
                # offset_xy[..., 2] = ground_z
                # transl_w = transl_w - offset_xy

                trans_glob_mat = np.eye(4)
                trans_glob_mat[:3, :3] = trans_matrix
                # trans_glob_mat[:3, 3] = -offset_xy
                # T_c2w = T_c2w 
                T_w2c = T_w2c @ np.linalg.inv(trans_glob_mat)

                # T_w2c = np.linalg.inv(T_c2w)
                R_w2c = T_w2c[:3, :3]
                t_w2c = T_w2c[:3, 3]

                global_orient_w_mat = axis_angle_to_matrix(torch.from_numpy(global_orient_w).float())
                global_orient_w_c_mat = R_w2c @ global_orient_w_mat.numpy()
                global_orient_w_c = matrix_to_axis_angle(torch.from_numpy(global_orient_w_c_mat).float()).numpy()
                transl_w_c = R_w2c @ transl_w + t_w2c

                cam_R_lst.append(R_w2c)
                cam_T_lst.append(t_w2c)

                if vis:
                    focal = (intrins_global[0], intrins_global[1])
                    princpt = (intrins_global[2], intrins_global[3])
                    output_w = smplx_layer(
                        betas=torch.from_numpy(beta).to(device=device, dtype=torch.float32).unsqueeze(0),
                        body_pose=torch.from_numpy(pose_body).to(device=device, dtype=torch.float32).unsqueeze(0),
                        global_orient=torch.from_numpy(global_orient_w_c).to(device=device, dtype=torch.float32).unsqueeze(0),
                        right_hand_pose=torch.from_numpy(right_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0),
                        left_hand_pose=torch.from_numpy(left_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0),
                        transl=torch.from_numpy(transl_w_c).to(device=device, dtype=torch.float32).unsqueeze(0),
                        jaw_pose=torch.zeros([1, 3]).to(device),
                        leye_pose=torch.zeros([1, 3]).to(device),
                        reye_pose=torch.zeros([1, 3]).to(device),
                        expression=torch.from_numpy(expr).to(device=device, dtype=torch.float32).unsqueeze(0),
                    )
                    joints_w = output_w.joints

                    mesh_cam_w = output_w.vertices.detach().cpu().numpy()[0]
                    frame_global = render_mesh(frame.copy(), mesh_cam_w, smplx_layer.faces, {'focal': focal, 'princpt': princpt})

                    frame = np.concatenate([frame_global, frame_loc], axis=1)
                    out.write(frame.astype(np.uint8))
                # R_vec = cv2.Rodrigues(R_w2c)[0]
                # t_vec = np.array(t_w2c)
                # # assert (R_vec == 0).all(), "R_vec is zero"

                # camera_matrix_global = np.array([[intrins_global[0], 0, intrins_global[2]],[0, intrins_global[1], intrins_global[3]],[0,0,1]])
                # camera_params_global = {"matrix": camera_matrix_global, "R": R_vec, "T": t_vec}
                # R = torch.from_numpy(R).to(device=device, dtype=torch.float32)[None, :, :]
                # T = torch.from_numpy(T).to(device=device, dtype=torch.float32)[None, :]
                # camera_matrix_global = torch.from_numpy(camera_matrix_global).to(device=device, dtype=torch.float32)[None, :, :]
                # image_size = torch.tensor([height, width], device=device).unsqueeze(0)
                # try:
                #     cameras_global = cameras_from_opencv_projection(R, T, camera_matrix_global, image_size)
                # except:
                #     import ipdb; ipdb.set_trace()

                pose_lst.append(pose_body)
                global_orient_w_lst.append(global_orient_w)
                global_orient_c_lst.append(global_orient_c)
                transl_w_lst.append(transl_w)
                transl_c_lst.append(transl_c)
                beta_lst.append(beta)
                lhand_lst.append(left_hand_pose)
                rhand_lst.append(right_hand_pose)
                expr_lst.append(expr)

                # vis = True
                # if vis:
                #     pose_body = torch.from_numpy(pose_body).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     global_orient_w = torch.from_numpy(global_orient_w).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     transl_w = torch.from_numpy(transl_w).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     left_hand_pose = torch.from_numpy(left_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     right_hand_pose = torch.from_numpy(right_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     expr = torch.from_numpy(expr).to(device=device, dtype=torch.float32).unsqueeze(0)
                #     beta = torch.from_numpy(beta).to(device=device, dtype=torch.float32).unsqueeze(0)

                #     output_w = smplx_layer(
                #         betas=beta,
                #         body_pose=pose_body,
                #         global_orient=global_orient_w,
                #         right_hand_pose=right_hand_pose,
                #         left_hand_pose=left_hand_pose,
                #         transl=transl_w,
                #         jaw_pose=torch.zeros([1, 3]).to(device),
                #         leye_pose=torch.zeros([1, 3]).to(device),
                #         reye_pose=torch.zeros([1, 3]).to(device),
                #         expression=expr,
                #     )
                #     joints_w = output_w.joints

                #     vertices = output_w.vertices
                #     mesh_cam_w = vertices #(bs 10475 3)

            cap.release()
            # out.release()
            pose_lst = torch.from_numpy(np.stack(pose_lst)).float()
            global_orient_w_lst = torch.from_numpy(np.stack(global_orient_w_lst)).float()
            global_orient_c_lst = torch.from_numpy(np.stack(global_orient_c_lst)).float()
            transl_w_lst = torch.from_numpy(np.stack(transl_w_lst)).float()
            transl_c_lst = torch.from_numpy(np.stack(transl_c_lst)).float()
            beta_lst = torch.from_numpy(np.stack(beta_lst)).float()
            lhand_lst = torch.from_numpy(np.stack(lhand_lst)).float()
            rhand_lst = torch.from_numpy(np.stack(rhand_lst)).float()
            expr_lst = torch.from_numpy(np.stack(expr_lst)).float()
            cam_R_lst = torch.from_numpy(np.stack(cam_R_lst)).float()
            cam_T_lst = torch.from_numpy(np.stack(cam_T_lst)).float()
            body_kpts_lst = torch.from_numpy(np.stack(body_kpts_lst)).float()
            bbox_lst = torch.from_numpy(np.stack(bbox_lst)).float()

            motionx_db[vid_name] = {
                "pose": pose_lst,
                "trans": transl_w_lst,
                "trans_c": transl_c_lst,
                "global_orient": global_orient_w_lst,
                "global_orient_c": global_orient_c_lst,
                "beta": beta_lst,
                "lhand": lhand_lst,
                "rhand": rhand_lst,
                "expr": expr_lst,
                "text": s_text,
                "w_text": w_text_lst,
                "cam_R": cam_R_lst,
                "cam_T": cam_T_lst,
                "body_kpts": body_kpts_lst,
                "bbox": bbox_lst,
                "intrins": intrins_global,
                "subset": subset,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            # import ipdb; ipdb.set_trace()
print(f"total {len(motionx_db)} samples")
torch.save(motionx_db, "/mnt/disk3/motion-x++/motionxpp_smplxposev3.pth")

