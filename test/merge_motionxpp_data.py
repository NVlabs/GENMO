import glob
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import json

import cv2
import numpy as np
import torch
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
import contextlib
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
            db_global = COCO(global_motion_file)

        with torch.no_grad():
            bbox_lst = []
            for i, aid in enumerate(db_global.anns.keys()):
                # ret, frame = cap.read()
                # all_images.append(frame.copy())
                ann = db_local.anns[aid]

                bbox = ann["bbox"]
                bbox_lst.append(bbox)
            bbox_lst = np.stack(bbox_lst)
            data = motionx_db_old[vid_name]
            data["bbox"] = bbox_lst
            data["subset"] = subset
            data["file_name"] = file_name
            motionx_db[vid_name] = data
        continue
        # try:
        #     db_global = COCO(global_motion_file)
        # except:
        #     print(f"skip {global_motion_file}")
        #     continue
        # try:
        #     db_keypoint = COCO(keypoint_file)
        # except:
        #     print(f"skip {keypoint_file}")
        #     continue
        # with open(semmantic_text_file, "r") as f:
        #     s_text = f.read()
        # if os.path.exists(whole_desc_file):
        #     with open(whole_desc_file, "r") as f:
        #         db_w_text = json.load(f)
        # else:
        #     db_w_text = None

        cap = cv2.VideoCapture(video_file)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            continue
        # fps = cap.get(cv2.CAP_PROP_FPS)

        # output_path = f"out/vis_motionxpp/{subset}/{file_name}.mp4"
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        with torch.no_grad():
            all_images = []
            pose_lst = []
            transl_lst = []
            beta_lst = []
            lhand_lst = []
            rhand_lst = []
            expr_lst = []
            w_text_lst = []
            cam_R_lst = []
            cam_T_lst = []
            body_kpts_lst = []
            for i, aid in tqdm(enumerate(db_global.anns.keys())):
                # ret, frame = cap.read()
                # all_images.append(frame.copy())
                ann = db_local.anns[aid]
                ann_global = db_global.anns[aid]
                ann_keypoint = db_keypoint.anns[aid]
                if db_w_text is not None:
                    w_text = db_w_text[str(i)]
                    w_text_lst.append(w_text)

                bbox = ann["bbox"]
                body_kpts = np.array(ann_keypoint["body_kpts"])
                body_kpts_lst.append(body_kpts)
                intrins_global = ann_global["cam_params"]["intrins"] # [1500.0, 1500.0, 960.0, 540.0]
                camera_matrix_global = np.array([[intrins_global[0], 0, intrins_global[2]],[0, intrins_global[1], intrins_global[3]],[0,0,1]])

                R = np.array(ann_global["cam_params"]["cam_R"])
                T = np.array(ann_global["cam_params"]["cam_T"])
                cam_R_lst.append(R)
                cam_T_lst.append(T)

                R_vec = cv2.Rodrigues(R)[0]
                T_vec = np.array(ann_global["cam_params"]["cam_T"])
                # assert (R_vec == 0).all(), "R_vec is zero"
                camera_params_global = {"matrix": camera_matrix_global, "R": R_vec, "T": T_vec}
                R = torch.from_numpy(R).to(device=device, dtype=torch.float32)[None, :, :]
                T = torch.from_numpy(T).to(device=device, dtype=torch.float32)[None, :]
                camera_matrix_global = torch.from_numpy(camera_matrix_global).to(device=device, dtype=torch.float32)[None, :, :]
                image_size = torch.tensor([height, width], device=device).unsqueeze(0)
                try:
                    cameras_global = cameras_from_opencv_projection(R, T, camera_matrix_global, image_size)
                except:
                    import ipdb; ipdb.set_trace()

                pose_body = np.array(ann_global['smplx_params']['pose_body'])
                global_orient_w = np.array(ann_global['smplx_params']['root_orient'])
                transl_w = np.array(ann_global['smplx_params']['trans'])
                left_hand_pose = np.array(ann_global['smplx_params']['pose_hand'][:45])
                right_hand_pose = np.array(ann_global['smplx_params']['pose_hand'][45:])
                expr = np.array(ann_global['smplx_params']['face_expr'][:10])
                beta = np.ones_like(expr)

                pose_lst.append(pose_body)
                transl_lst.append(transl_w)
                beta_lst.append(beta)
                lhand_lst.append(left_hand_pose)
                rhand_lst.append(right_hand_pose)
                expr_lst.append(expr)

                vis = False
                if vis:
                    pose_body = torch.from_numpy(pose_body).to(device=device, dtype=torch.float32).unsqueeze(0)
                    global_orient_w = torch.from_numpy(global_orient_w).to(device=device, dtype=torch.float32).unsqueeze(0)
                    transl_w = torch.from_numpy(transl_w).to(device=device, dtype=torch.float32).unsqueeze(0)
                    left_hand_pose = torch.from_numpy(left_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
                    right_hand_pose = torch.from_numpy(right_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
                    expr = torch.from_numpy(expr).to(device=device, dtype=torch.float32).unsqueeze(0)
                    beta = torch.from_numpy(beta).to(device=device, dtype=torch.float32).unsqueeze(0)

                    output_w = smplx_layer(
                        betas=beta,
                        body_pose=pose_body,
                        global_orient=global_orient_w,
                        right_hand_pose=right_hand_pose,
                        left_hand_pose=left_hand_pose,
                        transl=transl_w,
                        jaw_pose=torch.zeros([1, 3]).to(device),
                        leye_pose=torch.zeros([1, 3]).to(device),
                        reye_pose=torch.zeros([1, 3]).to(device),
                        expression=expr,
                    )
                    joints_w = output_w.joints

                    vertices = output_w.vertices
                    mesh_cam_w = vertices #(bs 10475 3)

            # cap.release()
            pose_lst = np.stack(pose_lst)
            transl_lst = np.stack(transl_lst)
            beta_lst = np.stack(beta_lst)
            lhand_lst = np.stack(lhand_lst)
            rhand_lst = np.stack(rhand_lst)
            expr_lst = np.stack(expr_lst)
            cam_R_lst = np.stack(cam_R_lst)
            cam_T_lst = np.stack(cam_T_lst)
            body_kpts_lst = np.stack(body_kpts_lst)

            motionx_db[vid_name] = {
                "pose": pose_lst,
                "trans": transl_lst,
                "beta": beta_lst,
                "lhand": lhand_lst,
                "rhand": rhand_lst,
                "expr": expr_lst,
                "text": s_text,
                "w_text": w_text_lst,
                "cam_R": cam_R_lst,
                "cam_T": cam_T_lst,
                "body_kpts": body_kpts_lst,
                "intrins": camera_matrix_global.cpu().numpy(),
                "subset": subset,
                "file_name": file_name,
            }

print(f"total {len(motionx_db)} samples")
torch.save(motionx_db, "motionxpp_smplxpose.pth")
