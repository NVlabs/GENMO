import argparse
import os
import subprocess
from glob import glob
from pathlib import Path

import cv2
import ffmpeg
import hydra
import imageio.v3 as iio
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from einops import einsum, rearrange
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.model.gvhmr.unimfm_demo import UNIMFM_demo
from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    compute_T_ayfz2ay,
    normalize_T_w2c,
)
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import (
    Extractor,
    ExtractorVIMO,
    SLAMModel,
    Tracker,
    VitPoseExtractor,
)
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    concat_videos,
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)
from hmr4d.utils.vis.o3d_render import Settings, create_meshes, get_ground
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras,
    get_global_cameras_static,
    get_global_cameras_static_v2,
    get_ground_params_from_points,
)
from motiondiff.models.mdm.rotation_conversions import quaternion_to_matrix

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def create_text_video(
    output_path,
    text,
    fps=30,
    num_frames=90,
    width=1280,
    height=720,
    font_path="arial.ttf",
    font_size=60,
    text_color=(255, 255, 255),
):
    """
    Create a video with text centered on a black background.
    Text will automatically wrap if it's too long for the screen width.

    Args:
        output_path: Path to save the output video
        text: Text to display
        fps: Frames per second
        num_frames: Total number of frames
        width: Video width
        height: Video height
        font_path: Path to font file
        font_size: Font size
        text_color: RGB tuple for text color
    """
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a black frame with text
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default font if specified font not found
        font = ImageFont.load_default(size=font_size)

    # Create a black image with text
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Calculate max width for text (with some margin)
    max_text_width = width * 0.9

    # Wrap text
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        # Check if adding this word exceeds the max width
        test_line = current_line + " " + word
        test_width = draw.textbbox((0, 0), test_line, font=font)[2]

        if test_width <= max_text_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)  # Add the last line

    # Calculate total text height
    line_height = font_size * 1.2  # Add some line spacing
    total_text_height = len(lines) * line_height

    # Calculate starting y position to center all text vertically
    y_position = (height - total_text_height) // 2

    # Draw each line of text centered horizontally
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x_position = (width - line_width) // 2
        draw.text((x_position, y_position), line, font=font, fill=text_color)
        y_position += line_height

    # Convert PIL Image to OpenCV format
    frame = np.array(img)
    # Convert RGB to BGR (OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to video multiple times
    for _ in range(num_frames):
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_list", nargs="+", help="Input list of videos or text files"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/demo",
        help="by default to outputs/demo",
    )
    parser.add_argument("--repeat", type=int, help="number of input_list repeat")
    # parser.add_argument(
    #     "-s1", "--static_cam1", action="store_true", help="If true, skip DPVO"
    # )
    # parser.add_argument(
    #     "-s2", "--static_cam2", action="store_true", help="If true, skip DPVO"
    # )
    parser.add_argument(
        "--verbose", action="store_true", help="If true, draw intermediate results"
    )
    args = parser.parse_args()

    if args.repeat is not None:
        args.input_list = args.input_list * args.repeat

    input_list = []
    video1_name = None
    video1_width = None
    video1_height = None
    for mid, input_path in enumerate(args.input_list):
        if input_path.endswith(".txt"):
            with open(input_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        vname = Path(input_path).stem
                        input_list.append(
                            {
                                "mid": mid,
                                "vname": vname,
                                "type": "text",
                                "caption": line,
                            }
                        )
        elif input_path.endswith(".mp4"):
            video_path = Path(input_path)
            orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            length, width, height = get_video_lwh(video_path)
            vname = video_path.stem
            if video1_name is None:
                video1_name = vname
                video1_width = width
                video1_height = height
            input_list.append(
                {
                    "mid": mid,
                    "vname": vname,
                    "type": "video",
                    "video": video_path,
                    "orig_fps": orig_fps,
                    "length": length,
                    "width": width,
                    "height": height,
                }
            )
        elif os.path.isdir(input_path):
            video_path = os.path.join(args.output_root, f"demo{mid}.mp4")
            if Path(video_path).exists():
                orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                length, width, height = get_video_lwh(video_path)
                vname = video_path.stem
                if video1_name is None:
                    video1_name = vname
                    video1_width = width
                    video1_height = height
                input_list.append(
                    {
                        "mid": mid,
                        "vname": vname,
                        "type": "video",
                        "video": video_path,
                        "orig_fps": orig_fps,
                        "length": length,
                        "width": width,
                        "height": height,
                    }
                )
            else:
                os.makedirs(
                    os.path.dirname(video_path) + f"_imgs_{vname}", exist_ok=True
                )
                # merge images to video
                all_imgfiles = []
                exts = ["jpg", "jpeg", "png"]
                for ext in exts:
                    all_imgfiles.extend(glob(os.path.join(input_path, f"*.{ext}")))
                all_imgfiles = sorted(all_imgfiles)
                frames = []
                for imgfile in all_imgfiles:
                    frames.append(cv2.imread(imgfile))
                height, width, _ = frames[0].shape
                writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
                )
                for frame in tqdm(frames, desc="Merge Images"):
                    writer.write(frame)
                writer.release()
                vname = video_path.stem
                length, width, height = get_video_lwh(video_path)
                input_list.append(
                    {
                        "mid": mid,
                        "vname": vname,
                        "type": "video",
                        "video": video_path,
                        "orig_fps": 30,
                        "length": length,
                        "width": width,
                        "height": height,
                    }
                )

            if video1_name is None:
                video1_name = Path(video_path).stem
    Log.info(f"[Input]: {input_list}")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video1_name={video1_name}",
            f"video1_width={video1_width}",
            f"video1_height={video1_height}",
            # f"video2_name={video2_path.stem}",
            # f"static_cam1={args.static_cam1}",
            # f"static_cam2={args.static_cam2}",
            # f"verbose={args.verbose}",
            # f"text1={args.text1}",
            # f"orig_fps1={orig_fps1}",
            # f"orig_fps2={orig_fps2}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        # cfg = compose(config_name="demo", overrides=overrides)
        cfg = compose(config_name="demo_unimfm_mix", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    # Path(cfg.preprocess_dir + "_1").mkdir(parents=True, exist_ok=True)
    # Path(cfg.preprocess_dir + "_2").mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    video_paths = {}
    for mid, input in enumerate(input_list):
        vname = input["vname"]
        if input["type"] == "video":
            video_path = input["video"]
            # OmegaConf.update(cfg, f"video{mid+1}_path", f"{cfg.output_dir}/0_input_video{mid+1}.mp4")
            video_paths[f"video{mid + 1}_path"] = (
                f"{cfg.output_dir}/0_input_video{vname}.mp4"
            )
            Log.info(
                f"[Copy Video{mid}] {video_path} -> {video_paths[f'video{mid + 1}_path']}"
            )
            if (
                not Path(video_paths[f"video{mid + 1}_path"]).exists()
                or get_video_lwh(video_path)[0]
                != get_video_lwh(video_paths[f"video{mid + 1}_path"])[0]
            ):
                if not os.path.exists(video_paths[f"video{mid + 1}_path"]):
                    reader = get_video_reader(video_path)
                    writer = get_writer(
                        video_paths[f"video{mid + 1}_path"], fps=30, crf=CRF
                    )
                    for img in tqdm(
                        reader, total=get_video_lwh(video_path)[0], desc=f"Copy"
                    ):
                        writer.write_frame(img)
                    writer.close()
                    reader.close()

    return cfg, input_list, video_paths


@torch.no_grad()
def run_preprocess_text(cfg, input_item, vid=2):
    Log.info(f"[Preprocess] Start text {vid}!")

    text = input_item["caption"]

    text_length = 300
    return_data = {
        "meta": {
            "vid": "text",
            "caption": [text],
        },
        "length": torch.tensor(text_length),
        "bbx_xys": torch.zeros(text_length, 3),
        "K_fullimg": torch.eye(3).repeat(text_length, 3, 3),
        "f_imgseq": torch.zeros(text_length, 1024),
        "kp2d": torch.zeros(text_length, 17, 3),
        "cam_angvel": torch.zeros_like(
            compute_cam_angvel(torch.eye(3)[None].repeat(text_length, 1, 1))
        ),
        "cam_tvel": torch.zeros(text_length, 3),
        "R_w2c": torch.eye(3).reshape(1, 3, 3).repeat(text_length, 1, 1),
        "T_w2c": torch.eye(4).reshape(1, 4, 4).repeat(text_length, 1, 1),
        "gt_T_w2c": torch.eye(4).reshape(1, 4, 4).repeat(text_length, 1, 1),
        "gender": "neutral",
        "caption": text,
        "has_text": True,
        "mask": {
            "valid": torch.ones(text_length),
            "vitpose": False,
            "bbx_xys": False,
            "f_imgseq": False,
            "spv_incam_only": False,
        },
    }
    return return_data


@torch.no_grad()
def run_preprocess(cfg, input_item, vid=1):
    vname = input_item["vname"]
    Log.info(f"[Preprocess] Start {vid} {vname}!")
    Path(cfg.preprocess_dir + f"_{vname}").mkdir(parents=True, exist_ok=True)
    tic = Log.time()
    # video_path = video_path
    video_path = f"{cfg.output_dir}/0_input_video{vname}.mp4"
    bbx_path = f"{cfg.output_dir}/preprocess_{vname}/bbx.pt"
    bbx_xyxy_video_overlay_path = (
        f"{cfg.output_dir}/preprocess_{vname}/bbx_xyxy_video_overlay.mp4"
    )
    vitpose_path = f"{cfg.output_dir}/preprocess_{vname}/vitpose.pt"
    vitpose_video_overlay_path = (
        f"{cfg.output_dir}/preprocess_{vname}/vitpose_video_overlay.mp4"
    )
    slam_path = f"{cfg.output_dir}/preprocess_{vname}/camera.npy"
    static_cam = False
    vimo_pred_path = f"{cfg.output_dir}/preprocess_{vname}/vimo_pred.pt"
    vit_features_path = f"{cfg.output_dir}/preprocess_{vname}/vit_features.pt"
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(bbx_path).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy, base_enlarge=1.2
        ).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, bbx_path)
        del tracker
    else:
        bbx_xys = torch.load(bbx_path)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {bbx_path}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(bbx_path)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, bbx_xyxy_video_overlay_path)

    # Get VitPose
    if not Path(vitpose_path).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, vitpose_path)
        del vitpose_extractor
    else:
        vitpose = torch.load(vitpose_path)
        Log.info(f"[Preprocess] vitpose from {vitpose_path}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, vitpose_video_overlay_path)

    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]

    # Get DROID-SLAM results
    if not static_cam:  # use slam to get cam rotation
        if not Path(slam_path).exists():
            length, width, height = get_video_lwh(video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            cam_int = [
                K_fullimg[0, 0],
                K_fullimg[1, 1],
                K_fullimg[0, 2],
                K_fullimg[1, 2],
            ]
            out_dir = os.path.dirname(slam_path)
            np.save(f"{out_dir}/cam_int.npy", cam_int)

            # parse video to frames
            video = read_video_np(video_path)
            img_dir = os.path.join(os.path.dirname(video_path), f"imgs_{vname}")
            os.makedirs(img_dir, exist_ok=True)
            for i, frame in enumerate(video):
                cv2.imwrite(f"{img_dir}/{i:06d}.jpg", frame[..., ::-1])
                i += 1

            cmd = f"python tools/estimate_camera_dir.py --img_dir {img_dir} --out_dir {out_dir}"
            Log.info(f"[DROID-SLAM] {cmd}")
            subprocess.run(cmd, shell=True)

        else:
            Log.info(f"[Preprocess] slam results from {slam_path}")
    else:
        length, width, height = get_video_lwh(video_path)
        K_fullimg = estimate_K(width, height)
        intrinsics = convert_K_to_K4(K_fullimg)
        cam_int = [
            K_fullimg[0, 0],
            K_fullimg[1, 1],
            K_fullimg[0, 2],
            K_fullimg[1, 2],
        ]
        out_dir = os.path.dirname(slam_path)
        np.save(f"{out_dir}/cam_int.npy", cam_int)

    # Get vit features
    if not Path(vit_features_path).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, vit_features_path)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {vit_features_path}")

    # Get vit features
    if not Path(vimo_pred_path).exists():
        extractor = ExtractorVIMO()
        out_dir = os.path.dirname(slam_path)
        calib = np.load(f"{out_dir}/cam_int.npy")
        pred_smpl = extractor.extract_video_features(
            video_path, bbx_xys, calib, img_ds=1.0
        )
        torch.save(pred_smpl, vimo_pred_path)
        del extractor
    else:
        Log.info(f"[Preprocess] VIMO features from {vimo_pred_path}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg, input_item, fps=30, vid=1):
    # paths = cfg.paths
    vname = input_item["vname"]
    video_path = f"{cfg.output_dir}/0_input_video{vname}.mp4"
    bbx_path = f"{cfg.output_dir}/preprocess_{vname}/bbx.pt"
    bbx_xyxy_video_overlay_path = (
        f"{cfg.output_dir}/preprocess_{vname}/bbx_xyxy_video_overlay.mp4"
    )
    vitpose_path = f"{cfg.output_dir}/preprocess_{vname}/vitpose.pt"
    vitpose_video_overlay_path = (
        f"{cfg.output_dir}/preprocess_{vname}/vitpose_video_overlay.mp4"
    )
    slam_path = f"{cfg.output_dir}/preprocess_{vname}/camera.npy"
    static_cam = False
    vimo_pred_path = f"{cfg.output_dir}/preprocess_{vname}/vimo_pred.pt"
    vit_features_path = f"{cfg.output_dir}/preprocess_{vname}/vit_features.pt"

    length, width, height = get_video_lwh(video_path)
    if static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
        mean_scale = torch.tensor(1.0)
        scales = torch.ones(length)
        T_w2c = torch.eye(4)[None].repeat(length, 1, 1)
        T_w2c[:, :3, :3] = R_w2c
        T_w2c[:, :3, 3] = t_w2c
    else:
        droid_traj = np.load(slam_path, allow_pickle=True).item()
        R_c2w = torch.from_numpy(droid_traj["pred_cam_R"]).float()
        t_c2w = torch.from_numpy(droid_traj["pred_cam_T"]).float()
        scales = torch.from_numpy(droid_traj["all_scales"]).float()
        mean_scale = droid_traj["scale"]
        T_c2w = torch.eye(4)[None].repeat(length, 1, 1).to(R_c2w)
        T_c2w[:, :3, :3] = R_c2w
        T_c2w[:, :3, 3] = t_c2w
        T_w2c = T_c2w.inverse()
        T_w2c = normalize_T_w2c(T_w2c)

        R_w2c = T_w2c[:, :3, :3]
        t_w2c = T_w2c[:, :3, 3]

    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

    vimo_pred = torch.load(vimo_pred_path)
    vimo_smpl_params = {
        "pred_cam": vimo_pred["pred_cam"],
        "pred_pose": vimo_pred["pred_pose"],
        "pred_shape": vimo_pred["pred_shape"],
        "pred_trans_c": vimo_pred["pred_trans"],
    }
    vitpose = torch.load(vitpose_path)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]

    data = {
        # "meta": {
        #     "vid":
        # }
        "length": torch.tensor(length),
        "bbx_xys": torch.load(bbx_path)["bbx_xys"],
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "f_imgseq": torch.load(vit_features_path),
        "music_embed": torch.zeros(length, cfg.pipeline.args.encoded_music_dim),
        # "vimo_smpl_params": vimo_smpl_params,
        "scales": scales,
        "mean_scale": mean_scale,
        "T_w2c": T_w2c,
    }

    def interpolate(data_vec, orig_fps, new_fps):
        # data_vec: (L, ...)
        # return: (L_new, ...)
        L = data_vec.shape[0]
        L_new = int(L * new_fps / orig_fps)
        # put L to the last dimension
        if data_vec.ndim > 2:
            shape = data_vec.shape
            data_vec = data_vec.reshape(data_vec.shape[0], -1).contiguous()
        else:
            shape = data_vec.shape
        data_vec = data_vec.transpose(0, 1).contiguous()
        data_vec_new = torch.nn.functional.interpolate(
            data_vec[None], size=L_new, mode="linear", align_corners=False
        )[0]
        data_vec_new = data_vec_new.transpose(0, 1).contiguous()

        if len(shape) > 2:
            data_vec_new = data_vec_new.reshape(-1, *shape[1:])
        return data_vec_new

    # interpolate to 30 fps
    if fps != 30:
        data["f_imgseq"] = interpolate(data["f_imgseq"], fps, 30)
        data["bbx_xys"] = interpolate(data["bbx_xys"], fps, 30)
        data["kp2d"] = interpolate(data["kp2d"], fps, 30)
        data["K_fullimg"] = interpolate(data["K_fullimg"], fps, 30)
        data["cam_angvel"] = interpolate(data["cam_angvel"], fps, 30)
        data["cam_tvel"] = interpolate(data["cam_tvel"], fps, 30)
        data["R_w2c"] = interpolate(data["R_w2c"], fps, 30)
        data["T_w2c"] = interpolate(data["T_w2c"], fps, 30)
        data["length"] = torch.tensor(data["f_imgseq"].shape[0])
        data["music_embed"] = interpolate(data["music_embed"], fps, 30)

    return data


def render_incam(cfg, vid_slice, vname, vid=1):
    incam_video_path = Path(f"{cfg.output_dir}/1_incam{vid}.mp4")
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred_full = torch.load(cfg.paths.hmr4d_results)
    pred = {"smpl_params_incam": pred_full["smpl_params_incam"]}
    start_idx, end_idx, fps = vid_slice[vid]
    for k in pred_full.keys():
        if k not in ["smpl_params_incam", "K_fullimg"]:
            continue
        if isinstance(pred_full[k], dict):
            pred[k] = {}
            for kk in pred_full[k].keys():
                pred[k][kk] = pred_full[k][kk][start_idx:end_idx]
        else:
            pred[k] = pred_full[k][start_idx:end_idx]

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    # -- rendering code -- #
    video_path = f"{cfg.output_dir}/0_input_video{vname}.mp4"
    video_30fps_path = str(video_path).replace(".mp4", "_30fps.mp4")

    # convert to 30 fps
    if fps != 30:
        stream = ffmpeg.input(video_path).filter("setpts", f"{30.0 / fps}*PTS")
        output = ffmpeg.output(stream, video_30fps_path)
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    else:
        video_30fps_path = video_path

    length, width, height = get_video_lwh(video_30fps_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_30fps_path)  # (F, H, W, 3), uint8, numpy
    bbx_path = f"{cfg.output_dir}/preprocess_{vname}/bbx.pt"
    bbx_xys_render = torch.load(bbx_path)["bbx_xys"]
    color = torch.ones(3).float().cuda() * 0.8
    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).cuda()
    color[0] = color_purple[0]
    color[1] = color_purple[1]
    color[2] = color_purple[2]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    assert abs(get_video_lwh(video_30fps_path)[0] - len(verts_incam)) < 10, (
        f"Video length mismatch: {get_video_lwh(video_30fps_path)[0]} != {len(verts_incam)}"
    )
    for i, img_raw in tqdm(
        enumerate(reader),
        total=get_video_lwh(video_30fps_path)[0],
        desc=f"Rendering Incam",
    ):
        if i >= verts_incam.shape[0]:
            break
        img = renderer.render_mesh(
            verts_incam[i].cuda(), img_raw, [color[0], color[1], color[2]]
        )

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global_o3d(cfg, orig_fps):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(
        "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
    ).cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
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

    verts_glob_list = move_to_start_point_face_z(pred_ay_verts, J_regressor)
    joints_glob_list = einsum(J_regressor, verts_glob_list, "j v, l v i -> l j i")
    length = verts_glob_list.shape[0]

    # -- rendering code -- #
    # video_path = cfg.video1_path
    # orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    # length, width, height = get_video_lwh(video_path)
    width = cfg.video1_width
    height = cfg.video1_height
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens
    device = verts_glob_list.device
    # global_video_path = f"out/{vis_type}_video/{fname}.mp4"
    os.makedirs(os.path.dirname(global_video_path), exist_ok=True)
    writer = get_writer(global_video_path, fps=orig_fps, crf=CRF)

    mat_settings = Settings()

    color_purple = torch.tensor([0.69019608, 0.39215686, 0.95686275]).to(device)
    color_green = torch.tensor([0.46666667, 0.90196078, 0.74901961]).to(device)
    color_light_purple = torch.tensor([1.0, 0.65490196, 0.95294118]).to(device)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(
        joints_glob_list[:, 0], verts_glob_list
    )
    scale = max(scale, 3)
    ground_geometry = get_ground(scale * 1.5, cx, cz)
    # color = torch.ones(3).float().cuda() * 0.8

    T, V, _ = verts_glob_list.shape

    position, target, up = get_global_cameras_static_v2(
        # verts_list[0].cpu(),
        verts_glob_list.cpu().clone(),
        beta=1.5,
        # beta=4.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    trans_mat_box = mat_settings._materials[Settings.Transparency]
    lit_mat_box = mat_settings._materials[Settings.LIT]

    colors = color_purple[None, :].repeat(T, 1)
    # colors[:, 0] = torch.linspace(color_green[0], color_purple[0], T)
    # colors[:, 1] = torch.linspace(color_green[1], color_purple[1], T)
    # colors[:, 2] = torch.linspace(color_green[2], color_purple[2], T)

    colors_trans = torch.zeros_like(colors)
    colors_trans[:, 0] = torch.linspace(color_green[0], color_light_purple[0], T)
    colors_trans[:, 1] = torch.linspace(color_green[1], color_light_purple[1], T)
    colors_trans[:, 2] = torch.linspace(color_green[2], color_light_purple[2], T)
    faces = torch.from_numpy(faces_smpl.astype("int")).to(device)

    # colors = torch.stack([torch.from_numpy(color_rgb[i % len(color_rgb)]).float().cuda() for i in range(len(verts_glob_list))], dim=0)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    # renderer.scene.camera.set_projection(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height, 0.1, 100.0)
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )

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
        img = renderer.render_to_image()

        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat_box)
        # mesh = create_meshes(verts[i], faces_list[i], colors[t])
        # renderer.scene.add_geometry(f"mesh_{i}_{t}", mesh, mat)

        img = renderer.render_to_image()
        # import ipdb; ipdb.set_trace()
        # o3d.io.write_image("out/tmp.png", img)
        # img = cv2.imread("out/tmp.png")
        writer.write_frame(np.array(img))
    writer.close()
    print(f"Saved to {global_video_path}")


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(
        "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
    ).cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    def move_to_start_point_face_z(verts):
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
        return verts, T_ay2ayfz

    verts_glob, T_ay2ayfz = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    # global_R, global_T, global_lights = get_global_cameras_static(
    #     verts_glob.cpu(),
    #     beta=2.0,
    #     cam_height_degree=20,
    #     target_center_height=1.0,
    # )
    global_R, global_T, global_lights = get_global_cameras(
        verts_glob.cpu(),
    )
    # pred_T_w2c = pred["net_outputs"]["pred_T_w2c"].to(T_ay2ayfz)
    # T_w2c = (T_ay2ayfz @ pred_T_w2c.inverse()).inverse()

    # # pred_T_c2w = pred_T_w2c.inverse()
    # global_R = T_w2c[:, :3, :3]
    # global_T = T_w2c[:, :3, 3]

    # -- rendering code -- #
    video_path = video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]], color[None], cameras, global_lights
        )
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg, input_list, video_paths = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    # ===== Preprocess and save to disk ===== #
    data_list = []
    start_video_height = None
    start_video_width = None
    share_K_fullimg = None
    for mid, input in enumerate(input_list):
        if input["type"] == "video":
            run_preprocess(cfg, input, vid=mid + 1)
            orig_fps = input["orig_fps"]
            data_video = load_data_dict(cfg, input, fps=orig_fps, vid=mid + 1)
            data_video["type"] = "video"
            data_video["vname"] = input["vname"]
            data_video["fps"] = orig_fps
            data_list.append(data_video)
            if start_video_height is None:
                start_video_height = input["height"]
                start_video_width = input["width"]
                share_K_fullimg = data_video["K_fullimg"][:1]
        elif input["type"] == "text":
            data_text = run_preprocess_text(cfg, input_item=input, vid=mid + 1)
            data_text["type"] = "text"
            data_text["vname"] = input["vname"]
            data_text["fps"] = 30
            data_list.append(data_text)

            # generate text video
            text_video_path = f"{cfg.output_dir}/0_input_text{mid + 1}.mp4"
            Log.info("[Generate Text Video]")
            text_length = 300
            # length, width, height = get_video_lwh(cfg.video1_path)
            create_text_video(
                text_video_path,
                input["caption"],
                fps=30,
                num_frames=text_length,
                width=start_video_width,
                height=start_video_height,
                font_size=int(min(start_video_width, start_video_height) * 0.1),
            )

    # merge data
    data = dict()
    multi_text_data = {
        "vid": [],
        "caption": [],
        "text_ind": [],
        "window_start": [],
        "window_end": [],
    }
    vid_slice = dict()
    tot_length = sum([data_item["length"] for data_item in data_list])
    eval_gen_only_mask = torch.zeros(tot_length)
    current_ind = 0
    for mid, data_item in enumerate(data_list):
        if data_item["type"] == "video":
            pass
        elif data_item["type"] == "text":
            data_item["K_fullimg"] = share_K_fullimg.repeat(data_item["length"], 1, 1)

            multi_text_data["vid"].append(f"text{mid + 1}")
            multi_text_data["caption"].append(data_item["caption"])
            multi_text_data["text_ind"].append(mid)
            multi_text_data["window_start"].append(current_ind / tot_length)
            multi_text_data["window_end"].append(
                (current_ind + data_item["length"]) / tot_length
            )
            eval_gen_only_mask[current_ind : current_ind + data_item["length"]] = 1

        print(current_ind, current_ind + int(data_item["length"]), data_item["fps"])
        vid_slice[mid + 1] = (
            current_ind,
            current_ind + int(data_item["length"]),
            data_item["fps"],
        )
        current_ind += int(data_item["length"])

    multi_text_data["window_start"] = torch.tensor(multi_text_data["window_start"])
    multi_text_data["window_end"] = torch.tensor(multi_text_data["window_end"])
    data["meta"] = [
        {
            "vid1": cfg.video1_name,
            "caption": "",
            "multi_text_data": multi_text_data,
        }
    ]
    data["length"] = tot_length
    data["caption"] = data_text["caption"]
    data["bbx_xys"] = torch.cat([item["bbx_xys"] for item in data_list], dim=0)
    data["kp2d"] = torch.cat([item["kp2d"] for item in data_list], dim=0)
    data["K_fullimg"] = torch.cat([item["K_fullimg"] for item in data_list], dim=0)
    data["cam_angvel"] = torch.cat([item["cam_angvel"] for item in data_list], dim=0)
    data["cam_tvel"] = torch.cat([item["cam_tvel"] for item in data_list], dim=0)
    data["f_imgseq"] = torch.cat([item["f_imgseq"] for item in data_list], dim=0)
    data["R_w2c"] = torch.cat([item["R_w2c"] for item in data_list], dim=0)
    data["T_w2c"] = torch.cat([item["T_w2c"] for item in data_list], dim=0)

    data["eval_gen_only_mask"] = eval_gen_only_mask

    debug = False
    if debug:
        data = data_text
        data["meta"] = [
            {
                "vid1": cfg.video1_name,
                "caption": cfg.text1,
                "eval_gen_only": True,
                # "multi_text_data": multi_text_data,
            }
        ]
    static_cam = cfg.static_cam1 and cfg.static_cam2
    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(
            f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s"
        )
        torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    # orig_fps = int(cfg.orig_fps1 + 0.5)
    in_video_paths = []
    for mid, data_item in enumerate(data_list):
        if data_item["type"] == "video":
            render_incam(cfg, vid_slice, vname=data_item["vname"], vid=mid + 1)
            incam_video_path = f"{cfg.output_dir}/1_incam{mid + 1}.mp4"
            in_video_paths.append(incam_video_path)
        elif data_item["type"] == "text":
            text_video_path = f"{cfg.output_dir}/0_input_text{mid + 1}.mp4"
            in_video_paths.append(text_video_path)
    render_global_o3d(cfg, 30)

    concat_videos(cfg, paths.incam_video, in_video_paths)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal(
            [paths.incam_video, paths.global_video], paths.incam_global_horiz_video
        )
