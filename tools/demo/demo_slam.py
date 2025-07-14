import argparse
import os
import subprocess
from glob import glob
from pathlib import Path

import cv2
import hydra
import imageio.v3 as iio
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from einops import einsum, rearrange
from hydra import compose, initialize_config_module
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.genmo.genmo_demo import GENMO_demo
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
from motiondiff.utils.tools import subprocess_run, rsync_file_from_remote

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/demo",
        help="by default to outputs/demo",
    )
    parser.add_argument(
        "-s", "--static_cam", action="store_true", help="If true, skip DPVO"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If true, draw intermediate results"
    )
    args = parser.parse_args()

    if args.img_dir is not None:
        Log.info(f"[Input]: {args.img_dir}")
        video_path = os.path.join(args.output_root, "demo.mp4")
        assert args.video is None
        if Path(video_path).exists():
            args.video = video_path
        else:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            # merge images to video
            all_imgfiles = []
            exts = ["jpg", "jpeg", "png"]
            for ext in exts:
                all_imgfiles.extend(glob(os.path.join(args.img_dir, f"*.{ext}")))
            all_imgfiles = sorted(all_imgfiles)
            frames = []
            for imgfile in all_imgfiles:
                frames.append(cv2.imread(imgfile))
            height, width, _ = frames[0].shape
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
            )
            for frame in tqdm(frames[:300], desc="Merge Images"):
                writer.write(frame)
            writer.release()
            args.video = video_path

    video_path = Path(args.video)
    orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        # cfg = compose(config_name="demo", overrides=overrides)
        cfg = compose(config_name="demo_genmo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if (
        not Path(cfg.video_path).exists()
        or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]
    ):
        if not os.path.exists(cfg.video_path):
            reader = get_video_reader(video_path)
            writer = get_writer(cfg.video_path, fps=30, crf=CRF)
            for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
                writer.write_frame(img)
            writer.close()
            reader.close()

    return cfg, orig_fps


@torch.no_grad()
def run_preprocess(cfg, orig_fps):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy, base_enlarge=1.2
        ).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay, fps=orig_fps)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose[0], 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay, fps=orig_fps)

    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]

    # Get DROID-SLAM results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            cam_int = [
                K_fullimg[0, 0],
                K_fullimg[1, 1],
                K_fullimg[0, 2],
                K_fullimg[1, 2],
            ]
            out_dir = os.path.dirname(paths.slam)
            np.save(f"{out_dir}/cam_int.npy", cam_int)

            # parse video to frames
            video = read_video_np(video_path)
            img_dir = os.path.join(os.path.dirname(cfg.video_path), "imgs")
            os.makedirs(img_dir, exist_ok=True)
            for i, frame in enumerate(video):
                cv2.imwrite(f"{img_dir}/{i:06d}.jpg", frame[..., ::-1])
                i += 1

            cmd = f"python tools/estimate_camera_dir.py --img_dir {img_dir} --out_dir {out_dir}"
            Log.info(f"[DROID-SLAM] {cmd}")
            subprocess.run(cmd, shell=True)

        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")
    else:
        length, width, height = get_video_lwh(cfg.video_path)
        K_fullimg = estimate_K(width, height)
        cam_int = [
            K_fullimg[0, 0],
            K_fullimg[1, 1],
            K_fullimg[0, 2],
            K_fullimg[1, 2],
        ]
        out_dir = os.path.dirname(paths.slam)
        np.save(f"{out_dir}/cam_int.npy", cam_int)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get vit features
    if not Path(paths.vimo_pred).exists():
        extractor = ExtractorVIMO()
        out_dir = os.path.dirname(paths.slam)
        calib = np.load(f"{out_dir}/cam_int.npy")
        pred_smpl = extractor.extract_video_features(
            video_path, bbx_xys, calib, img_ds=1.0
        )
        torch.save(pred_smpl, paths.vimo_pred)
        del extractor
    else:
        Log.info(f"[Preprocess] VIMO features from {paths.vimo_pred}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        t_w2c = torch.zeros(length, 3)
        mean_scale = torch.tensor(1.0)
        scales = torch.ones(length)
        T_w2c = torch.eye(4)[None].repeat(length, 1, 1)
        T_w2c[:, :3, :3] = R_w2c
        T_w2c[:, :3, 3] = t_w2c
    else:
        droid_traj = np.load(cfg.paths.slam, allow_pickle=True).item()
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

    vimo_pred = torch.load(paths.vimo_pred)
    vimo_smpl_params = {
        "pred_cam": vimo_pred["pred_cam"],
        "pred_pose": vimo_pred["pred_pose"],
        "pred_shape": vimo_pred["pred_shape"],
        "pred_trans_c": vimo_pred["pred_trans"],
    }
    vitpose = torch.load(paths.vitpose)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]

    data = {
        # "meta": {
        #     "vid":
        # }
        "meta": [{"vid": Path(cfg.video_path).stem}],
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "f_imgseq": torch.load(paths.vit_features),
        "music_embed": torch.zeros(length, cfg.pipeline.args.encoded_music_dim),
        "vimo_smpl_params": vimo_smpl_params,
        "scales": scales,
        "mean_scale": mean_scale,
        "T_w2c": T_w2c,
        "mask": {
            "has_img_mask": torch.ones(length).bool(),
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }
    return data


def render_incam(cfg, orig_fps):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack(
        [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    color = torch.tensor([0.69019608, 0.39215686, 0.95686275]).cuda()

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=orig_fps, crf=CRF)
    for i, img_raw in tqdm(
        enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"
    ):
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
    video_path = cfg.video_path
    # orig_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    length, width, height = get_video_lwh(video_path)
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
        beta=3.0,
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
    video_path = cfg.video_path
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
    cfg, orig_fps = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg, orig_fps)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        if not os.path.exists(cfg.ckpt_path):
            remote_run_dir = cfg.remote_results_path
            print(f"rsyncing from remote: {remote_run_dir}")
            os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
            rsync_file_from_remote(
                cfg.ckpt_path,
                cfg.remote_results_path,
                "outputs",
                hostname="cs-oci-ord-dc-03",
            )
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(
            f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s"
        )
        torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    orig_fps = int(orig_fps + 0.5)
    render_incam(cfg, orig_fps)
    render_global_o3d(cfg, orig_fps)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal(
            [paths.incam_video, paths.global_video], paths.incam_global_horiz_video
        )
