import os

import torch
from tqdm import tqdm

from motiondiff.utils.vis_scenepic import ScenepicVisualizer

motion_files = torch.load("inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3.pth")

device = torch.device("cuda")
sp_visualizer = ScenepicVisualizer(
    "/home/jiefengl/git/physdiff_megm/data/smpl_data", device=device
)
smpl_dict = sp_visualizer.smpl_dict

aligned_motion_files = {}

tot_len = 0
for idx, vid in tqdm(enumerate(motion_files)):
    motion_data = motion_files[vid]
    smpl_layer = smpl_dict["neutral"]

    body_pose = motion_data["pose"].reshape(-1, 21, 3)
    global_orient = motion_data["global_orient"].reshape(-1, 1, 3)
    beta = motion_data["beta"]
    trans = motion_data["trans"]
    R_w2c = motion_data["cam_R"]
    t_w2c = motion_data["cam_T"]
    T_w2c = torch.eye(4)[None].repeat(len(R_w2c), 1, 1)
    T_w2c[:, :3, :3] = R_w2c
    T_w2c[:, :3, 3] = t_w2c

    T_c2w = torch.linalg.inv(T_w2c)
    t_c2w = T_c2w[:, :3, 3]
    if body_pose.shape[0] < 3:
        import ipdb

        ipdb.set_trace()
    if body_pose.shape[0] < 30:
        continue
    tot_len += body_pose.shape[0]
    pose_pad = torch.cat([body_pose, torch.zeros_like(body_pose[:, :2])], dim=1)
    smpl_output = smpl_layer(
        betas=beta.to(device),
        global_orient=global_orient.to(device),
        body_pose=pose_pad.to(device),
        transl=trans.to(device),
        orig_joints=True,
        pose2rot=True,
    )
    j3d = smpl_output.joints.detach()

    # put the person on the ground by -min(z)
    ground_z = j3d[..., 2].flatten(-2).min(dim=-1)[0].cpu()  # (B,)  Minimum z value
    offset_xy = trans[:1].clone()
    offset_xy[..., 2] = ground_z

    trans = trans - offset_xy
    t_c2w = t_c2w - offset_xy
    T_c2w[:, :3, 3] = t_c2w
    T_w2c = torch.linalg.inv(T_c2w)
    R_w2c = T_w2c[:, :3, :3]
    t_w2c = T_w2c[:, :3, 3]

    motion_data["cam_R"] = R_w2c
    motion_data["cam_T"] = t_w2c
    motion_data["trans"] = trans
    aligned_motion_files[vid] = motion_data

    # smpl_output = smpl_layer(
    #     betas=beta.to(device),
    #     global_orient=global_orient.to(device),
    #     body_pose=pose_pad.to(device),
    #     transl=trans.to(device),
    #     orig_joints=True,
    #     pose2rot=True,
    # )
    # j3d = smpl_output.joints.detach()

    # smpl_seq = {
    #     "text": '',
    #     "joints_pos": j3d,
    #     # "joints_pos": torch.from_numpy(data_jts[:, :24]).float(),
    # }

    # html_file = f"vis/motionxpp_{idx:05d}.html"
    # os.makedirs(os.path.dirname(html_file), exist_ok=True)
    # sp_visualizer.vis_smpl_scene(smpl_seq, html_file)
    # print(vid)
    # import ipdb; ipdb.set_trace()


# print(f"total {len(aligned_motion_files)} samples")
# torch.save(aligned_motion_files, "/mnt/disk3/motion-x++/motionxpp_smplxposev3_aligned.pth")
print(tot_len, tot_len / 30 / 60, tot_len / 30 / 60 / 60)
