import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import time

import numpy as np
import pyvista
import scenepic as sp
import torch
import wandb

from motiondiff.models.common.smpl import SMPL
from motiondiff.utils.torch_transform import (
    angle_axis_to_quaternion,
    quat_apply,
    quat_between_two_vec,
    quaternion_to_angle_axis,
)
from motiondiff.utils.vis import make_checker_board_texture


class TrajActor:
    def __init__(self, scene, name, joint_pos, color="Orange", radius=0.1):
        self.scene = scene
        self.name = name
        self.color = getattr(sp.Colors, color, sp.Colors.Green)

        root_pos = joint_pos[:, 0]  # T x 3
        root_pos[:, 2] = 0.0  # project on floor

        self.traj_mesh = scene.create_mesh(f"{name}_traj", layer_id=f"{self.name}")
        # self.traj_mesh.add_lines(root_pos[:-1], root_pos[1:], self.color)
        for i in range(root_pos.shape[0] - 1):
            self.traj_mesh.add_thickline(
                color=self.color,
                start_point=root_pos[i],
                end_point=root_pos[i + 1],
                start_thickness=radius,
                end_thickness=radius,
            )

    def add_mesh_to_frames(self, sp_frame):
        sp_frame.add_mesh(self.traj_mesh)


class SkeletonActor:
    def __init__(
        self,
        scene,
        name,
        joint_parents,
        joint_color="Yellow",
        bone_color="Green",
        root_color="Yellow",
        joint_constr_color="Cyan",
        joint_radius=0.06,
        bone_radius=0.04,
        joint_constr_radius=0.1,
        opacity=1.0,
    ):
        self.scene = scene
        self.name = name
        self.joint_parents = joint_parents
        self.joint_radius = joint_radius
        self.joint_color = getattr(sp.Colors, joint_color, sp.Colors.Green)
        self.bone_color = getattr(sp.Colors, bone_color, sp.Colors.Green)
        self.root_color = getattr(sp.Colors, root_color, sp.Colors.Green)
        self.joint_constr_color = getattr(
            sp.Colors, joint_constr_color, sp.Colors.Green
        )
        self.bone_radius = bone_radius
        self.joint_meshes = []
        self.joint_constr_meshes = []
        self.bone_meshes = []
        self.bone_pairs = []

        self.floor_img = scene.create_image(image_id="floor")
        self.floor_img.from_numpy(make_checker_board_texture("#81C6EB", "#D4F1F7"))
        self.floor_mesh = scene.create_mesh(texture_id="floor", layer_id="floor")
        self.floor_mesh.add_image(transform=sp.Transforms.Scale(20))

        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = scene.create_mesh(f"{name}_joint{j}", layer_id=f"{self.name}")
            # joint_mesh.add_sphere(color=self.root_color if j==0 else self.joint_color,
            #                       transform=sp.Transforms.scale(joint_radius))
            joint_mesh.add_sphere(
                color=self.root_color if j == 0 else self.joint_color,
                transform=sp.Transforms.scale(joint_radius),
            )

            self.joint_meshes.append(joint_mesh)
            # joint constraints
            joint_constr_mesh = scene.create_mesh(
                f"{name}_joint_constr{j}", layer_id=f"{self.name}"
            )
            joint_constr_mesh.add_sphere(
                color=self.joint_constr_color,
                transform=sp.Transforms.scale(joint_constr_radius),
            )
            self.joint_constr_meshes.append(joint_constr_mesh)
            # bone
            if pa >= 0:
                bone_mesh = scene.create_mesh(
                    f"{name}_bone{j}", layer_id=f"{self.name}"
                )
                bone_mesh.add_cone(
                    color=self.bone_color,
                    transform=sp.Transforms.scale(
                        np.array([1, joint_radius, joint_radius])
                    ),
                )
                self.bone_meshes.append(bone_mesh)
                self.bone_pairs.append((j, pa, bone_mesh))

    def add_mesh_to_frames(self, sp_frame, jpos):
        sp_frame.add_mesh(self.floor_mesh)
        # joint
        for j, pos in enumerate(jpos):
            sp_frame.add_mesh(
                self.joint_meshes[j], transform=sp.Transforms.translate(pos)
            )

        # bone
        vec = []
        for j, pa, _ in self.bone_pairs:
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(
            quat_between_two_vec(torch.tensor([-1.0, 0.0, 0.0]).expand_as(vec), vec)
        ).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)

        for (j, pa, bone_mesh), angle_i, axis_i, dist_i in zip(
            self.bone_pairs, angle, axis, dist
        ):
            transform = sp.Transforms.translate((jpos[pa] + jpos[j]) * 0.5)
            transform = transform @ sp.Transforms.RotationMatrixFromAxisAngle(
                axis_i, angle_i
            )
            transform = transform @ sp.Transforms.Scale(np.array([dist_i, 1, 1]))
            sp_frame.add_mesh(bone_mesh, transform=transform)

    def add_root_mesh_to_frames(self, sp_frame, jpos, path_only=False):
        sp_frame.add_mesh(self.floor_mesh)
        pos = jpos[0]
        if path_only:
            pos[2] = 0.0
        sp_frame.add_mesh(self.joint_meshes[0], transform=sp.Transforms.translate(pos))

    def add_joint_constr_meshes_to_frames(self, sp_frame, jpos, joint_mask):
        # joint constraints
        for j, pos in enumerate(jpos):
            if joint_mask[j].any():
                sp_frame.add_mesh(
                    self.joint_constr_meshes[j], transform=sp.Transforms.translate(pos)
                )

    def add_joints_mesh_to_frames(
        self, sp_frame, jpos, joint_inds=[21, 20, 7, 8], proj_to_floor=False
    ):
        sp_frame.add_mesh(self.floor_mesh)
        for idx in joint_inds:
            cur_pos = jpos[idx]
            if (
                proj_to_floor and idx == 0
            ):  # root TODO: this is hacky, shouldn't assume only the root is projected
                cur_pos = np.copy(cur_pos)
                cur_pos[2] = 0.0
            sp_frame.add_mesh(
                self.joint_meshes[idx], transform=sp.Transforms.translate(cur_pos)
            )


class ScenepicVisualizer:
    def __init__(
        self,
        smpl_model_dir=None,
        device=torch.device("cpu"),
        show_skeleton_jpos=True,
        show_ik_smpl_pose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smpl_dict = {
            "neutral": SMPL(smpl_model_dir, create_transl=False, gender="neutral").to(
                device
            ),
            "male": SMPL(smpl_model_dir, create_transl=False, gender="male").to(device),
            "female": SMPL(smpl_model_dir, create_transl=False, gender="female").to(
                device
            ),
        }
        smpl = self.smpl_dict["male"]
        faces = smpl.faces.copy()
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = smpl.parents.cpu().numpy()
        self.device = device
        self.color_sequences = [
            ["Yellow", "Green", "Teal"],
            ["Yellow", "Red", "Teal"],
            ["Yellow", "Blue", "Teal"],
            ["Yellow", "Purple", "Teal"],
            ["Yellow", "Orange", "Teal"],
        ]
        self.show_skeleton_jpos = show_skeleton_jpos
        self.show_ik_smpl_pose = show_ik_smpl_pose

    def load_default_camera(self):
        return sp.Camera(
            center=(5, 0, 1.5),
            look_at=(0, 0, 0.8),
            up_dir=(0, 0, 1),
            fov_y_degrees=45.0,
            aspect_ratio=1.0,
        )

    def vis_smpl_scene(
        self, smpl_seq=None, html_path=None, window_size=(400, 400), fps=30
    ):
        scene = self.generate_smpl_scene(smpl_seq, window_size=window_size, fps=fps)
        scene.save_as_html(html_path)

    # def vis_smpl_scene(self, smpl_seq=None, video_path=None, window_size=(400, 400)):
    #     scene = self.generate_smpl_scene(smpl_seq, window_size=window_size)
    #     scene.save_image(html_path)

    def generate_smpl_scene(
        self, smpl_seq=None, window_size=None, return_canvas=False, fps=30
    ):
        scene = sp.Scene()
        scene.framerate = fps
        canvas = scene.create_canvas_3d(width=window_size[0], height=window_size[1])

        if "pose" in smpl_seq or "joints_pos" in smpl_seq:  # single person
            smpl_seq = {"skel0": smpl_seq}
        smpl_seq = {
            k: v.copy() for k, v in smpl_seq.items()
        }  # copy to avoid inplace modification

        num_fr = -1
        for i, (skel_name, pose_dict) in enumerate(smpl_seq.items()):
            colors = self.color_sequences[i % len(self.color_sequences)]
            normal_shape_len = {"pose": 2, "trans": 2, "shape": 2, "joints_pos": 3}
            for key in ["pose", "trans", "shape", "joints_pos"]:
                if (
                    key in pose_dict
                    and len(pose_dict[key].shape) > normal_shape_len[key]
                ):
                    pose_dict[key] = pose_dict[key][0]

            if self.show_ik_smpl_pose and "pose" in pose_dict:
                pose_dict["skeleton_fk"] = SkeletonActor(
                    scene,
                    skel_name,
                    self.smpl_joint_parents,
                    joint_color=colors[0],
                    bone_color=colors[1],
                )

                pose = pose_dict["pose"].to(self.device)
                trans = pose_dict["trans"].to(self.device)
                shape = pose_dict["shape"].to(self.device)
                num_fr = max(num_fr, pose.shape[0])

                # print(pose[..., :3].view(-1, 3))
                gender = pose_dict.get("gender", "neutral")
                smpl_motion = self.smpl_dict[gender](
                    global_orient=pose[..., :3],
                    body_pose=pose[..., 3:],
                    betas=shape,
                    root_trans=trans,
                    return_full_pose=True,
                    orig_joints=True,
                )
                smpl_joints = smpl_motion.joints
                pose_dict["joints_fk"] = smpl_joints
                if "offset" in pose_dict:
                    pose_dict["joints_fk"] = pose_dict["joints_fk"] + pose_dict[
                        "offset"
                    ].to(self.device)

            if "joints_pos" in pose_dict:
                num_fr = max(num_fr, pose_dict["joints_pos"].shape[0])
                if "offset" in pose_dict:
                    pose_dict["joints_pos"] = pose_dict["joints_pos"] + pose_dict[
                        "offset"
                    ].to(self.device)
                # the predicted motion
                pose_dict["skeleton_jpos"] = SkeletonActor(
                    scene,
                    f"{skel_name}_jpos",
                    self.smpl_joint_parents,
                    joint_color=colors[0],
                    bone_color=colors[2] if "skeleton_fk" in pose_dict else colors[1],
                    joint_constr_color="Brown" if skel_name == "gt" else "Cyan",
                )
                # dummy joints on the feet to show contacts
                if "foot_contacts" in pose_dict:
                    pose_dict["foot_contacts_jpos"] = SkeletonActor(
                        scene,
                        f"{skel_name}_foot_contacts_jpos",
                        self.smpl_joint_parents,
                        joint_color="Red",
                        bone_color="Yellow",
                        root_color="Red",
                        joint_radius=0.07,
                    )

                # create a different skeleton for each keyframe
                if "keyframes" in pose_dict:
                    keyframes_dense_names = []
                    keyframes_root_traj_names = []
                    keyframes_knownt_idx = []
                    keyframes_knownt_names = []
                    keyframes_unknownt_idx = []
                    keyframes_unknownt_names = []
                    for key_name, keyframe_dict in pose_dict["keyframes"].items():
                        unknownt = keyframe_dict["unknownt"]
                        num_keys = len(keyframe_dict["idx"])
                        is_root_traj = False
                        if not keyframe_dict.get("is_dense", False):
                            if unknownt:
                                keyframes_unknownt_idx.append(keyframe_dict["idx"])
                                keyframes_unknownt_names.append(
                                    np.array([key_name] * num_keys)
                                )
                            else:
                                keyframes_knownt_idx.append(keyframe_dict["idx"])
                                keyframes_knownt_names.append(
                                    np.array([key_name] * num_keys)
                                )
                        else:
                            if keyframe_dict.get("root_proj_to_floor", False):
                                # will draw path on ground instead of showing at every frame
                                keyframes_root_traj_names.append(key_name)
                                is_root_traj = True

                            # dense root trajectories need to handle differently to render at every frame without messing up sparse keyframes
                            keyframes_dense_names.append(key_name)

                        if is_root_traj:
                            # visualization of full trajectory path
                            traj_path = TrajActor(
                                scene,
                                f"{skel_name}_{key_name}",
                                keyframe_dict["joints_pos"].cpu().numpy(),
                                color="Orange",
                                radius=0.03,
                            )
                            keyframe_dict["traj_path"] = traj_path

                        # skeleton visualizing full body of ground truth where keyframe is from
                        skeleton_key_jpos = SkeletonActor(
                            scene,
                            f"{skel_name}_{key_name}",
                            self.smpl_joint_parents,
                            joint_color="Teal" if unknownt else "Orange",
                            bone_color="Purple" if unknownt else "Yellow",
                            root_color="Red",
                        )
                        keyframe_dict["skeleton_jpos"] = skeleton_key_jpos
                        # skeleton highlighting just the subset of joints in the keyframe if not full-body
                        skeleton_key_jinds = None
                        if "joint_inds" in keyframe_dict:
                            skeleton_key_jinds = SkeletonActor(
                                scene,
                                f"{skel_name}_{key_name}_joints",
                                self.smpl_joint_parents,
                                joint_color="Teal" if unknownt else "Orange",
                                bone_color="Purple" if unknownt else "Yellow",
                                root_color="Red",
                                joint_radius=0.12,
                            )
                            keyframe_dict["skeleton_jinds"] = skeleton_key_jinds

                    # order all the keyframes by when they will appear
                    if len(keyframes_knownt_idx) > 0:
                        keyframes_knownt_idx = np.concatenate(keyframes_knownt_idx)
                        keyframes_knownt_names = np.concatenate(keyframes_knownt_names)
                        sort_knownt = np.argsort(keyframes_knownt_idx)
                        keyframes_knownt_idx = keyframes_knownt_idx[sort_knownt]
                        keyframes_knownt_names = keyframes_knownt_names[sort_knownt]
                    if len(keyframes_unknownt_idx) > 0:
                        keyframes_unknownt_idx = np.concatenate(keyframes_unknownt_idx)
                        keyframes_unknownt_names = np.concatenate(
                            keyframes_unknownt_names
                        )
                        sort_unknownt = np.argsort(keyframes_unknownt_idx)
                        keyframes_unknownt_idx = keyframes_unknownt_idx[sort_unknownt]
                        keyframes_unknownt_names = keyframes_unknownt_names[
                            sort_unknownt
                        ]

        if not self.show_skeleton_jpos:
            layer_settings = {
                f"{skel_name}_jpos": {"filled": False}
                for skel_name, pose_dict in smpl_seq.items()
                if "skeleton_jpos" in pose_dict
            }
            canvas.set_layer_settings(layer_settings)
        else:
            layer_settings = {
                f"{skel_name}_jpos": {"filled": True}
                for skel_name, pose_dict in smpl_seq.items()
                if "skeleton_jpos" in pose_dict
            }
            keyframe_settings = dict()
            for i, (skel_name, pose_dict) in enumerate(smpl_seq.items()):
                if "keyframes" in pose_dict:
                    for key_name, keyframe_dict in pose_dict["keyframes"].items():
                        if (
                            "skeleton_jinds" in keyframe_dict
                            and not "traj_path" in keyframe_dict
                        ):
                            # keyframe is on a subset of joints, so we need to turn down opacity of full-body to visualize individual joints
                            keyframe_settings[f"{skel_name}_{key_name}"] = {
                                "filled": True,
                                "opacity": 0.15,
                            }
            # keyframe_settings = {f'{skel_name}_keyframe_jpos': {'filled': True, 'opacity' : 0.15} for skel_name, pose_dict in smpl_seq.items() if 'keyframe_jinds' in pose_dict}
            layer_settings.update(keyframe_settings)
            canvas.set_layer_settings(layer_settings)

        for fr in range(num_fr):
            main_frame = canvas.create_frame()
            main_frame.camera = self.load_default_camera()

            for skel_name, pose_dict in smpl_seq.items():
                if "skeleton_fk" in pose_dict:
                    ind = min(fr, pose_dict["joints_fk"].shape[0] - 1)
                    pose_dict["skeleton_fk"].add_mesh_to_frames(
                        main_frame, pose_dict["joints_fk"][ind].cpu().numpy()
                    )

                if "skeleton_jpos" in pose_dict:
                    ind = min(fr, pose_dict["joints_pos"].shape[0] - 1)

                    if "keyframes" in pose_dict:
                        # visualize next knownt keyframe(s) to hit
                        if len(keyframes_knownt_idx) > 0:
                            ind_diff = keyframes_knownt_idx - ind
                            if (
                                np.sum(ind_diff >= 0) > 0
                            ):  # we're not past the last keyframe
                                viz_ind = np.argmax(ind_diff >= 0)
                                viz_ind_list = np.nonzero(
                                    ind_diff == ind_diff[viz_ind]
                                )[0]
                                for viz_ind in viz_ind_list:
                                    viz_name = keyframes_knownt_names[viz_ind]
                                    viz_ind = keyframes_knownt_idx[viz_ind]
                                    viz_key = pose_dict["keyframes"][viz_name]
                                    key_ind = viz_key[
                                        "idx"
                                    ].index(
                                        viz_ind
                                    )  # keyframe index within this constraint (if multiple)
                                    if "skeleton_jinds" in viz_key:
                                        viz_key[
                                            "skeleton_jinds"
                                        ].add_joints_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][key_ind]
                                            .cpu()
                                            .numpy(),
                                            viz_key["joint_inds"],
                                            viz_key["root_proj_to_floor"],
                                        )
                                    if (
                                        "hide_full_pose" not in viz_key
                                        or not viz_key["hide_full_pose"]
                                    ):
                                        viz_key["skeleton_jpos"].add_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][key_ind]
                                            .cpu()
                                            .numpy(),
                                        )

                        # visualize all unknownt keyframes that have not been passed
                        if len(keyframes_unknownt_idx) > 0:
                            for viz_ind, viz_name in zip(
                                keyframes_unknownt_idx, keyframes_unknownt_names
                            ):
                                if viz_ind > ind:
                                    viz_key = pose_dict["keyframes"][viz_name]
                                    key_ind = viz_key[
                                        "idx"
                                    ].index(
                                        viz_ind
                                    )  # keyframe index within this constraint (if multiple)
                                    if "skeleton_jinds" in viz_key:
                                        viz_key[
                                            "skeleton_jinds"
                                        ].add_joints_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][key_ind]
                                            .cpu()
                                            .numpy(),
                                            viz_key["joint_inds"],
                                            viz_key["root_proj_to_floor"],
                                        )
                                    if (
                                        "hide_full_pose" not in viz_key
                                        or not viz_key["hide_full_pose"]
                                    ):
                                        viz_key["skeleton_jpos"].add_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][key_ind]
                                            .cpu()
                                            .numpy(),
                                        )
                                else:
                                    # we're past current frame
                                    break

                        # always show any dense
                        if len(keyframes_dense_names) > 0:
                            for viz_name in keyframes_dense_names:
                                viz_key = pose_dict["keyframes"][viz_name]
                                ind_diff = np.array(viz_key["idx"]) - ind
                                if (
                                    np.sum(ind_diff >= 0) > 0
                                ):  # we're not past the last keyframe
                                    viz_ind = np.argmax(ind_diff >= 0)
                                    if "skeleton_jinds" in viz_key:
                                        viz_key[
                                            "skeleton_jinds"
                                        ].add_joints_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][viz_ind]
                                            .cpu()
                                            .numpy(),
                                            viz_key["joint_inds"],
                                            viz_key["root_proj_to_floor"],
                                        )
                                    if (
                                        "hide_full_pose" not in viz_key
                                        or not viz_key["hide_full_pose"]
                                    ):
                                        viz_key["skeleton_jpos"].add_mesh_to_frames(
                                            main_frame,
                                            viz_key["joints_pos"][viz_ind]
                                            .cpu()
                                            .numpy(),
                                        )

                        # always show any paths
                        if len(keyframes_root_traj_names) > 0:
                            for viz_name in keyframes_root_traj_names:
                                viz_key = pose_dict["keyframes"][viz_name]
                                viz_key["traj_path"].add_mesh_to_frames(main_frame)

                    # always add current pose
                    pose_dict["skeleton_jpos"].add_mesh_to_frames(
                        main_frame, pose_dict["joints_pos"][ind].cpu().numpy()
                    )

                    if "foot_contacts" in pose_dict:
                        cur_contacts = pose_dict["foot_contacts"][ind]
                        if np.sum(cur_contacts) > 0:
                            foot_inds = np.array(
                                [7, 10, 8, 11]
                            )  # ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
                            contact_inds = foot_inds[cur_contacts]
                            pose_dict["foot_contacts_jpos"].add_joints_mesh_to_frames(
                                main_frame,
                                pose_dict["joints_pos"][ind].cpu().numpy(),
                                contact_inds,
                            )

            # if 'text' in smpl_seq:
            #     label = scene.create_label(text=smpl_seq['text'], color=sp.Colors.White, size_in_pixels=60, offset_distance=0.0, horizontal_align='center', camera_space=True)
            #     main_frame.add_label(label=label, position=[0.0, 1.5, -5.0])
        if return_canvas:
            return scene, canvas
        else:
            return scene


if __name__ == "__main__":
    sp_visualizer = ScenepicVisualizer("data/smpl_data", device="cuda")

    smpl_pose, smpl_trans = torch.load(f"out/smpl_seq.pt")
    ind = 0
    smpl_seq = {
        "pose": smpl_pose[ind],
        "trans": smpl_trans[ind],
        "shape": torch.zeros_like(smpl_pose[ind, :, :10]),
        "gender": "male",
    }
    smpl_seq2 = smpl_seq.copy()
    # smpl_seq2['trans'] = smpl_seq2['trans'].clone()
    # smpl_seq2['trans'][..., 1] += 0.5
    smpl_seq2["offset"] = torch.tensor([0.0, 0.8, 0.0])

    smpl_seq_all = {"skel0": smpl_seq, "skel1": smpl_seq2}

    # smpl_seq1 = {
    #     'pose': torch.zeros((80, 72)),
    #     'trans': torch.zeros((80, 3)),
    #     'shape': torch.zeros((80, 10)),
    #     'gender': 'male'
    # }

    # smpl_seq2 = {
    #     'pose': torch.rand((1, 72)).expand_as(smpl_seq1['pose']),
    #     'trans': torch.zeros((80, 3)),
    #     'shape': torch.zeros((80, 10)),
    #     'gender': 'male'
    # }

    # smpl_seq_interp = {}
    # for key in smpl_seq1:
    #     if key in {'gender'}:
    #         continue
    #     smpl_seq_interp[key] = torch.zeros_like(smpl_seq1[key])
    #     for fr in range(smpl_seq1['pose'].shape[0]):
    #         s = fr / (smpl_seq1['pose'].shape[0] - 1)
    #         smpl_seq_interp[key][[fr]] = smpl_seq1[key][[fr]] * (1 - s) + smpl_seq2[key][[fr]] * s

    sp_visualizer.vis_smpl_scene(smpl_seq_all, "out/test.html")
