import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import copy
import glob
import select
import termios
import textwrap
import time
import tty

import matplotlib.cm as cm
import numpy as np
import pyvista
import torch
from pyvista.plotting import Color
from pywavefront import Wavefront
from scipy.interpolate import CubicSpline, interp1d
from torch.utils.data import DataLoader
from vtk import vtkTransform

from motiondiff.models.common.smpl import SMPL, SMPL_BONE_ORDER_NAMES
from motiondiff.utils.torch_utils import interp_scipy_ndarray

from .torch_transform import (
    angle_axis_to_quaternion,
    quat_apply,
    quat_between_two_vec,
    quaternion_to_angle_axis,
)
from .visualizer3d import Visualizer3D

DEBUG_MODE = False

ActorMapping = {}
Mesh2Actor = {}
pickable_actors = set()

FPS_TEXT = "FPS: %.1f"
SELECTION_TEXT = "Selection mode: %s"

NAME_MAP = {"pred": "Generated", "gt": "Mocap"}

PICKING_NAME_MAP = {
    "actor_joint": "Joint Keyframe",
    "actor_body": "Body Keyframe",
    "surface_root_traj": "Path Following",
    "surface_root_key": "Waypoint",
}


def get_line_angle_axis_dist(start_pos, end_pos):
    vec = end_pos - start_pos
    dist = np.linalg.norm(vec)
    vec = vec / dist
    vec = torch.tensor(vec).float()
    aa = quaternion_to_angle_axis(
        quat_between_two_vec(torch.tensor([0.0, 0.0, 1.0]).expand_as(vec), vec)
    ).numpy()
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / (angle + 1e-6)
    return angle, axis, dist


class SMPLActor:
    def __init__(
        self, pl, verts, faces, uv, color="#FF8A82", visible=True, tex_paths=[]
    ):
        self.pl = pl
        self.verts = verts
        self.face = faces
        self.uv = uv
        self.color = color

        self.tex_list = None
        self.tex_ind = None
        if len(tex_paths) > 0:
            self.tex_list = []
            for tex_pth in tex_paths:
                self.tex_list.append(pyvista.read_texture(tex_pth))
            self.tex_list.append(None)  # option to not render texture last
            self.tex_ind = 0

        self.mesh = pyvista.PolyData(verts, faces)
        if self.tex_list is not None:
            self.mesh.active_t_coords = self.uv
            self.actor = self.pl.add_mesh(
                self.mesh, texture=self.tex_list[self.tex_ind], render=False
            )
        else:
            self.actor = self.pl.add_mesh(
                self.mesh,
                color=color,
                pbr=True,
                metallic=0.0,
                roughness=0.3,
                diffuse=1,
                render=False,
            )
        # self.actor = self.pl.add_mesh(self.mesh, color=color, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        self.set_visibility(visible)

    def update_verts(self, new_verts):
        self.mesh.points[...] = new_verts
        self.mesh.compute_normals(inplace=True)

    def set_opacity(self, opacity):
        self.actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)

    def get_visibility(self):
        return self.actor.visibility

    def set_color(self, color):
        rgb_color = Color(color)
        self.actor.GetProperty().SetColor(rgb_color)

    def next_tex(self):
        if self.tex_list:
            self.tex_ind = (self.tex_ind + 1) % len(self.tex_list)
            cur_vis = self.get_visibility()
            if (
                self.tex_list[self.tex_ind] is None
            ):  # if we're on no texture, need to change mesh
                self.pl.remove_actor(self.actor, render=False)
                self.actor = self.pl.add_mesh(
                    self.mesh,
                    color=self.color,
                    pbr=True,
                    metallic=0.0,
                    roughness=0.3,
                    diffuse=1,
                    render=False,
                )
            elif self.tex_ind == 0:  # back to being textured
                self.pl.remove_actor(self.actor, render=False)
                self.actor = self.pl.add_mesh(
                    self.mesh, texture=self.tex_list[self.tex_ind], render=False
                )
            else:
                self.actor.texture = self.tex_list[self.tex_ind]
            self.set_visibility(cur_vis)


class ActorWrapper:
    def __init__(self, actor, ind, type, mesh, skeleton, color, parent=None):
        self.actor = actor
        self.ind = ind
        self.type = type
        self.mesh = mesh
        self.skeleton = skeleton
        self.color = color
        self.parent = parent
        self.pos = None
        ActorMapping[actor] = self
        Mesh2Actor[id(mesh)] = self
        if type in {"joint", "bone", "joint_constraint"}:
            pickable_actors.add(self)

    @property
    def prop(self):
        return self.actor.prop

    def GetProperty(self):
        return self.actor.GetProperty()

    def SetVisibility(self, flag):
        self.actor.SetVisibility(flag)

    def SetOpacity(self, opacity):
        self.actor.SetOpacity(opacity)

    def SetUserTransform(self, trans):
        self.actor.SetUserTransform(trans)

    def SetPosition(self, pos):
        self.pos = pos

    def reset_picked(self):
        self.prop.color = self.color

    def remove(self, pl):
        pl.remove_actor(self.actor, render=False)
        if type in {"joint", "bone", "joint_constraint"}:
            pickable_actors.discard(self)


class SkeletonActor:
    def __init__(
        self,
        pl,
        joint_parents,
        joint_color="green",
        bone_color="yellow",
        joint_radius=0.03,
        bone_radius=0.02,
        visible=True,
        name="skel1",
    ):
        self.pl = pl
        self.joint_parents = joint_parents
        self.name = name
        self.joint_meshes = []
        self.joint_actors = []
        self.bone_meshes = []
        self.bone_actors = []
        self.bone_pairs = []
        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = pyvista.Sphere(
                radius=joint_radius,
                center=(0, 0, 0),
                theta_resolution=10,
                phi_resolution=10,
            )
            # joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
            joint_actor = self.pl.add_mesh(
                joint_mesh,
                color=joint_color,
                ambient=0.3,
                diffuse=0.5,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
                render=False,
            )
            joint_actor = ActorWrapper(
                joint_actor, j, "joint", joint_mesh, self, joint_color, parent=self
            )
            joint_actor.SetVisibility(False)
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(
                    radius=bone_radius,
                    center=(0, 0, 0),
                    direction=(0, 0, 1),
                    resolution=30,
                )
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(
                    bone_mesh,
                    color=bone_color,
                    ambient=0.3,
                    diffuse=0.5,
                    specular=0.8,
                    specular_power=5,
                    smooth_shading=True,
                    render=False,
                )
                bone_actor = ActorWrapper(
                    bone_actor,
                    len(self.bone_actors),
                    "bone",
                    bone_mesh,
                    self,
                    bone_color,
                    parent=self,
                )
                bone_actor.SetVisibility(False)
                self.bone_meshes.append(bone_mesh)
                self.bone_actors.append(bone_actor)
                self.bone_pairs.append((j, pa))
        self.set_visibility(visible)

    def update_joints(self, jpos):
        # joint
        for actor, pos in zip(self.joint_actors, jpos):
            trans = vtkTransform()
            trans.Translate(*pos)
            actor.SetUserTransform(trans)
            actor.SetPosition(pos)
        # bone
        vec = []
        for actor, (j, pa) in zip(self.bone_actors, self.bone_pairs):
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(
            quat_between_two_vec(torch.tensor([0.0, 0.0, 1.0]).expand_as(vec), vec)
        ).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)

        for actor, (j, pa), angle_i, axis_i, dist_i in zip(
            self.bone_actors, self.bone_pairs, angle, axis, dist
        ):
            trans = vtkTransform()
            trans.Translate(*(jpos[pa] + jpos[j]) * 0.5)
            trans.RotateWXYZ(np.rad2deg(angle_i), *axis_i)
            trans.Scale(1, 1, dist_i)
            actor.SetUserTransform(trans)

    def set_opacity(self, opacity):
        if isinstance(opacity, (int, float)):
            opacity = [opacity] * len(self.joint_actors)
        for i, actor in enumerate(self.joint_actors):
            actor.GetProperty().SetOpacity(opacity[i])
        for i, actor in enumerate(self.bone_actors):
            actor.GetProperty().SetOpacity(opacity[self.bone_pairs[i][0]])

    def set_visibility(self, flag):
        for actor in self.joint_actors:
            actor.SetVisibility(flag)
        for actor in self.bone_actors:
            actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = Color(color)
        for actor in self.joint_actors:
            actor.GetProperty().SetColor(rgb_color)
        for actor in self.jbone_actors:
            actor.GetProperty().SetColor(rgb_color)

    def remove(self):
        for joint in self.joint_actors:
            joint.remove(self.pl)
        for bone in self.bone_actors:
            bone.remove(self.pl)


def compute_root_rot(skel):
    joint_names = ["R_Hip", "L_Hip", "R_Shoulder", "L_Shoulder"]
    joint_idx = [SMPL_BONE_ORDER_NAMES.index(jname) for jname in joint_names]
    r_hip, l_hip, r_shoulder, l_shoulder = joint_idx

    # only care about 2d (x,y)
    skel2d = skel * np.array([[1, 1, 0]])
    # average based on hips and shoulders
    across1 = skel2d[r_hip] - skel2d[l_hip]
    across2 = skel2d[r_shoulder] - skel2d[l_shoulder]
    across = across1 + across2
    across = across / np.linalg.norm(across)

    forward = np.cross(np.array([0, 0, 1]), across)
    forward = forward / np.linalg.norm(forward)
    return forward


class JointConstraints:
    def __init__(
        self,
        pos,
        frame,
        joint_actor,
        visualizer,
        color="purple",
        size=0.04,
        root_color="purple",
    ):
        self.pos = pos
        self.frame = frame
        self.joint_actor = joint_actor
        self.ind = joint_actor.ind
        self.jname = SMPL_BONE_ORDER_NAMES[self.ind]
        self.visualizer = visualizer
        self.pl = visualizer.pl
        self.color = color
        self.mesh = pyvista.Sphere(
            radius=size, center=pos, theta_resolution=10, phi_resolution=10
        )
        actor = self.pl.add_mesh(
            self.mesh,
            color=color,
            ambient=0.3,
            diffuse=0.5,
            specular=0.8,
            specular_power=5,
            smooth_shading=True,
            render=False,
        )
        self.actor = ActorWrapper(
            actor, self.ind, "joint_constraint", self.mesh, self, color
        )
        if DEBUG_MODE:
            self.label = self.pl.add_point_labels(
                pos,
                [f"F{frame} {self.jname} ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"],
            )

        # save root pos/heading so can convert other joints to local frame later
        # assumes skeleton is currently posed for the timestep "frame"
        parent_skel = np.array(
            [
                joint_actor.parent.joint_actors[ji].pos
                for ji in range(len(joint_actor.parent.joint_actors))
            ]
        )
        root_idx = SMPL_BONE_ORDER_NAMES.index("Pelvis")
        self.root_pos = parent_skel[root_idx]
        self.root_rot = compute_root_rot(parent_skel)  # heading vector
        # show that root has been constrained too
        self.root_mesh = pyvista.Sphere(
            radius=size, center=self.root_pos, theta_resolution=10, phi_resolution=10
        )
        root_actor = self.pl.add_mesh(
            self.root_mesh,
            color=root_color,
            ambient=0.3,
            diffuse=0.5,
            specular=0.8,
            specular_power=5,
            smooth_shading=True,
            render=False,
        )
        self.root_actor = ActorWrapper(
            root_actor, root_idx, "joint_constraint", self.root_mesh, self, root_color
        )
        if DEBUG_MODE:
            self.root_label = self.pl.add_point_labels(
                self.root_pos,
                [
                    f"F{frame} {'Root'} ({self.root_pos[0]:.2f}, {self.root_pos[1]:.2f}, {self.root_pos[2]:.2f})"
                ],
            )

    def to_dict(self):
        return {
            "joint_idx": self.ind,
            "joint_name": self.jname,
            "frame": self.frame,
            "joint_pos": self.pos,
            "root_pos": self.root_pos,
            "root_rot": self.root_rot,
        }

    def remove(self):
        self.actor.remove(self.pl)
        self.root_actor.remove(self.pl)
        if DEBUG_MODE:
            self.pl.remove_actor(self.label, render=False)
            self.pl.remove_actor(self.root_label, render=False)


class BodyConstraints:
    def __init__(
        self,
        frame,
        skel_actor,
        visualizer,
        joint_color="purple",
        bone_color="purple",
        joint_radius=0.04,
        bone_radius=0.03,
    ):
        self.frame = frame
        self.skel_actor = skel_actor
        self.visualizer = visualizer
        self.pl = visualizer.pl

        # assumes skeleton is currently posed for the timestep "frame"
        self.jpos = np.array(
            [
                skel_actor.joint_actors[ji].pos
                for ji in range(len(skel_actor.joint_actors))
            ]
        )
        self.viz_actor = SkeletonActor(
            self.pl,
            visualizer.smpl_joint_parents,
            joint_color=joint_color,
            bone_color=bone_color,
            joint_radius=joint_radius,
            bone_radius=bone_radius,
            name=f"body_constraint_{self.frame}",
        )
        self.viz_actor.update_joints(self.jpos)

        # save root pos/heading so can convert other joints to local frame later
        root_idx = SMPL_BONE_ORDER_NAMES.index("Pelvis")
        self.root_pos = self.jpos[root_idx]
        self.root_rot = compute_root_rot(self.jpos)  # heading vector
        non_root_inds = np.concatenate(
            [np.arange(root_idx), np.arange(root_idx + 1, self.jpos.shape[0])]
        )
        self.jpos = self.jpos[non_root_inds]

        if DEBUG_MODE:
            self.label = self.pl.add_point_labels(
                self.root_pos, [f"F{frame} full-body"]
            )

    def to_dict(self):
        return {
            "frame": self.frame,
            "root_pos": self.root_pos,
            "root_rot": self.root_rot,
            "joint_pos": self.jpos,
        }

    def remove(self):
        self.viz_actor.remove()
        if DEBUG_MODE:
            self.pl.remove_actor(self.label, render=False)


class TrajWaypoint:
    def __init__(
        self,
        pos,
        frame,
        visualizer,
        color,
        size,
        line_color,
        line_size,
        prev_waypoint=None,
        add_label=True,
        add_point_actor=True,
        add_disc_actor=False,
    ):
        self.pos = pos
        self.frame = frame
        self.visualizer = visualizer
        self.pl = visualizer.pl
        self.color = color
        self.size = size
        self.actor = self.disc_actor = self.mesh = self.disc_mesh = None
        if add_point_actor:
            self.mesh = pyvista.Sphere(
                radius=size, center=pos, theta_resolution=10, phi_resolution=10
            )
            actor = self.pl.add_mesh(
                self.mesh,
                color=color,
                ambient=0.3,
                diffuse=0.5,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
                render=False,
            )
            self.actor = ActorWrapper(
                actor, -1, "traj_waypoint", self.mesh, self, color
            )
        if add_disc_actor:
            disc_pos = copy.copy(pos)
            disc_pos[2] = 0.01  # float off ground slightly so visible
            self.disc_mesh = pyvista.Disc(
                center=disc_pos, inner=size + 0.1, outer=size + 0.2, c_res=30
            )
            disc_actor = self.pl.add_mesh(
                self.disc_mesh,
                color=color,
                ambient=0.3,
                diffuse=0.5,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
                render=False,
            )
            self.disc_actor = ActorWrapper(
                disc_actor, -1, "traj_waypoint", self.disc_mesh, self, color
            )
        self.line_mesh_actor = None
        if prev_waypoint is not None:
            self.line_mesh = pyvista.Cylinder(
                radius=line_size, center=(0, 0, 0), direction=(0, 0, 1), resolution=30
            )
            self.line_mesh_actor = self.pl.add_mesh(
                self.line_mesh,
                color=line_color,
                ambient=0.3,
                diffuse=0.5,
                specular=0.8,
                specular_power=5,
                smooth_shading=True,
                opacity=0.0,
                render=False,
            )
            self.update_line_transform(prev_waypoint.pos, pos)
        if add_label and DEBUG_MODE:
            self.label = self.pl.add_point_labels(
                pos, [f"F{self.frame} ({pos[0]:.2f}, {pos[1]:.2f})"]
            )
        else:
            self.label = None

    def update_line_transform(self, start_pos, end_pos):
        angle, axis, dist = get_line_angle_axis_dist(start_pos, end_pos)
        trans = vtkTransform()
        trans.Translate(*(start_pos + end_pos) * 0.5)
        trans.RotateWXYZ(np.rad2deg(angle), *axis)
        trans.Scale(1, 1, dist)
        self.line_mesh_actor.SetUserTransform(trans)
        self.line_mesh_actor.prop.SetOpacity(1.0)

    def remove(self):
        if self.actor is not None:
            self.actor.remove(self.pl)
        if self.disc_actor is not None:
            self.disc_actor.remove(self.pl)
        if self.line_mesh_actor is not None:
            self.pl.remove_actor(self.line_mesh_actor, render=False)
        if self.label is not None:
            self.pl.remove_actor(self.label, render=False)


class RootTrajConstraints:
    def __init__(
        self,
        visualizer,
        waypt_only=False,
        point_color="cyan",
        point_size=0.03,
        line_color="cyan",
        line_size=0.02,
        time_cmap="cool",
    ):
        self.visualizer = visualizer
        self.waypt_only = waypt_only
        self.pl = visualizer.pl
        self.point_color = point_color
        self.point_size = point_size
        self.line_color = line_color
        self.line_size = line_size
        self.way_points = []
        self.cmap = cm.get_cmap(time_cmap)

    def add_way_point(self, pos, frame, threshold=0.02, is_spline_waypoint=False):
        dist = (
            np.linalg.norm(pos - self.way_points[-1].pos)
            if len(self.way_points) > 0
            else 10.0
        )
        if dist < threshold:
            return
        prev_waypoint = self.way_points[-1] if len(self.way_points) > 0 else None
        cur_color = self.cmap(frame / (self.visualizer.orig_num_fr - 1))
        if is_spline_waypoint:
            self.way_points.append(
                TrajWaypoint(
                    pos,
                    frame,
                    self.visualizer,
                    cur_color,
                    self.point_size,
                    cur_color,
                    self.line_size,
                    prev_waypoint,
                    add_label=False,
                    add_point_actor=False,
                )
            )
        elif self.waypt_only:
            self.way_points.append(
                TrajWaypoint(
                    pos,
                    frame,
                    self.visualizer,
                    cur_color,
                    self.point_size,
                    cur_color,
                    self.line_size,
                    None,
                    add_disc_actor=True,
                )
            )
        else:
            self.way_points.append(
                TrajWaypoint(
                    pos,
                    frame,
                    self.visualizer,
                    cur_color,
                    self.point_size,
                    cur_color,
                    self.line_size,
                    prev_waypoint,
                )
            )

    def remove(self):
        for waypt in self.way_points:
            waypt.remove()


class ConstraintConfig:
    def __init__(self, init_val):
        self.val = init_val

    def update(self, new_val):
        self.val = new_val

    def get_val(self):
        return self.val


class ConstraintConfigHandler:
    def __init__(self, visualizer, config_settings):
        self.visualizer = visualizer
        self.config_settings = config_settings
        self.persistent_keys = ["foot_skate", "steps", "cfg"]
        self.configs = dict()
        self.model_configs = ["steps", "cfg"]
        self.num_guidance_configs = 0
        self.num_model_configs = 0
        self.bool_configs = []

    def init_configs(self):
        """add initial configs that are always there"""
        for k in self.persistent_keys:
            if k in self.config_settings:
                self.add_config(k)

    def contains(self, constraint_name):
        return constraint_name in self.configs

    def add_config(self, constraint_name):
        if self.contains(constraint_name):
            return

        self.configs[constraint_name] = dict()
        settings = self.config_settings[constraint_name]
        for hyperparam, info in settings.items():
            constr_cfg = ConstraintConfig(info["value"])
            self.configs[constraint_name][hyperparam] = constr_cfg
            numer_type = info.get("type", "float")

            num_configs = (
                self.num_model_configs
                if constraint_name in self.model_configs
                else self.num_guidance_configs
            )
            yloc = 0.87 - num_configs * 0.13
            xloc = (
                (0.73, 0.95) if constraint_name in self.model_configs else (0.03, 0.25)
            )

            if numer_type == "bool":
                # checkbox
                pos = [
                    self.visualizer.pl.window_size[0] * xloc[0],
                    self.visualizer.pl.window_size[1] * yloc,
                ]
                checkbox = self.visualizer.pl.add_checkbox_button_widget(
                    callback=constr_cfg.update,
                    value=info["value"],
                    position=pos,
                    border_size=5,
                    size=50,
                )
                pos[0] += 60.0
                label_str = info.get("name", f"{constraint_name}-\n{hyperparam}")
                check_label = self.visualizer.pl.add_text(
                    label_str, position=pos, font_size=18, color="black"
                )
                self.bool_configs.append((checkbox, check_label))
            else:
                # slider
                str_fmt = "%0.0f" if numer_type == "int" else None
                label_str = info.get("name", f"{constraint_name}-{hyperparam}")
                self.visualizer.pl.add_slider_widget(
                    callback=constr_cfg.update,
                    rng=info["range"],
                    value=info["value"],
                    title=label_str,
                    pointa=(xloc[0], yloc),
                    pointb=(xloc[1], yloc),
                    style="modern",
                    title_color="black",
                    fmt=str_fmt,
                )

            if constraint_name in self.model_configs:
                self.num_model_configs += 1
            else:
                self.num_guidance_configs += 1

    def get_configs(self):
        cur_configs = dict()
        for constraint, config in self.configs.items():
            cur_configs[constraint] = dict()
            for param, cfg_obj in config.items():
                cur_configs[constraint][param] = cfg_obj.get_val()
        return cur_configs

    def reset_configs(self):
        # update persistent settings with current values so can reload after clearing
        for save_k in self.persistent_keys:
            if save_k in self.configs:
                for hyperparam in self.configs[save_k]:
                    cur_fs = self.configs[save_k][hyperparam]
                    self.config_settings[save_k][hyperparam]["value"] = cur_fs.get_val()
        self.visualizer.pl.clear_slider_widgets()
        if self.visualizer.timeline:
            self.visualizer.pl.slider_widgets = [
                self.visualizer.timeline
            ]  # don't want to delete the timeline
        for checkbox, text in self.bool_configs:
            self.visualizer.pl.button_widgets.remove(checkbox)
            self.visualizer.pl.remove_actor(text, render=False)
        self.bool_configs.clear()
        self.configs = dict()
        self.num_guidance_configs = 0
        self.num_model_configs = 0
        self.init_configs()


class SMPLVisualizer(Visualizer3D):
    def __init__(
        self,
        smpl_model_dir=None,
        generator_func=None,
        constraint_config=None,
        device=torch.device("cpu"),
        show_smpl=SMPL,
        show_skeleton=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton

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
        self.smpl_joint_parents = smpl.parents.cpu().numpy()
        self.smpl_uv = np.load(
            os.path.join(smpl_model_dir, "cont_uv_map", "uv_table.npy")
        )
        para_file = os.path.join(
            smpl_model_dir, "cont_uv_map", "paras_h0256_w0256_SMPL.npz"
        )
        para = np.load(para_file)
        self.smpl_vt2v = torch.LongTensor(para["vt2v"])
        self.smpl_uv = para["texcoords"]
        faces = para["vt_faces"]  # update faces to work with textures
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])

        self.smpl_tex = []
        smpl_text_pth = os.path.join(smpl_model_dir, "textures")
        if os.path.exists(smpl_text_pth):
            self.smpl_tex = glob.glob(smpl_text_pth + "/*.png")
        self.generator_func = generator_func
        self.config_handler = ConstraintConfigHandler(self, constraint_config)
        self.smpl_seq = None
        self.device = device
        self.color_sequences = [
            ["Yellow", "Green", "cyan"],
            ["Yellow", "Red", "cyan"],
            ["Yellow", "Blue", "cyan"],
            ["Yellow", "Purple", "cyan"],
            ["Yellow", "Orange", "cyan"],
        ]
        self.timeline = None
        self.ui_hidden = False
        self.text_actors = []

    def update_smpl_seq(self, smpl_seq=None, mode="default"):
        if smpl_seq is None:
            smpl_seq = self.generator_func()
        self.smpl_seq = smpl_seq

        for i, (skel_name, pose_dict) in enumerate(smpl_seq.items()):
            colors = self.color_sequences[i % len(self.color_sequences)]
            normal_shape_len = {"pose": 2, "trans": 2, "shape": 2, "joints_pos": 3}
            for key in ["pose", "trans", "shape", "joints_pos"]:
                if (
                    key in pose_dict
                    and len(pose_dict[key].shape) > normal_shape_len[key]
                ):
                    pose_dict[key] = pose_dict[key][0]
            pose_dict["skeleton_jpos"] = SkeletonActor(
                self.pl,
                self.smpl_joint_parents,
                joint_color=colors[0],
                bone_color=colors[1],
                name=skel_name,
            )

            if self.show_smpl and "pose" in pose_dict and "trans" in pose_dict:
                smpl_verts = self.calc_smpl_vert_seq(pose_dict)
                pose_dict["vert_seq"] = smpl_verts
                pose_dict["smpl_actor"] = SMPLActor(
                    self.pl,
                    smpl_verts[0],
                    self.smpl_faces,
                    self.smpl_uv,
                    color="#e0c3fc",
                    visible=False,
                    tex_paths=self.smpl_tex,
                )

        self.fr = 0
        self.num_fr = smpl_seq["pred"]["joints_pos"].shape[0]
        self.orig_num_fr = smpl_seq["pred"]["orig_num_frames"]
        self.mode = mode
        self.set_camera()

    def calc_smpl_vert_seq(self, pose_dict):
        gender = pose_dict["gender"]
        pose = pose_dict["pose"]
        trans = pose_dict["trans"]
        shape = pose_dict.get("shape", torch.zeros((pose.size(0), 10)).to(self.device))
        smpl_motion = self.smpl_dict[gender](
            global_orient=pose[..., :3].view(-1, 3),
            body_pose=pose[..., 3:].view(-1, 69),
            betas=shape.view(-1, 10),
            root_trans=trans.view(-1, 3),
            orig_joints=True,
        )
        smpl_verts = (
            smpl_motion.vertices[:, self.smpl_vt2v].cpu().numpy()
        )  # .reshape(*orig_pose_shape[:-1], -1, 3)
        return smpl_verts

    def init_camera(self):
        super().init_camera()

    def set_camera(self):
        root_pos = self.smpl_seq["pred"]["joints_pos"][0, 0].cpu().numpy()
        view_vec = np.asarray(self.pl.camera.position) - np.asarray(
            self.pl.camera.focal_point
        )
        new_focal = np.array([root_pos[0], root_pos[1], 0.8])
        new_pos = new_focal + view_vec
        self.pl.camera.up = (0, 0, 1)
        self.pl.camera.focal_point = new_focal.tolist()
        self.pl.camera.position = new_pos.tolist()

    def update_camera(self, interactive):
        return

    def update_skel_vis(self, state, checkbox_name):
        self.smpl_seq[checkbox_name]["skeleton_jpos"].set_visibility(state)

    def update_smpl_vis(self, state, checkbox_name):
        if "smpl_actor" in self.smpl_seq[checkbox_name]:
            self.smpl_seq[checkbox_name]["smpl_actor"].set_visibility(state)
        else:
            print("SMPL not available to visualize!")

    def add_skel_vis_checkboxes(self, smpl_seq=None):
        if smpl_seq is None:
            return

        for i, skel_name in enumerate(smpl_seq.keys()):

            def create_box(box_name):
                colors = self.color_sequences[i % len(self.color_sequences)]
                pos = [
                    self.pl.window_size[0] * 0.4,
                    self.pl.window_size[1] * 0.85 - 0.05 * self.pl.window_size[1] * i,
                ]
                self.pl.add_checkbox_button_widget(
                    callback=lambda state: self.update_skel_vis(state, box_name),
                    value=True if box_name == "pred" else False,
                    position=pos,
                    color_on=colors[1],
                    background_color=colors[1],  # same as bone colors
                    border_size=5,
                    size=50,
                )
                pos[0] += 60.0
                if self.show_smpl:
                    self.pl.add_checkbox_button_widget(
                        callback=lambda state: self.update_smpl_vis(state, box_name),
                        value=False,
                        position=pos,
                        # color_on=colors[1],
                        # background_color=colors[1], # same as bone colors
                        border_size=5,
                        size=50,
                    )
                    pos[0] += 60.0
                self.text_actors.append(
                    self.pl.add_text(
                        NAME_MAP[box_name], position=pos, font_size=18, color=colors[1]
                    )
                )

            create_box(skel_name)
            self.update_skel_vis(True if skel_name == "pred" else False, skel_name)

    def init_scene(self, init_args):
        global pickable_actors, Mesh2Actor, ActorMapping
        pickable_actors = set()
        Mesh2Actor = {}
        ActorMapping = {}

        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        init_smpl_seq = init_args.get("smpl_seq", None)
        self.update_smpl_seq(init_smpl_seq, init_args.get("mode", "gt"))
        self.add_skel_vis_checkboxes(init_smpl_seq)

        self.joint_constraints = []
        self.body_constraints = []
        self.root_traj_constraint = None
        self.picked_actor = None
        self.picking_mode = "actor_joint"
        self.enable_actor_picking()
        # self.add_test_constraints()
        self.text_constraint = None
        self.text_constr_actor = self.pl.add_text(
            "", position="lower_edge", font="courier", font_size=20, color="black"
        )

        # text to elicit user input prompt
        self.text_in_actor = self.pl.add_text(
            "",
            position=(0.15, 0.5),
            viewport=True,
            font="arial",
            font_size=28,
            color="black",
            shadow=True,
            render=False,
        )

        self.config_handler.init_configs()

        self.fps_text_actor = self.pl.add_text(
            FPS_TEXT % (self.fps), position="upper_right"
        )
        self.select_mode_text_actor = self.pl.add_text(
            SELECTION_TEXT % (PICKING_NAME_MAP[self.picking_mode]),
            position="upper_left",
        )
        self.text_actors += [
            self.text_constr_actor,
            self.fps_text_actor,
            self.select_mode_text_actor,
        ]

        self.timeline = self.init_timeline()

    def init_timeline(self):
        def timeline_callback(time):
            self.fr = int(round(time))
            self.update_scene()

        cap_color = Color("black")
        tube_color = Color("#b2b3b5ff")
        slider_color = Color("#6e7175ff")
        cap_width = 0.05
        cap_length = 0.015
        tube_width = 0.05
        slider_width = 0.05
        slider_length = 0.01

        slider = self.pl.add_slider_widget(
            callback=timeline_callback,
            rng=[0, self.orig_num_fr - 1],
            value=0,
            pointa=(0.05, 0.15),
            pointb=(0.95, 0.15),
            fmt="%0.0f",
            slider_width=slider_width,
            tube_width=tube_width,
            interaction_event="always",
            title="Timeline",
        )
        slider.GetRepresentation().GetCapProperty().SetColor(cap_color.float_rgb)
        slider.GetRepresentation().GetTubeProperty().SetColor(tube_color.float_rgb)
        slider.GetRepresentation().GetSliderProperty().SetColor(slider_color.float_rgb)
        slider.GetRepresentation().SetEndCapWidth(cap_width)
        slider.GetRepresentation().SetEndCapLength(cap_length)
        slider.GetRepresentation().SetSliderLength(slider_length)
        slider.GetRepresentation().GetLabelProperty().SetColor(cap_color.float_rgb)

        return slider

    def set_all_unpickable(self):
        for actor in self.pl.renderers[0].actors.values():
            actor.SetPickable(False)

    def enable_actor_picking(self):
        def callback_actor_joint(vtk_actor):
            """single joint constraint"""
            self.reset_picked_actors()
            if vtk_actor in ActorMapping:
                actor = ActorMapping[vtk_actor]
                actor.prop.color = "orange"
                self.picked_actor = actor

        def callback_actor_body(vtk_actor):
            """full body constraint"""
            self.reset_picked_actors()
            if vtk_actor in ActorMapping:
                actor = ActorMapping[vtk_actor]
                if actor.type in {"joint", "bone"}:
                    # get the parent skeleton and set this as picked
                    parent_skel = actor.parent
                    self.picked_actor = parent_skel
                    for joint in parent_skel.joint_actors:
                        joint.prop.color = "orange"

        callbacks = {
            "actor_joint": callback_actor_joint,
            "actor_body": callback_actor_body,
        }
        self.pl.enable_mesh_picking(
            callback=callbacks[self.picking_mode],
            use_actor=True,
            left_clicking=True,
            show=False,
            show_message=False,
        )
        self.set_all_unpickable()
        self.pl.pickable_actors = [actor.actor for actor in pickable_actors]
        # print('num_pickable: ', sum([x.GetPickable() for x in self.pl.renderers[0].actors.values()]))

    def enable_surface_picking(self):
        def callback_surface(point):
            """Create a cube and a label at the click point."""
            mesh = self.pl._picked_mesh
            # print(point)
            # print(mesh)
            # if mesh == self.floor_mesh:
            # cube = pyvista.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
            # self.pl.add_mesh(cube, style='wireframe', color='r')
            # self.pl.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
            if self.picking_mode == "surface_root_traj":
                self.add_path_constraints(point)
            elif self.picking_mode == "surface_root_key":
                self.add_path_constraints(point, waypt_only=True)

        self.pl.enable_surface_picking(
            callback=callback_surface,
            left_clicking=True,
            show_point=False,
            show_message=False,
        )
        self.set_all_unpickable()
        self.pl.pickable_actors = [self.floor_actor]

    def add_path_constraints(
        self, point, is_spline=False, waypt_only=False, frame=None
    ):
        if self.root_traj_constraint is None:
            self.root_traj_constraint = RootTrajConstraints(
                self,
                waypt_only=waypt_only,
                point_color="cyan",
                point_size=0.03,
                line_color="purple",
                line_size=0.02,
            )
            if not is_spline and not waypt_only:
                self.root_traj_constraint.add_way_point(
                    np.array([0, 0, 0]), 0
                )  # at frame 0

        if frame is None:
            frame = self.fr
        if is_spline:
            self.root_traj_constraint.add_way_point(
                point, frame, threshold=0, is_spline_waypoint=True
            )
        else:
            self.root_traj_constraint.add_way_point(point, frame)

        self.set_all_unpickable()
        self.pl.pickable_actors = [self.floor_actor]
        # print('num_pickable: ', sum([x.GetPickable() for x in self.pl.renderers[0].actors.values()]))
        # print('num_waypoints: ', len(self.root_traj_constraint.way_points))

        if self.root_traj_constraint.waypt_only:
            if not self.config_handler.contains("root2d_keyframes"):
                self.config_handler.add_config("root2d_keyframes")
        else:
            if not self.config_handler.contains("root_traj_xy"):
                self.config_handler.add_config("root_traj_xy")

    def create_spline_path(self):
        if self.root_traj_constraint is not None:
            wp = np.stack([x.pos for x in self.root_traj_constraint.way_points])
            # interpolate based on specified frame indices
            frame_ind = np.array(
                [x.frame for x in self.root_traj_constraint.way_points]
            )
            # frame_ind = np.linspace(0, self.orig_num_fr - 1, len(wp))
            # Define the new frame indices you want to interpolate at (e.g., dense frame indices)
            dense_frame_indices = np.arange(
                self.orig_num_fr
            )  # This creates an array from the first to the last frame index
            # Create a cubic spline interpolation for all dimensions (x, y, z) simultaneously
            spline = CubicSpline(frame_ind, wp, axis=0)
            # Interpolate the waypoints for the dense frame indices using the cubic spline
            dense_waypoints = spline(dense_frame_indices)
            self.reset_path_constraints()
            for i, wp in enumerate(dense_waypoints):
                self.add_path_constraints(
                    wp, frame=dense_frame_indices[i], is_spline=True
                )
        return

    def add_joint_constraints(self):
        if self.picked_actor is not None:
            if isinstance(self.picked_actor, SkeletonActor):
                new_body_constr = BodyConstraints(self.fr, self.picked_actor, self)
                self.body_constraints.append(new_body_constr)
                if not self.config_handler.contains("body_keyframes"):
                    self.config_handler.add_config("body_keyframes")
            elif self.picked_actor.type == "joint":
                new_joint_constr = JointConstraints(
                    self.picked_actor.pos, self.fr, self.picked_actor, self
                )
                self.joint_constraints.append(new_joint_constr)
                if not self.config_handler.contains("joint_keyframes"):
                    self.config_handler.add_config("joint_keyframes")
        self.reset_picked_actors()

    def add_text_constraint(self, text_input):
        self.text_constraint = text_input.strip()
        self.text_constr_actor.SetText(4, self.text_constraint)

    def reset_picked_actors(self):
        self.picked_actor = None
        for actor in pickable_actors:
            actor.reset_picked()

    def update_scene(self):
        super().update_scene()
        if self.timeline:
            self.timeline.GetRepresentation().SetValue(self.fr)

        # hack to update text: 3 and 2 corresponds to upper_right/upper_left
        #       (https://vtk.org/doc/nightly/html/classvtkCornerAnnotation.html#a2bc8727c59f24241121a92a72dc2fe86)
        self.fps_text_actor.SetText(3, FPS_TEXT % (self.fps))

        if self.show_skeleton:
            for skel_name, pose_dict in self.smpl_seq.items():
                pose_dict["skeleton_jpos"].update_joints(
                    pose_dict["joints_pos"][
                        min(self.fr, pose_dict["joints_pos"].shape[0] - 1)
                    ]
                    .cpu()
                    .numpy()
                )
        if self.show_smpl:
            for skel_name, pose_dict in self.smpl_seq.items():
                pose_dict["smpl_actor"].update_verts(
                    pose_dict["vert_seq"][
                        min(self.fr, pose_dict["vert_seq"].shape[0] - 1)
                    ]
                )

    def prepare_constraints(self):
        constraints = dict()
        # text
        if self.text_constraint:
            constraints["text"] = self.text_constraint
        # 2d path
        if self.root_traj_constraint:
            wp = np.stack([x.pos for x in self.root_traj_constraint.way_points])
            frame_ind = np.array(
                [x.frame for x in self.root_traj_constraint.way_points]
            )
            if self.root_traj_constraint.waypt_only:
                root_traj = wp
            else:
                # interpolate based on given frames idx for each waypoint
                interpf = interp1d(
                    frame_ind, wp, axis=0, assume_sorted=True, fill_value="extrapolate"
                )
                root_traj = interpf(np.arange(self.orig_num_fr))
                # root_traj = interp_scipy_ndarray(wp, new_len=self.orig_num_fr, dim=0)
            root_traj_xy = np.stack(
                [root_traj[:, 1], root_traj[:, 0]], axis=-1
            )  # switch x, y (go from UI frame to model frame) -- TODO should move this to the test_gui
            vec = root_traj[1:, :2] - root_traj[:-1, :2]
            vec = np.stack([vec[:, 0], -vec[:, 1]], axis=-1)
            vec = np.concatenate([vec, vec[-1:]], axis=0)
            root_rot_cos_sin = vec / (np.linalg.norm(vec, axis=-1)[:, None] + 1e-8)
            if self.root_traj_constraint.waypt_only:
                constraints["root2d_keyframes"] = {
                    "pos": root_traj_xy,
                    "rot": root_rot_cos_sin,
                    "frames": list(frame_ind),
                }
            else:
                constraints["root_traj_xy"] = {
                    "pos": root_traj_xy,
                    "rot": root_rot_cos_sin,
                }
        # sparse 3d joints
        if len(self.joint_constraints) > 0:
            constr_list = []
            for jconstr in self.joint_constraints:
                constr_list.append(jconstr.to_dict())
            constraints["joint_keyframes"] = constr_list
        # sparse full-body joints
        if len(self.body_constraints) > 0:
            constr_list = []
            for bconstr in self.body_constraints:
                constr_list.append(bconstr.to_dict())
            constraints["body_keyframes"] = constr_list

        configs = self.config_handler.get_configs()
        return constraints, configs

    def update_motion_from_constraints(self):
        constraints, configs = self.prepare_constraints()
        new_smpl_seq = self.generator_func(constraints, configs)
        for skel_name, pose_dict in self.smpl_seq.items():
            new_pose_dict = new_smpl_seq[skel_name]
            pose_dict["joints_pos"] = new_pose_dict["joints_pos"]
            if "pose" in new_pose_dict and "trans" in new_pose_dict:
                smpl_verts = self.calc_smpl_vert_seq(new_pose_dict)
                pose_dict["vert_seq"] = smpl_verts
        self.fr = 0
        self.update_scene()

    def reset_path_constraints(self):
        if self.root_traj_constraint is not None:
            self.root_traj_constraint.remove()
            self.root_traj_constraint = None

    def reset_constraints(self):
        # joint constraints
        for constr in self.joint_constraints:
            constr.remove()
        self.joint_constraints = []
        # body constraints
        for constr in self.body_constraints:
            constr.remove()
        self.body_constraints = []
        # path constraints
        self.reset_path_constraints()
        # update picking to account for missing actors
        if self.picking_mode in {"actor_joint", "actor_body"}:
            self.enable_actor_picking()
        elif self.picking_mode in {"surface_root_traj", "surface_root_key"}:
            self.enable_surface_picking()
        # reset sliders
        self.config_handler.reset_configs()
        # remove text input
        self.text_constraint = None
        self.text_constr_actor.SetText(4, "")

    def hide_ui(self):
        all_text = [
            text for text in self.text_actors if text != self.text_constr_actor
        ]  # want to keep this on for visualization
        all_text += [text for _, text in self.config_handler.bool_configs]

        if not self.ui_hidden:
            # make everything invisible
            for widget in self.pl.slider_widgets:
                widget.Off()
            for widget in self.pl.button_widgets:
                widget.Off()
                widget.GetRepresentation().VisibilityOff()
            for text in all_text:
                text.VisibilityOff()
        else:
            # make everything visible
            for widget in self.pl.slider_widgets:
                widget.On()
            for widget in self.pl.button_widgets:
                widget.On()
                widget.GetRepresentation().VisibilityOn()
            for text in all_text:
                text.VisibilityOn()

        self.ui_hidden = not self.ui_hidden

    def next_smpl_tex(self):
        if self.show_smpl:
            for skel_name, pose_dict in self.smpl_seq.items():
                pose_dict["smpl_actor"].next_tex()

    def get_user_prompt(self):
        # get continual user input
        input_buffer = ""
        print("Enter motion text prompt: ")

        prompt_str = "Type Input Text: %s"
        self.text_in_actor.SetInput(prompt_str % ("") + "...")

        # Save current terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())

            while True:
                self.render(interactive=False)  # need to keep refreshing to update text
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char == "\r":  # Enter key
                        break
                    elif char == "\x7f":  # Backspace
                        input_buffer = input_buffer[:-1]  # Remove the last character
                    else:
                        input_buffer += char

                    self.text_in_actor.SetInput(
                        "\n".join(textwrap.wrap(prompt_str % (input_buffer), 40))
                    )
                    sys.stdout.write("\x1b[1K\r" + f"{input_buffer}")
                    sys.stdout.flush()
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print()  # Move to the next line after exiting
            self.text_in_actor.SetInput("")

        prompt_in = input_buffer.strip()

        return prompt_in

    def setup_key_callback(self):
        super().setup_key_callback()

        def reset_camera():
            self.init_camera()

        def print_camera():
            print("focal", self.pl.camera.focal_point)
            print("position", self.pl.camera.position)
            print("elevation", self.pl.camera.elevation)
            print("azimuth", self.pl.camera.azimuth)

        def add_joint_constraint():
            self.add_joint_constraints()

        def add_test_constraints():
            self.add_test_constraints()
            self.create_spline_path()

        def reset_constraints():
            print("Removing all constraints...")
            self.reset_constraints()

        def hide_ui():
            self.hide_ui()

        def cycle_character():
            self.next_smpl_tex()

        def update_motion_constr():
            gen_txt = self.pl.add_text(
                "Generating...",
                position=(0.38, 0.5),
                viewport=True,
                font_size=32,
                color="red",
                shadow=True,
            )
            self.update_motion_from_constraints()
            self.pl.remove_actor(gen_txt, render=False)

        def toggle_picking_mode():
            picking_modes = [
                "actor_joint",
                "actor_body",
                "surface_root_traj",
                "surface_root_key",
            ]
            cur_picking_mode = self.picking_mode
            self.picking_mode = picking_modes[
                (picking_modes.index(self.picking_mode) + 1) % len(picking_modes)
            ]
            self.select_mode_text_actor.SetText(
                2, SELECTION_TEXT % (PICKING_NAME_MAP[self.picking_mode])
            )
            self.reset_picked_actors()
            print(f"picking mode: {self.picking_mode}")
            if self.picking_mode in {"actor_joint", "actor_body"}:
                self.enable_actor_picking()
            elif self.picking_mode in {"surface_root_traj", "surface_root_key"}:
                self.enable_surface_picking()

        def add_text_constr():
            # prompt_in = input("Enter motion text prompt: ").strip()
            prompt_in = self.get_user_prompt()

            if prompt_in == "":
                self.text_constraint = None
                self.text_constr_actor.SetText(4, "")
                return
            prompt_in = textwrap.wrap(prompt_in, 40)
            prompt_in = "\n".join(prompt_in)
            self.add_text_constraint(prompt_in)

        def new_gt_motion():
            gen_txt = self.pl.add_text(
                "Loading next\nmotion...",
                position=(0.38, 0.5),
                viewport=True,
                font_size=32,
                color="red",
                shadow=True,
            )
            self.reset_constraints()
            new_smpl_seq = self.generator_func(new_gt_motion=True)
            for skel_name, pose_dict in self.smpl_seq.items():
                new_pose_dict = new_smpl_seq[skel_name]
                pose_dict["joints_pos"] = new_pose_dict["joints_pos"]
                if self.show_smpl:
                    if "pose" in new_pose_dict and "trans" in new_pose_dict:
                        smpl_verts = self.calc_smpl_vert_seq(new_pose_dict)
                        pose_dict["vert_seq"] = smpl_verts

            self.fr = 0
            self.update_scene()
            self.pl.remove_actor(gen_txt, render=False)

        def create_spline_path():
            self.create_spline_path()

        self.pl.add_key_event("y", print_camera)
        self.pl.add_key_event("c", add_joint_constraint)
        self.pl.add_key_event("x", toggle_picking_mode)
        self.pl.add_key_event("g", update_motion_constr)
        self.pl.add_key_event("t", add_text_constr)
        self.pl.add_key_event("z", reset_constraints)
        self.pl.add_key_event("n", new_gt_motion)
        self.pl.add_key_event("k", create_spline_path)
        self.pl.add_key_event("h", hide_ui)
        self.pl.add_key_event("b", cycle_character)
