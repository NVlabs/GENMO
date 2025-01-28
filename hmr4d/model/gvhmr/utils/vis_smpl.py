import os, sys
sys.path.append(os.path.join(os.getcwd()))
import pyvista
import time
import torch
import numpy as np
from motiondiff.utils.visualizer3d import Visualizer3D
from motiondiff.models.common.smpl import SMPL
from torch.utils.data import DataLoader
from pyvista.plotting import Color
from vtk import vtkTransform
from motiondiff.utils.torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion
from hmr4d.utils.smplx_utils import make_smplx


class SMPLActor():

    def __init__(self, pl, verts, faces, color='#FF8A82', visible=True):
        self.pl = pl
        self.verts = verts
        self.face = faces
        self.mesh = pyvista.PolyData(verts, faces)
        self.actor = self.pl.add_mesh(self.mesh, color=color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
        # self.actor = self.pl.add_mesh(self.mesh, color=color, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        self.set_visibility(visible)

    def update_verts(self, new_verts):
        self.mesh.points[...] = new_verts
        self.mesh.compute_normals(inplace=True)

    def set_opacity(self, opacity):
        self.actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)

    def set_color(self, color):
        rgb_color = Color(color)
        self.actor.GetProperty().SetColor(rgb_color)


class SkeletonActor():

    def __init__(self, pl, joint_parents, joint_color='green', bone_color='yellow', joint_radius=0.03, bone_radius=0.02, visible=True):
        self.pl = pl
        self.joint_parents = joint_parents
        self.joint_meshes = []
        self.joint_actors = []
        self.bone_meshes = []
        self.bone_actors = []
        self.bone_pairs = []
        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = pyvista.Sphere(radius=joint_radius, center=(0, 0, 0), theta_resolution=10, phi_resolution=10)
            # joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
            joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(radius=bone_radius, center=(0, 0, 0), direction=(0, 0, 1), resolution=30)
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
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
        # bone
        vec = []
        for actor, (j, pa) in zip(self.bone_actors, self.bone_pairs):
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor([0., 0., 1.]).expand_as(vec), vec)).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)
        
        for actor, (j, pa), angle_i, axis_i, dist_i in zip(self.bone_actors, self.bone_pairs, angle, axis, dist):
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



class SMPLVisualizer(Visualizer3D):

    def __init__(self, generator_func=None, device=torch.device('cuda'), show_smpl=True, show_skeleton=False, **kwargs):
        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        smpl_model_dir = 'inputs/smpl_data'
        
        self.smpl_dict = {
            'neutral': SMPL(smpl_model_dir, create_transl=False, gender='neutral').to(device),
            'male': SMPL(smpl_model_dir, create_transl=False, gender='male').to(device),
            'female': SMPL(smpl_model_dir, create_transl=False, gender='female').to(device)
        }
        smpl = self.smpl_dict['male']
        faces = smpl.faces.copy()
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = smpl.parents.cpu().numpy()
        
        self.generator_func = generator_func
        self.smpl_motion_generator = None
        self.device = device
        
    def update_smpl_seq(self, smpl_seq=None, mode='gt'):
        if smpl_seq is None:
            try:
                smpl_seq = next(self.smpl_motion_generator)
            except:
                self.smpl_motion_generator = self.generator_func()
                smpl_seq = next(self.smpl_motion_generator)
        self.smpl_seq = smpl_seq
        smpl_seq['body_pose'] = torch.cat([smpl_seq['body_pose'], torch.zeros_like(smpl_seq['body_pose'][..., :6])], dim=1)
        outs = self.smpl_dict['neutral'](**smpl_seq)
        self.smpl_verts = outs.vertices
        self.smpl_joints = outs.joints

        self.fr = 0
        self.num_fr = self.smpl_verts.shape[1]
        self.mode = mode

    def init_camera(self):
        super().init_camera()

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        # self.floor_mesh.points[:, 2] -= 0.08
        self.update_smpl_seq(init_args.get('smpl_seq', None), init_args.get('mode', 'gt'))
        self.num_actors = init_args.get('num_actors', 1)
        if self.show_smpl and self.smpl_verts is not None:
            vertices = self.smpl_verts[0].cpu().numpy()
            self.smpl_actors = [SMPLActor(self.pl, vertices, self.smpl_faces, color="#e0c3fc") for _ in range(self.num_actors)]
        
    def update_camera(self, interactive):
        root_pos = self.smpl_joints[self.fr, 0].cpu().numpy()
        roll = self.pl.camera.roll
        view_vec = np.asarray(self.pl.camera.position) - np.asarray(self.pl.camera.focal_point)
        new_focal = np.array([root_pos[0], root_pos[1], 0.8])
        new_pos = new_focal + view_vec
        self.pl.camera.up = (0, 0, 1)
        self.pl.camera.focal_point = new_focal.tolist()
        self.pl.camera.position = new_pos.tolist()
        # self.pl.camera.roll = roll   # don't set roll

    def update_scene(self):
        super().update_scene()

        if self.show_smpl and self.smpl_verts is not None:
            opacity = 1.0
            
            for i, actor in enumerate(self.smpl_actors):
                actor.set_visibility(True)
                actor.update_verts(self.smpl_verts[self.fr].cpu().numpy())
                actor.set_opacity(opacity)

    def setup_key_callback(self):
        super().setup_key_callback()

        def next_data():
            self.update_smpl_seq()

        def reset_camera():
            self.init_camera()

        def print_camera():
            print('focal', self.pl.camera.focal_point)
            print('position', self.pl.camera.position)
            print('elevation', self.pl.camera.elevation)
            print('azimuth', self.pl.camera.azimuth)

        self.pl.add_key_event('z', next_data)
        self.pl.add_key_event('t', print_camera)
