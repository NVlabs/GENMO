import os, sys
sys.path.append(os.path.join(os.getcwd()))
import pyvista
import time
import torch
import numpy as np
from motiondiff.utils.visualizer3d import Visualizer3D
from motiondiff.models.common.smpl import SMPL
from torch.utils.data import DataLoader
from vtk import vtkTransform
from motiondiff.utils.torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion


ActorMapping = {}
Mesh2Actor = {}


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

class ActorWrapper:

    def __init__(self, actor, mesh, skeleton, color):
        self.actor = actor
        self.mesh = mesh
        self.skeleton = skeleton
        self.color = color
        ActorMapping[actor] = self
        Mesh2Actor[id(mesh)] = self

    def GetProperty(self):
        return self.actor.GetProperty()
    
    def SetVisibility(self, flag):
        self.actor.SetVisibility(flag)

    def SetUserTransform(self, trans):
        self.actor.SetUserTransform(trans)

    @property
    def prop(self):
        return self.actor.prop

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
            joint_actor = ActorWrapper(joint_actor, joint_mesh, self, joint_color)
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(radius=bone_radius, center=(0, 0, 0), direction=(0, 0, 1), resolution=30)
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
                bone_actor = ActorWrapper(bone_actor, bone_mesh, self, bone_color)
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


class SMPLVisualizer(Visualizer3D):

    def __init__(self, joint_parents=None, generator_func=None, device=torch.device('cpu'), show_skeleton=True, **kwargs):
        if 'smpl' in kwargs:
            faces = kwargs['smpl'].faces.copy()
            self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
            del kwargs['smpl']
        super().__init__(**kwargs)
        self.joint_parents = joint_parents
        self.show_skeleton = show_skeleton
        self.generator_func = generator_func
        self.device = device
        self.color_sequences = {
            'gt': ['Yellow', 'Green', 'Teal'],
            'smpl': ['Yellow', 'Red', 'Teal'],
            'retarget': ['Yellow', 'Blue', 'Teal'],
            'retarget_ik': ['Yellow', 'Cyan', 'Teal'],
        }
        
    def update_smpl_seq(self, smpl_seq=None, mode='gt'):
        self.smpl_seq = smpl_seq
        if smpl_seq is None:
            try:
                smpl_seq = next(self.smpl_motion_generator)
            except:
                self.smpl_motion_generator = self.generator_func()
                smpl_seq = next(self.smpl_motion_generator)
        
        for i, (skel_name, pose_dict) in enumerate(smpl_seq.items()):
            if 'gt' in skel_name:
                colors = self.color_sequences['gt']
            elif 'smpl' in skel_name:
                colors = self.color_sequences['smpl']
            elif 'retarget_ik' in skel_name:
                colors = self.color_sequences['retarget_ik']
            elif 'retarget' in skel_name:
                colors = self.color_sequences['retarget']
            else:
                colors = self.color_sequences[i % len(self.color_sequences)]
            normal_shape_len = {'joints_pos': 3}
            for key in ['joints_pos']:
                if key in pose_dict and len(pose_dict[key].shape) > normal_shape_len[key]:
                    pose_dict[key] = pose_dict[key][0]
            joint_parents = pose_dict.get('parents') if pose_dict.get('parents') is not None else self.joint_parents
            if pose_dict.get('show_smpl_mesh'):
                pose_dict['smpl_actor'] = SMPLActor(self.pl, pose_dict['smpl_verts'][0].cpu().numpy(), self.smpl_faces, color="#e0c3fc")
                pose_dict['smpl_actor'].set_opacity(pose_dict['opacity'])
            else:
                pose_dict['skeleton_jpos'] = SkeletonActor(self.pl, joint_parents, joint_color=colors[0], bone_color=colors[1])

        self.fr = 0
        if 'gt' in smpl_seq:
            self.num_fr = smpl_seq['gt']['joints_pos'].shape[0] - 1
        else:
            self.num_fr = smpl_seq['smpl_joints']['joints_pos'].shape[0] - 1
        self.mode = mode

    def init_camera(self):
        super().init_camera()

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        # self.floor_mesh.points[:, 2] -= 0.08
        self.update_smpl_seq(init_args.get('smpl_seq', None), init_args.get('mode', 'gt'))
        self.view_dir = init_args.get('view_dir', 'XY+')
        
    def update_camera(self, interactive):
        if 'gt' in self.smpl_seq:
            root_pos = self.smpl_seq['gt']['joints_pos'][self.fr, 0].cpu().numpy()
        else:
            root_pos = self.smpl_seq['smpl_joints']['joints_pos'][self.fr, 0].cpu().numpy()
        roll = self.pl.camera.roll
        if self.view_dir == 'XY+':
            view_vec = np.asarray((self.distance/2, self.distance, self.elevation))
        elif self.view_dir == 'Y-':
            view_vec = np.asarray((0, -self.distance, self.elevation))
        else:
            raise ValueError(f"Not implemented view_dir=[{self.view_dir}]")
        new_focal = np.array([root_pos[0], root_pos[1], 0.8])
        new_pos = new_focal + view_vec

        ''' orthogonal view '''
        # new_focal = np.array([root_pos[0], root_pos[1], root_pos[2]])
        # new_pos = np.asarray((root_pos[0], self.distance, root_pos[2]))

        self.pl.camera.up = (0, 0, 1)
        self.pl.camera.focal_point = new_focal.tolist()
        self.pl.camera.position = new_pos.tolist()
        # self.pl.camera.roll = roll   # don't set roll

    def update_scene(self):
        super().update_scene()

        if self.show_skeleton:
            for skel_name, pose_dict in self.smpl_seq.items():
                if 'skeleton_jpos' in pose_dict:
                    pose_dict['skeleton_jpos'].update_joints(pose_dict['joints_pos'][self.fr].cpu().numpy())
                if 'smpl_actor' in pose_dict:
                    pose_dict['smpl_actor'].update_verts(pose_dict['smpl_verts'][self.fr].cpu().numpy())

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


if __name__ == '__main__':
    
    from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
    from motiondiff.utils.hybrik import batch_rigid_transform

    bvh_file = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw/P8/BVH/big_light_one_hand_behind_high_to_right_side_high_R_001__A527_M.bvh'
    file_names = [
        'P8/BVH/big_light_one_hand_behind_high_to_right_side_high_R_001__A527_M.bvh',
    ]

    # skeleton = Skeleton()
    # skeleton.load_from_bvh(bvh_file)
    # root_trans, joint_rot_mats = load_bvh_animation(bvh_file, skeleton)
    # torch.save((root_trans, joint_rot_mats), 'out/joint_rot.p')
    root_trans, joint_rot_mats = torch.load('out/joint_rot.p')
    # parent_indices = skeleton.get_parent_indices()
    # joints = skeleton.get_neutral_joints()

    # rot_mats = torch.tensor(joint_rot_mats)
    # joints = torch.tensor(joints).unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)
    # parents = torch.LongTensor(parent_indices)
    # torch.save((rot_mats, joints, parents), 'out/joint_info.p')
    rot_mats, joints, parents = torch.load('out/joint_info.p')
    # rot_mats[:] = torch.eye(3)
    joints -= joints[:, [0]]
    
    posed_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, parents)
    posed_joints += torch.tensor(root_trans).unsqueeze(1)
    posed_joints = torch.stack([posed_joints[:, :, 0], posed_joints[:, :, 2], posed_joints[:, :, 1]], dim=-1)

    vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10)
    smpl_seq = {
        'gt':{
            'joints_pos': posed_joints.float() * 0.01
        }
    }
    video_path = f'out/bones/test.mp4'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # vis.save_animation_as_video(video_path, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), frame_dir='out/bones/frames')
    vis.show_animation(init_args={'smpl_seq': smpl_seq, 'mode': 'gt'},
                                   window_size=(1500, 1500),
                                   fps=30,
                                   show_axes=False,
                                   enable_shadow=False)