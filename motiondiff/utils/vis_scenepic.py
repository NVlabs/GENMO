import os, sys
sys.path.append(os.path.join(os.getcwd()))
import pyvista
import time
import torch
import numpy as np
from motiondiff.models.common.smpl import SMPL
from motiondiff.utils.torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion
from motiondiff.utils.vis import make_checker_board_texture
import scenepic as sp
import wandb


class SkeletonActor():

    def __init__(self, scene, name, joint_parents, joint_color='Yellow', bone_color='Green', joint_constr_color='Cyan', joint_radius=0.06, bone_radius=0.04, joint_constr_radius=0.1):
        self.scene = scene
        self.name = name
        self.joint_parents = joint_parents
        self.joint_radius = joint_radius
        self.joint_color = getattr(sp.Colors, joint_color, sp.Colors.Green)
        self.bone_color = getattr(sp.Colors, bone_color, sp.Colors.Green)
        self.joint_constr_color = getattr(sp.Colors, joint_constr_color, sp.Colors.Green)
        self.bone_radius = bone_radius
        self.joint_meshes = []
        self.joint_constr_meshes = []
        self.bone_meshes = []
        self.bone_pairs = []

        self.floor_img = scene.create_image(image_id="floor")
        self.floor_img.from_numpy(make_checker_board_texture('#81C6EB', '#D4F1F7'))
        self.floor_mesh = scene.create_mesh(texture_id="floor", layer_id="floor")
        self.floor_mesh.add_image(transform=sp.Transforms.Scale(20))

        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = scene.create_mesh(f"{name}_joint{j}", layer_id=f"{self.name}")
            joint_mesh.add_sphere(color=self.joint_color, transform=sp.Transforms.scale(joint_radius))
            self.joint_meshes.append(joint_mesh)
            # joint constraints
            joint_constr_mesh = scene.create_mesh(f"{name}_joint_constr{j}", layer_id=f"{self.name}")
            joint_constr_mesh.add_sphere(color=self.joint_constr_color, transform=sp.Transforms.scale(joint_constr_radius))
            self.joint_constr_meshes.append(joint_constr_mesh)
            # bone
            if pa >= 0:
                bone_mesh = scene.create_mesh(f"{name}_bone{j}", layer_id=f"{self.name}")
                bone_mesh.add_cone(color=self.bone_color, transform=sp.Transforms.scale(np.array([1, joint_radius, joint_radius])))
                self.bone_meshes.append(bone_mesh)
                self.bone_pairs.append((j, pa, bone_mesh))

    def add_mesh_to_frames(self, sp_frame, jpos):

        sp_frame.add_mesh(self.floor_mesh)
        # joint
        for j, pos in enumerate(jpos):
            sp_frame.add_mesh(self.joint_meshes[j], transform=sp.Transforms.translate(pos))
        
        # bone
        vec = []
        for j, pa, _ in self.bone_pairs:
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor([-1., 0., 0.]).expand_as(vec), vec)).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)
        
        for (j, pa, bone_mesh), angle_i, axis_i, dist_i in zip(self.bone_pairs, angle, axis, dist):
            transform = sp.Transforms.translate((jpos[pa] + jpos[j]) * 0.5)
            transform = transform @ sp.Transforms.RotationMatrixFromAxisAngle(axis_i, angle_i)
            transform = transform @ sp.Transforms.Scale(np.array([dist_i, 1, 1]))
            sp_frame.add_mesh(bone_mesh, transform=transform)
    
    def add_joint_constr_meshes_to_frames(self, sp_frame, jpos, joint_mask):
        # joint constraints
        for j, pos in enumerate(jpos):
            if joint_mask[j].any():
                sp_frame.add_mesh(self.joint_constr_meshes[j], transform=sp.Transforms.translate(pos))
        


class ScenepicVisualizer:

    def __init__(self, smpl_model_dir=None, device=torch.device('cpu'), show_skeleton_jpos=True, show_ik_smpl_pose=False, **kwargs):
        super().__init__(**kwargs)
        self.smpl_dict = {
            'neutral': SMPL(smpl_model_dir, create_transl=False, gender='neutral').to(device),
            'male': SMPL(smpl_model_dir, create_transl=False, gender='male').to(device),
            'female': SMPL(smpl_model_dir, create_transl=False, gender='female').to(device)
        }
        smpl = self.smpl_dict['male']
        faces = smpl.faces.copy()
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = smpl.parents.cpu().numpy()
        self.device = device
        self.color_sequences = [
            ['Yellow', 'Green', 'Teal'],
            ['Yellow', 'Red', 'Teal'],
            ['Yellow', 'Blue', 'Teal'],
            ['Yellow', 'Purple', 'Teal'],
            ['Yellow', 'Orange', 'Teal']
        ]
        self.show_skeleton_jpos = show_skeleton_jpos
        self.show_ik_smpl_pose = show_ik_smpl_pose

    def load_default_camera(self):
        return sp.Camera(center=(5, 0, 1.5), look_at=(0, 0, 0.8), up_dir=(0, 0, 1), 
                         fov_y_degrees=45.0, aspect_ratio=1.0)


    def vis_smpl_scene(self, smpl_seq=None, html_path=None, window_size=(400, 400)):
        scene = self.generate_smpl_scene(smpl_seq, window_size=window_size)
        scene.save_as_html(html_path)
        
    def generate_smpl_scene(self, smpl_seq=None, window_size=None):
        scene = sp.Scene()
        main = scene.create_canvas_3d(width=window_size[0], height=window_size[1])

        if 'pose' in smpl_seq or 'joints_pos' in smpl_seq:  # single person
            smpl_seq = {'skel0': smpl_seq}
        smpl_seq = {k: v.copy() for k, v in smpl_seq.items()}   # copy to avoid inplace modification

        num_fr = -1
        for i, (skel_name, pose_dict) in enumerate(smpl_seq.items()):
            colors = self.color_sequences[i % len(self.color_sequences)]
            normal_shape_len = {'pose': 2, 'trans': 2, 'shape': 2, 'joints_pos': 3}
            for key in ['pose', 'trans', 'shape', 'joints_pos']:
                if key in pose_dict and len(pose_dict[key].shape) > normal_shape_len[key]:
                    pose_dict[key] = pose_dict[key][0]

            if self.show_ik_smpl_pose and 'pose' in pose_dict:
                pose_dict['skeleton_fk'] = SkeletonActor(scene, skel_name, self.smpl_joint_parents, joint_color=colors[0], bone_color=colors[1])

                pose = pose_dict['pose'].to(self.device)
                trans = pose_dict['trans'].to(self.device)
                shape = pose_dict['shape'].to(self.device)
                num_fr = max(num_fr, pose.shape[0])
            
                # print(pose[..., :3].view(-1, 3))
                gender = pose_dict.get('gender', 'neutral')
                smpl_motion = self.smpl_dict[gender](
                    global_orient=pose[..., :3],
                    body_pose=pose[..., 3:],
                    betas=shape,
                    root_trans=trans,
                    return_full_pose=True,
                    orig_joints=True
                )
                smpl_joints = smpl_motion.joints            
                pose_dict['joints_fk'] = smpl_joints
                if 'offset' in pose_dict:
                    pose_dict['joints_fk'] = pose_dict['joints_fk'] + pose_dict['offset'].to(self.device)

            if 'joints_pos' in pose_dict:
                num_fr = max(num_fr, pose_dict['joints_pos'].shape[0])
                if 'offset' in pose_dict:
                    pose_dict['joints_pos'] = pose_dict['joints_pos'] + pose_dict['offset'].to(self.device)
                pose_dict['skeleton_jpos'] = SkeletonActor(scene, f'{skel_name}_jpos', self.smpl_joint_parents, joint_color=colors[0], bone_color=colors[2] if 'skeleton_fk' in pose_dict else colors[1], joint_constr_color='Brown' if skel_name == 'gt' else 'Cyan')
                if 'keyframe_idx' in pose_dict:
                    pose_dict['skeleton_keyframe_jpos'] = SkeletonActor(scene, f'{skel_name}_keyframe_jpos', self.smpl_joint_parents, joint_color='Orange', bone_color='Yellow')
                if 'target_jpos' in pose_dict:
                    pose_dict['skeleton_keyframe_jpos_unknownt'] = SkeletonActor(scene, f'{skel_name}_keyframe_jpos_unknownt', self.smpl_joint_parents, joint_color='Orange', bone_color='Pink', joint_constr_color='Purple')
        
        if not self.show_skeleton_jpos:
            main.set_layer_settings({f'{skel_name}_jpos': {'filled': False} for skel_name, pose_dict in smpl_seq.items() if 'skeleton_jpos' in pose_dict})


        for fr in range(num_fr):
            main_frame = main.create_frame()
            main_frame.camera = self.load_default_camera()

            for skel_name, pose_dict in smpl_seq.items():
                if 'skeleton_fk' in pose_dict:
                    ind = min(fr, pose_dict['joints_fk'].shape[0] - 1)
                    pose_dict['skeleton_fk'].add_mesh_to_frames(main_frame, pose_dict['joints_fk'][ind].cpu().numpy())
                if 'skeleton_jpos' in pose_dict:
                    ind = min(fr, pose_dict['joints_pos'].shape[0] - 1)
                    if skel_name == 'pred' and pose_dict['is_unknown_t']:
                        target_jpos = pose_dict['target_jpos'].clone()
                        if 'offset' in pose_dict:
                            target_jpos += pose_dict['offset'].to(self.device)
                        if 'skeleton_keyframe_jpos_unknownt' in pose_dict:
                            pose_dict['skeleton_keyframe_jpos_unknownt'].add_mesh_to_frames(main_frame, target_jpos.cpu().numpy())
                        if 'unknownt_local_joints_mask' in pose_dict:
                            joint_mask = pose_dict['unknownt_local_joints_mask'].view(-1, 3)
                            if joint_mask.any():
                                pose_dict['skeleton_keyframe_jpos_unknownt'].add_joint_constr_meshes_to_frames(main_frame, target_jpos[:-2].cpu().numpy(), joint_mask.cpu().numpy())

                    if pose_dict.get('keyframe_idx', None) is not None and any([ind+i in pose_dict['keyframe_idx'] for i in [-1, 0, 1]]):
                        pose_dict['skeleton_keyframe_jpos'].add_mesh_to_frames(main_frame, pose_dict['joints_pos'][ind].cpu().numpy())
                    else:
                        pose_dict['skeleton_jpos'].add_mesh_to_frames(main_frame, pose_dict['joints_pos'][ind].cpu().numpy())
                
                    if 'local_joints_mask' in pose_dict:
                        joint_pos = pose_dict['joints_pos'][ind, :22]
                        if pose_dict['mask_type'] == 'random_feat_mask':
                            joint_mask = pose_dict['local_joints_mask'][ind] > 0.0
                        else:
                            joint_mask = sum([pose_dict['local_joints_mask'][max(0, min(ind+i, pose_dict['local_joints_mask'].shape[0] - 1))] for i in [-1, 0, 1]]) > 0.0
                        joint_mask = joint_mask.view(-1, 3)
                        if joint_mask.any():
                            pose_dict['skeleton_jpos'].add_joint_constr_meshes_to_frames(main_frame, joint_pos.cpu().numpy(), joint_mask.cpu().numpy())
                    
                    if 'global_joint_constr' in pose_dict:
                        joint_constr = pose_dict['global_joint_constr'][:66, 0, ind].view(-1, 3).clone()
                        if 'offset' in pose_dict:
                            joint_constr += pose_dict['offset'].to(self.device)
                        joint_mask = sum([pose_dict['global_joint_mask'][max(0, min(ind+i, pose_dict['global_joint_mask'].shape[0] - 1))] for i in [-1, 0, 1]]) > 0.0
                        joint_mask = joint_mask.view(-1, 3)
                        if joint_mask.any():
                            pose_dict['skeleton_jpos'].add_joint_constr_meshes_to_frames(main_frame, joint_constr.cpu().numpy(), joint_mask.cpu().numpy())
        
            # if 'text' in smpl_seq:
            #     label = scene.create_label(text=smpl_seq['text'], color=sp.Colors.White, size_in_pixels=60, offset_distance=0.0, horizontal_align='center', camera_space=True)
            #     main_frame.add_label(label=label, position=[0.0, 1.5, -5.0])

        return scene


if __name__ == '__main__':

    sp_visualizer = ScenepicVisualizer('data/smpl_data', device='cuda')

    smpl_pose, smpl_trans = torch.load(f'out/smpl_seq.pt')
    ind = 0
    smpl_seq = {
        'pose': smpl_pose[ind],
        'trans': smpl_trans[ind],
        'shape': torch.zeros_like(smpl_pose[ind, :, :10]),
        'gender': 'male'
    }
    smpl_seq2 = smpl_seq.copy()
    # smpl_seq2['trans'] = smpl_seq2['trans'].clone()
    # smpl_seq2['trans'][..., 1] += 0.5
    smpl_seq2['offset'] = torch.tensor([0., 0.8, 0.])

    smpl_seq_all = {
        'skel0': smpl_seq,
        'skel1': smpl_seq2
    }

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

    sp_visualizer.vis_smpl_scene(smpl_seq_all, 'out/test.html')
