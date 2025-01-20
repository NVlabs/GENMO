import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import sys
sys.path.append('./')
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from motiondiff.models.common.utils.human_models import smpl_x, smpl
from motiondiff.utils.torch_utils import tensor_to, interp_scipy_ndarray, slerp_joint_rots
from motiondiff.data_pipeline.datasets.kp2d_dataset import KP2DDataset


def get_3d(human_model_param, human_model_type):
    if human_model_type == 'smplx':
        human_model = smpl_x
        rotation_valid = np.ones((smpl_x.orig_joint_num), dtype=np.float32)
        coord_valid = np.ones((smpl_x.joint_num), dtype=np.float32)

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
                                             human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param and human_model_param['lhand_valid']:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['lhand']] = 0
            coord_valid[smpl_x.joint_part['lhand']] = 0
        if 'rhand_pose' in human_model_param and human_model_param['rhand_valid']:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['rhand']] = 0
            coord_valid[smpl_x.joint_part['rhand']] = 0
        if 'jaw_pose' in human_model_param and 'expr' in human_model_param and human_model_param['face_valid']:
            jaw_pose = human_model_param['jaw_pose']
            expr = human_model_param['expr']
            expr_valid = True
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['face']] = 0
            coord_valid[smpl_x.joint_part['face']] = 0
            expr_valid = False
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # get mesh and joint coordinates
        zero_pose = torch.zeros((1, 3)).float()  # eye poses
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
        mesh_cam = output.vertices[0].numpy()
        joint_world = output.joints[0].numpy()[smpl_x.joint_idx, :].copy()

        # concat root, body, two hands, and jaw pose
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose))
    
    elif human_model_type == 'smpl':
        human_model = smpl
        pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        pose = torch.FloatTensor(pose).view(-1, 3)
        shape = torch.FloatTensor(shape).view(1, -1);
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation

        # get mesh and joint coordinates
        root_pose = pose[smpl.orig_root_joint_idx].view(1, 3)
        body_pose = torch.cat((pose[:smpl.orig_root_joint_idx, :], pose[smpl.orig_root_joint_idx + 1:, :])).view(1, -1)
        with torch.no_grad():
            output = smpl.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans)
        mesh_cam = output.vertices[0].numpy()

    joint_trunc = None

    # return results
    if human_model_type == 'smplx':
        pose = pose.reshape(-1)
        expr = expr.numpy().reshape(-1)
        return joint_world, joint_trunc, pose, shape, expr, rotation_valid, coord_valid, expr_valid
    elif human_model_type == 'smpl':
        pose = pose.reshape(-1)
        return joint_world, joint_trunc, pose, shape
    elif human_model_type == 'mano':
        pose = pose.reshape(-1)
        return joint_world, joint_trunc, pose, shape
    

class Human36M(KP2DDataset):
    def __init__(self, data_dir, split, num_keypoints=14, num_frames=196, img_w=1024, img_h=1024, num_views=4, cam_radius=8, cam_elevation=10, focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, **kwargs):
        super(Human36M, self).__init__(num_frames, img_w, img_h, num_views, cam_radius, cam_elevation, focal_scale, synthetic_view_type, normalize_type, bbox_scale)
        self.transform = transforms.ToTensor()
        self.split = split
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.img_dir = os.path.join(data_dir, 'images')
        self.annot_path = os.path.join(data_dir, 'annotations')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.joint_set = {
            'joint_num': 17,
            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
            'flip_pairs': ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)),
            'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
            'regressor': np.load(os.path.join(data_dir, 'J_regressor_h36m_smplx.npy'))
        }
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')
        self.datalist = self.load_data()

    def get_subsampling_ratio(self):
        if self.split == 'train':
            return 5
        elif self.split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.split == 'train':
            subject = [1, 5, 6, 7, 8]
        elif self.split == 'test':
            subject = [9, 11]
        else:
            assert 0, print("Unknown subset")
        return subject

    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        db = COCO()
        cameras = {}
        joints = {}
        smplx_params = {}
        for subject in subject_list:
            with open(os.path.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
                if len(db.dataset) == 0:
                    for k, v in annot.items():
                        db.dataset[k] = v
                else:
                    for k, v in annot.items():
                        db.dataset[k] += v
            with open(os.path.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            with open(os.path.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints[str(subject)] = json.load(f)
            with open(os.path.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPLX_NeuralAnnot.json'), 'r') as f:
                smplx_params[str(subject)] = json.load(f)
        db.createIndex()
        datalist = []
        for subject in smplx_params.keys():
            for action_idx in smplx_params[subject].keys():
                for subaction_idx in smplx_params[subject][action_idx].keys():
                    datalist.append({
                        'smplx_param': smplx_params[subject][action_idx][subaction_idx],
                    })
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        raw_smplx_param = data['smplx_param']
        smplx_params = {k: [] for k in raw_smplx_param['0'].keys()}
        smplx_params['joint_world'] = []
        for frame, frame_dict in raw_smplx_param.items():
            if len(smplx_params['joint_world']) > 300:
                break
            smplx_per_frame = {}
            for key, value in frame_dict.items():
                if key in {'root_pose', 'body_pose'}:
                    raw_param = np.array(value).reshape(-1, 3)
                elif key in {'shape', 'trans'}:
                    raw_param = np.array(value).reshape(1, -1)
                smplx_per_frame[key] = raw_param
                smplx_params[key].append(raw_param)
            smplx_joint_world, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid = get_3d(smplx_per_frame, 'smplx')
            smplx_joint_world = smplx_joint_world[:self.num_keypoints]
            smplx_params['joint_world'].append(smplx_joint_world)
        for key in smplx_params:
            smplx_params[key] = np.stack(smplx_params[key], axis=0)
            
        motion_3d = interp_scipy_ndarray(smplx_params['joint_world'], scale=0.4, dim=0)     # from 50 fps to 20 fps
        motion, cam_dict = self.generate_cam_and_2d_motion(motion_3d)
        motion, m_length = self.pad_motion(motion)
        text = ''   # no text data
        return text, motion, m_length, cam_dict

if __name__ == '__main__':
    dataset = Human36M(data_dir='/home/yey/repo/OSX/dataset/Human36M', split='test')
    for i in range(10):
        batch = dataset[i]
        print(f'data {i}')
