import bisect
import math
import os
import random

import numpy as np

occ_interval_amb = 8
pixel_std = 1
_aspect_ratio = 1
occ_jts_list = [
    [0, 1, 2, 3, 4, 5],
    [6, 7, 10, 11, 12, 13],
]
occ_jts_list = [
    [11, 12, 13, 14, 15, 16],
    [0, 1, 2, 3, 4, 7, 8, 9, 10]
]


def get_bbox_kp(img_idx, old_joints2ds, old_bboxs, scale_mult):
    old_bbox = old_bboxs[img_idx]
    scaled_bbox = adjust_bbox(old_bbox, scale_mult=scale_mult)

    # kp_2d = old_joints2ds[img_idx].reshape(-1, 3)
    # if kp_2d.shape[0] != 14:
    #     kp_2d = convert_kps(kp_2d.reshape(1, -1, 3), src="spin", dst="common").reshape(
    #         -1, 3
    #     )
    # assert kp_2d.shape[0] == 14
    return scaled_bbox


def rand_img_clip_transforms2(
    img_idx, frame_idx, max_idx, old_joints2ds, old_bboxs, scale_mult, img_path=None, imgscale=None
):
    local_random = random.Random(0)
    if frame_idx % occ_interval_amb == 0:
        local_random.seed(img_idx)
        scaled_bbox = get_bbox_kp(img_idx, old_joints2ds, old_bboxs, scale_mult)
        center, scale, _ = rand_img_clip_transform(
            old_joints2ds[img_idx, :, :2],
            old_joints2ds[img_idx, :, 2],
            scaled_bbox,
            local_random,
            imgscale,
        )

        return center, scale

    interv = frame_idx % occ_interval_amb
    last_img_idx = img_idx - interv
    next_img_idx = last_img_idx + occ_interval_amb

    if last_img_idx < 0:
        # print('smaller!', img_path)
        last_img_idx = 0
    if next_img_idx > max_idx - 1:
        # print('larger!', img_path)
        next_img_idx = max_idx - 1

    local_random.seed(last_img_idx)
    scaled_bbox = get_bbox_kp(last_img_idx, old_joints2ds, old_bboxs, scale_mult)
    center0, scale0, _ = rand_img_clip_transform(
        old_joints2ds[last_img_idx, :, :2],
        old_joints2ds[last_img_idx, :, 2],
        scaled_bbox,
        local_random,
        imgscale,
    )

    local_random.seed(next_img_idx)
    scaled_bbox = get_bbox_kp(next_img_idx, old_joints2ds, old_bboxs, scale_mult)
    center1, scale1, _ = rand_img_clip_transform(
        old_joints2ds[next_img_idx, :, :2],
        old_joints2ds[next_img_idx, :, 2],
        scaled_bbox,
        local_random,
        imgscale,
    )

    interp_coef = (img_idx - last_img_idx) * 1.0 / occ_interval_amb
    interp_f = lambda x, y: interp_coef * y + (1 - interp_coef) * x  # noqa: E731

    center = interp_f(center0[0], center1[0]), interp_f(center0[1], center1[1])
    scale = interp_f(scale0[0], scale1[0]), interp_f(scale0[1], scale1[1])

    # print(scale0[0], scale1[0])

    return np.array(center), np.array(scale)


def rand_img_clip_transform(joints, joints_vis, bbox_origin, local_random, imgscale=None):
    # occ_jts_list: list of list of jts indices
    part_joints_vis = [
        np.array([joints_vis[idx] for idx in occ_jts]) for occ_jts in occ_jts_list
    ]
    part_jts_is_visible = [
        item.sum() * 1.0 / len(item) > 0.5 for item in part_joints_vis
    ]

    valid_occ_jts_list = []
    for i in range(len(occ_jts_list)):
        if part_jts_is_visible[i]:
            valid_occ_jts_list.append(occ_jts_list[i])

    if len(valid_occ_jts_list) == 0:
        # print("????????????????????????????????? occlusion_utils")
        return bbox_origin[0], bbox_origin[1], 1

    len_occ = len(valid_occ_jts_list)
    prob_intervals = [0.667 / len_occ * (i + 1) for i in range(len_occ)]
    p = local_random.random()
    occ_idx = bisect.bisect_right(prob_intervals, p)
    if occ_idx == len_occ:
        # return None, None
        return bbox_origin[0], bbox_origin[1], 1

    # print(prob_intervals)
    selected_joints = []
    selected_ids = []
    for joint_id in range(17):
        if joints_vis[joint_id] > 0.5 and (joint_id not in valid_occ_jts_list[occ_idx]):
            selected_joints.append(joints[joint_id])
            selected_ids.append(joint_id)

    selected_joints = np.array(selected_joints, dtype=np.float32)
    left_top = np.amin(selected_joints, axis=0)
    right_bottom = np.amax(selected_joints, axis=0)
    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]

    # assert w > 0.5 and h > 0.5, f'{selected_joints}, {selected_ids}, {p}, {joints_vis}'
    if w < 30 or h < 30:
        return bbox_origin[0], bbox_origin[1], 1

    center = (
        (right_bottom[0] + left_top[0]) * 0.5,
        (right_bottom[1] + left_top[1]) * 0.5,
    )
    center = np.array(center)
    # rand_center_shift = np.random.randn(2) * 5
    if imgscale is not None:
        rand_center_shift = np.array(
            [local_random.random() * imgscale / 5 - imgscale / 10, local_random.random() * imgscale / 5 - imgscale / 10]
        )
    else:
        rand_center_shift = np.array(
            [local_random.random() * 200 - 100, local_random.random() * 200 - 100]
        )
    # rand_center_shift = 0
    center = center + rand_center_shift

    if w > _aspect_ratio * h:
        h = w * 1.0 / _aspect_ratio
    elif w < _aspect_ratio * h:
        w = h * _aspect_ratio

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)

    scale = scale * (1.15 + local_random.random() * 0.4 - 0.2)

    # print(joints, joints_vis, selected_ids)
    # print(left_top, right_bottom, scale)

    return center, scale, left_top


def get_synth_sizes(img_idx, bbox, imgwidth, imght):
    # interpolation in some frames to get a more continuous occlusion
    if img_idx % occ_interval_amb == 0:
        return get_synth_size(img_idx, bbox, imgwidth, imght)

    last_img_idx = (img_idx // occ_interval_amb) * occ_interval_amb
    next_img_idx = last_img_idx + occ_interval_amb

    synth_xmin0, synth_ymin0, synth_w0, synth_h0 = get_synth_size(
        last_img_idx, bbox, imgwidth, imght
    )
    synth_xmin1, synth_ymin1, synth_w1, synth_h1 = get_synth_size(
        next_img_idx, bbox, imgwidth, imght
    )

    interp_coef = (img_idx - last_img_idx) * 1.0 / occ_interval_amb

    interp_f = lambda x, y: interp_coef * y + (1 - interp_coef) * x  # noqa: E731

    return (
        interp_f(synth_xmin0, synth_xmin1),
        interp_f(synth_ymin0, synth_ymin1),
        interp_f(synth_w0, synth_w1),
        interp_f(synth_h0, synth_h1),
    )


def get_synth_size(img_idx, bbox, imgwidth, imght):
    local_random = random.Random(img_idx)
    xmin, ymin, xmax, ymax = bbox
    # print(bbox, imgwidth, imght)
    i = 0
    while True:
        i += 1
        # if i > 100:
        #   print(bbox, imgwidth, imght)
        #   print(synth_xmin, synth_ymin, synth_w, synth_h)
        #   raise NotImplementedError
        if i > 200:
            # raise NotImplementedError
            return 0, 0, 0, 0

        area_min = 0.0
        area_max = 0.3
        synth_area = (
            (local_random.random() * (area_max - area_min) + area_min)
            * (xmax - xmin)
            * (ymax - ymin)
        )

        ratio_min = 0.5
        ratio_max = 1 / 0.5
        synth_ratio = local_random.random() * (ratio_max - ratio_min) + ratio_min

        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = local_random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = local_random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        # if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
        # synth_xmin = max(0, int(synth_xmin))
        # synth_ymin = max(0, int(synth_ymin))
        # synth_w = max(min(imgwidth-synth_xmin, int(synth_w)), 1)
        # synth_h = max(min(imght-synth_ymin, int(synth_h)), 1)

        # if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
        #     return synth_xmin, synth_ymin, synth_w, synth_h
        if (
            synth_xmin >= 0
            and synth_ymin >= 0
            and synth_xmin + synth_w < imgwidth
            and synth_ymin + synth_h < imght
        ):
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)

            return synth_xmin, synth_ymin, synth_w, synth_h


def adjust_bbox(bbox_center_scale, aspect_ratio=1, scale_mult=1.25):
    xmin, ymin, xmax, ymax = (
        bbox_center_scale[0] - 0.5 * bbox_center_scale[2],
        bbox_center_scale[1] - 0.5 * bbox_center_scale[3],
        bbox_center_scale[0] + 0.5 * bbox_center_scale[2],
        bbox_center_scale[1] + 0.5 * bbox_center_scale[3],
    )

    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio, scale_mult=scale_mult
    )

    return center, scale


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def convert_kps(joints2d, src, dst):
    src_names = eval(f"get_{src}_joint_names")()
    dst_names = eval(f"get_{dst}_joint_names")()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


def get_spin_joint_names():
    return [
        "OP Nose",  # 0
        "OP Neck",  # 1
        "OP RShoulder",  # 2
        "OP RElbow",  # 3
        "OP RWrist",  # 4
        "OP LShoulder",  # 5
        "OP LElbow",  # 6
        "OP LWrist",  # 7
        "OP MidHip",  # 8
        "OP RHip",  # 9
        "OP RKnee",  # 10
        "OP RAnkle",  # 11
        "OP LHip",  # 12
        "OP LKnee",  # 13
        "OP LAnkle",  # 14
        "OP REye",  # 15
        "OP LEye",  # 16
        "OP REar",  # 17
        "OP LEar",  # 18
        "OP LBigToe",  # 19
        "OP LSmallToe",  # 20
        "OP LHeel",  # 21
        "OP RBigToe",  # 22
        "OP RSmallToe",  # 23
        "OP RHeel",  # 24
        "rankle",  # 25
        "rknee",  # 26
        "rhip",  # 27
        "lhip",  # 28
        "lknee",  # 29
        "lankle",  # 30
        "rwrist",  # 31
        "relbow",  # 32
        "rshoulder",  # 33
        "lshoulder",  # 34
        "lelbow",  # 35
        "lwrist",  # 36
        "neck",  # 37
        "headtop",  # 38
        "hip",  # 39 'Pelvis (MPII)', # 39
        "thorax",  # 40 'Thorax (MPII)', # 40
        "Spine (H36M)",  # 41
        "Jaw (H36M)",  # 42
        "Head (H36M)",  # 43
        "nose",  # 44
        "leye",  # 45 'Left Eye', # 45
        "reye",  # 46 'Right Eye', # 46
        "lear",  # 47 'Left Ear', # 47
        "rear",  # 48 'Right Ear', # 48
    ]


def get_common_joint_names():
    return [
        "rankle",  # 0  "lankle",    # 0
        "rknee",  # 1  "lknee",     # 1
        "rhip",  # 2  "lhip",      # 2
        "lhip",  # 3  "rhip",      # 3
        "lknee",  # 4  "rknee",     # 4
        "lankle",  # 5  "rankle",    # 5
        "rwrist",  # 6  "lwrist",    # 6
        "relbow",  # 7  "lelbow",    # 7
        "rshoulder",  # 8  "lshoulder", # 8
        "lshoulder",  # 9  "rshoulder", # 9
        "lelbow",  # 10  "relbow",    # 10
        "lwrist",  # 11  "rwrist",    # 11
        "neck",  # 12  "neck",      # 12
        "headtop",  # 13  "headtop",   # 13
    ]





def translate_img_path(img_path_origin, dataset_name):
    if dataset_name == "h36m":
        img_path_split = img_path_origin.split("/")
        img_path = os.path.join(
            "data/h36m/images", img_path_split[-2], img_path_split[-1]
        )
    elif dataset_name == "3doh" or dataset_name == "agora":
        img_path = img_path_origin
    elif dataset_name == "pw3d" or dataset_name == "3dpw":
        correct_img_name = img_path_origin.split("/")[-2:]
        correct_img_name = correct_img_name[0] + "/" + correct_img_name[1]

        img_path = os.path.join("data/pw3d/imageFiles", correct_img_name)
    else:
        raw_img_path = img_path_origin.split("/")[2:]
        if raw_img_path[0][0] == "S":  # train
            img_path = translate_mpii3d_imgname([img_path_origin])[0]
        else:
            img_path = translate_mpii3d_imgname2([img_path_origin])[0]

    return img_path


def translate_mpii3d_imgname(raw_names):
    # video_raw example: 'data/mpii_3d/S1/Seq1/video_0/000001.jpg'
    # actual video_name: 'data/mpii_3d/S1/Seq1/images/S1_Seq1_V0/img_S1_Seq1_V0_000001.jpg
    # print(raw_names.shape)
    new_names = []
    for item in raw_names:
        name_raw = item.split("/")
        parent_paths = name_raw[:4]

        subject = name_raw[2]
        seq = name_raw[3]
        video = name_raw[4].split("_")[-1]
        img_num = name_raw[-1]

        new_name = f"{subject}_{seq}_V{video}/img_{subject}_{seq}_V{video}_{img_num}"
        new_path = os.path.join(
            "data/3dhp/mpi_inf_3dhp_train_set",
            parent_paths[2],
            parent_paths[3],
            "images",
            new_name,
        )

        new_names.append(new_path)
    return np.array(new_names)


def translate_mpii3d_imgname2(raw_names):
    # video_raw example:
    # 'data/mpii_3d/mpi_inf_3dhp_test_set/TS1/images/mpi_inf_3dhp_test_set_TS1_VimageSequence/img_mpi_inf_3dhp_test_set_TS1_VimageSequence_img_000025.jpg'
    # actual video_name: 'mpii3d_test/TS1/imageSequence/img_000001.jpg'
    # print(raw_names.shape)
    new_names = []
    for item in raw_names:
        name_raw = item.split("/")
        ts = name_raw[3]
        img_name = name_raw[-1]
        img_index = img_name.split("_")[-1]

        new_path = f"data/3dhp/mpi_inf_3dhp_test_set/{ts}/imageSequence/img_{img_index}"

        new_names.append(new_path)
    return np.array(new_names)
