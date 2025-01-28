class BONESPose:
    Pelvis = 0
    Spine = 1
    Spine1 = 2
    Spine2 = 3
    Spine3 = 4
    Neck = 5
    Head = 6
    RightShoulder = 7
    RightArm = 8
    RightForeArm = 9
    RightHand = 10
    RightHandEnd = 11
    RightHandThumb1 = 12
    LeftHandThumb2 = 13
    LeftShoulder = 14
    LeftArm = 15
    LeftForeArm = 16
    LeftHand = 17
    LeftHandEnd = 18
    LeftHandThumb1 = 19
    RightHandThumb2 = 20
    RightUpLeg = 21
    RightLeg = 22
    RightFoot = 23
    RightToeBase = 24
    LeftUpLeg = 25
    LeftLeg = 26
    LeftFoot = 27
    LeftToeBase = 28
    Jaw = 29
    LEar = 29
    REar = 30
    LEye = 29
    REye = 30


class SMPLPose:
    Pelvis = 0
    LHip = 1
    RHip = 2
    Torso = 3
    LKnee = 4
    RKnee = 5
    Spine = 6
    LAnkle = 7
    RAnkle = 8
    Chest = 9
    LToe = 10
    RToe = 11
    Neck = 12
    LCollar = 13
    RCollar = 14
    Head = 15
    LShoulder = 16
    RShoulder = 17
    LElbow = 18
    RElbow = 19
    LWrist = 20
    RWrist = 21
    LHand = 22
    RHand = 23
    LEar = 24
    REar = 25
    LEye = 24
    REye = 25


bones_parents = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    4,
    7,
    8,
    9,
    10,
    10,
    12,
    4,
    14,
    15,
    16,
    17,
    17,
    19,
    0,
    21,
    22,
    23,
    0,
    25,
    26,
    27,
]

bones_children_map = [
    1,
    2,
    3,
    4,
    5,
    6,
    -1,
    8,
    9,
    10,
    11,
    -1,
    13,
    -1,
    15,
    16,
    17,
    18,
    -1,
    20,
    -1,
    22,
    23,
    24,
    -1,
    26,
    27,
    28,
    -1,
]

# Optimized beta for Bones unified skeleton
bones_beta = [
    1.5538,
    -1.6340,
    0.9470,
    1.8768,
    0.0643,
    0.3386,
    -0.4611,
    -0.0658,
    0.2195,
    -0.0370,
]

bones_joints_rest_jaw = [[-0.0033, -0.0056, 0.6661]]
bones_joints_rest_ear = [[0.0665, 0.0360, 0.7417], [-0.0939, 0.0176, 0.7328]]
bones_joints_rest_eye = [[0.0445, -0.0572, 0.7739], [-0.0481, -0.0681, 0.7717]]
smpl_ear_indices = [516, 4004]
smpl_eye_indices = [2811, 6276]
