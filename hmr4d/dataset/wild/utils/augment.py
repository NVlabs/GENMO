import torch


def augment_betas(betas, std=0.1):
    noise = torch.normal(mean=torch.zeros(10), std=torch.ones(10) * std)
    betas_aug = betas + noise[None]
    return betas_aug


def randomly_modify_hands_legs(j3d, left_hands=[12], right_hands=[13], left_legs=[5, 14, 15, 16], right_legs=[6, 17, 18, 19],
                               p_switch_hand=0.001, p_switch_leg=0.001,
                               p_wrong_hand0=0.001, p_wrong_hand1=0.001,
                               p_wrong_leg0=0.001, p_wrong_leg1=0.001):
    
    L, J, _ = j3d.shape
    j3d_orig = j3d.clone()

    mask = torch.rand(L) < p_switch_hand
    for i, j in zip(left_hands, right_hands):
        j3d[mask, i] = j3d_orig[mask, j]
        j3d[mask, j] = j3d_orig[mask, i]
    mask = torch.rand(L) < p_switch_leg
    for i, j in zip(left_legs, right_legs):
        j3d[mask, i] = j3d_orig[mask, j]
        j3d[mask, j] = j3d_orig[mask, i]
    mask = torch.rand(L) < p_wrong_hand0
    for i, j in zip(left_hands, right_hands):
        j3d[mask, i] = j3d_orig[mask, j]
    mask = torch.rand(L) < p_wrong_hand1
    for i, j in zip(left_hands, right_hands):
        j3d[mask, j] = j3d_orig[mask, i]
    mask = torch.rand(L) < p_wrong_leg0
    for i, j in zip(left_legs, right_legs):
        j3d[mask, i] = j3d_orig[mask, j]
    mask = torch.rand(L) < p_wrong_leg1
    for i, j in zip(left_legs, right_legs):
        j3d[mask, j] = j3d_orig[mask, i]

    return j3d
