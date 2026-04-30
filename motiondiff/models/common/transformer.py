import torch


def generate_ar_mask(
    tgt_size,
    src_size,
    tgt_start_dim=0,
    src_start_dim=0,
    tgt_end_dim=None,
    src_end_dim=None,
):
    mask = torch.zeros(tgt_size, src_size)
    if src_end_dim is None:
        src_end_dim = src_size
    if tgt_end_dim is None:
        tgt_end_dim = tgt_size
    mask_ = torch.triu(
        torch.full(
            (tgt_end_dim - tgt_start_dim, src_end_dim - src_start_dim), float("-inf")
        ),
        diagonal=1,
    )
    mask[tgt_start_dim:tgt_end_dim, src_start_dim:src_end_dim] = mask_
    # for i in range(tgt_end_dim - tgt_start_dim):
    # mask[tgt_start_dim + i, src_start_dim + i + 1: src_end_dim] = float('-inf')
    return mask
