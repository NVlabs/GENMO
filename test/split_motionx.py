import numpy as np
import torch

if __name__ == "__main__":
    data = torch.load(
        "inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3_aligned.pth"
    )
    data_subset_dict = {}
    for vid in data:
        subset = data[vid]["subset"]
        if subset not in data_subset_dict:
            data_subset_dict[subset] = []
        data_subset_dict[subset].append(vid)

    # split into train/val/test, train: 80%, val: 5%, test: 15%
    data_train = {}
    data_val = {}
    data_test = {}
    for subset in data_subset_dict:
        np.random.shuffle(data_subset_dict[subset])
        train_list = data_subset_dict[subset][
            : int(len(data_subset_dict[subset]) * 0.8)
        ]
        val_list = data_subset_dict[subset][
            int(len(data_subset_dict[subset]) * 0.8) : int(
                len(data_subset_dict[subset]) * 0.85
            )
        ]
        test_list = data_subset_dict[subset][
            int(len(data_subset_dict[subset]) * 0.85) :
        ]

        for vid in train_list:
            data_train[vid] = data[vid]
        for vid in val_list:
            data_val[vid] = data[vid]
        for vid in test_list:
            data_test[vid] = data[vid]

    torch.save(
        data_train, "inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3_train.pth"
    )
    torch.save(data_val, "inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3_val.pth")
    torch.save(
        data_test, "inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3_test.pth"
    )
