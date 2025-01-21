python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=0 num_parts=2 trial_ind=1 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=1 num_parts=2 trial_ind=1 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=0 num_parts=2 trial_ind=2 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=1 num_parts=2 trial_ind=2 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=0 num_parts=2 trial_ind=3 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=1 num_parts=2 trial_ind=3 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=0 num_parts=2 trial_ind=4 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"

python tools/test_slurm.py -u jinkunc -cmd "python tools/train_v2.py part_ind=1 num_parts=2 trial_ind=4 global/task=mv2d/save_text2motion_feats exp=unimfm/unimfm_est_st_norm_di_lg  exp_name_var=g8 +test_checkpoint=s050000 logger.entity=nv-welcome"