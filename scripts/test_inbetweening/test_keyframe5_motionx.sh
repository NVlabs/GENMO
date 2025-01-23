# humanml3d only
python tools/train_v2.py global/task=mv2d/test_inpainting exp=unimfm/unimfm_test_st_mha_n1_lg exp_name_var=g8 +test_checkpoint=s050000 model_cfg.diffusion.test_timestep_respacing="'50'" \
model_cfg.inpainting_3d.mode=body_pose_root_rot_keyframe5 test_datasets=[motionx_test] test_datasets.motionx_test.motion_start_mode=sample_early test_datasets.motionx_test.max_num_motions=128 

# without motionx
python tools/train_v2.py global/task=mv2d/test_inpainting exp=unimfm/unimfm_est_st_norm_di_lg exp_name_var=g8 +test_checkpoint=s050000 model_cfg.diffusion.test_timestep_respacing="'50'" \
model_cfg.inpainting_3d.mode=body_pose_root_rot_keyframe5 test_datasets=[motionx_test] test_datasets.motionx_test.motion_start_mode=sample_early test_datasets.motionx_test.max_num_motions=128

# with motionx
python tools/train_v2.py global/task=mv2d/test_inpainting exp=unimfm/unimfm_est_st_norm_di_lg_mx2_cp1 exp_name_var=g8 +test_checkpoint=s020000 model_cfg.diffusion.test_timestep_respacing="'50'" \
model_cfg.inpainting_3d.mode=body_pose_root_rot_keyframe5 test_datasets=[motionx_test] test_datasets.motionx_test.motion_start_mode=sample_early test_datasets.motionx_test.max_num_motions=128

# without estimation mode
python tools/train_v2.py global/task=mv2d/test_inpainting exp=unimfm/gen_wtext_grep_reg_mx exp_name_var='humanml3d' +test_checkpoint=s200000 model_cfg.diffusion.test_timestep_respacing="'50'" \
model_cfg.inpainting_3d.mode=body_pose_root_rot_keyframe5 remote_results_path=/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/jiefengl/gvhmr test_datasets=[motionx_test] test_datasets.motionx_test.motion_start_mode=sample_early test_datasets.motionx_test.max_num_motions=128

# ours new
python tools/train_v2.py global/task=mv2d/test_inpainting exp=unimfm/unimfm_est_st_norm_di_lg_mx3_cp4 exp_name_var=g8 +test_checkpoint=s050000 model_cfg.diffusion.test_timestep_respacing="'50'" \
model_cfg.inpainting_3d.mode=body_pose_root_rot_keyframe5 test_datasets=[motionx_test] test_datasets.motionx_test.motion_start_mode=sample_early test_datasets.motionx_test.max_num_motions=128 +rsync_ckpt=true
