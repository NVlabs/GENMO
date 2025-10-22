import os

import hydra
import pytorch_lightning as pl
import torch
from moviepy.audio.AudioClip import AudioArrayClip

from genmo.utils.vis_utils import visualize_smpl_scene_mp4
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx


class VisSpeech(pl.Callback):
    def __init__(
        self,
        vis_every_n_val=1,
        save_feats=False,
        save_dir=None,
        dataset_part_ind=-1,
        endecoder=None,
        trial_ind=0,
        text_len=120,
    ):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0
        self.save_feats = save_feats
        self.trial_ind = trial_ind
        self.save_dir = save_dir
        self.text_len = text_len
        self.dataset_part_ind = dataset_part_ind
        if endecoder is not None:
            self.endecoder = hydra.utils.instantiate(endecoder).cuda()
        # vid->result

        # SMPL
        self.smplx_model = {
            "male": make_smplx("supermotion_smpl24"),
            "female": make_smplx("supermotion_smpl24"),
            "neutral": make_smplx("supermotion_smpl24"),
        }
        self.smplx = make_smplx("supermotion")
        self.smplx2smpl = torch.load(
            "inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
        )

        self.J_regressor = torch.load(
            "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
        )
        self.smplx2smpl = torch.load(
            "inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
        )
        self.faces_smpl = make_smplx("smpl").faces
        self.faces_smplx = self.smplx_model["neutral"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = (
            self.on_predict_batch_end
        )

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = (
            self.on_predict_epoch_end
        )

        # Only validation record the metrics with logger
        self.on_test_epoch_start = self.on_validation_epoch_start = (
            self.on_predict_epoch_start
        )

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        data_name = batch["meta"][0]["data_name"]
        vid = batch["meta"][0]["vid"]

        if data_name not in ["beat2"]:
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.smplx = self.smplx.cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        # print(batch_idx)
        # os.makedirs('out/motions', exist_ok=True)
        # torch.save(outputs, f'out/motions/outputs_{batch_idx}.pt')
        # if 'multi_text_data' in batch['meta'][0]:
        #     multi_text_data = batch['meta'][0]['multi_text_data']
        #     for i in range(len(multi_text_data['vid'])):
        #         print(multi_text_data['vid'][i], multi_text_data['caption'][i])

        # seq_length = batch["length"][0].item()
        gender = "neutral"
        smpl_key = (
            "2d_pred_smpl_params_global"
            if data_name == "motion-x++2d"
            else "pred_smpl_params_global"
        )

        # Groundtruth (world, cam)
        if data_name == "beat2":
            target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
            target_w_j3d = self.smplx_model[gender](**target_w_params)
            offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
            target_w_j3d = target_w_j3d + offset
            # target_w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices])
            # target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)

        # 2. ay
        pred_smpl_params_global = outputs[smpl_key]
        # pred_ay_j3d = self.smplx_model["neutral"](**pred_smpl_params_global)
        pred_smplx = self.smplx(**pred_smpl_params_global)
        pred_ay_verts = torch.stack(
            [torch.matmul(self.smplx2smpl, v_) for v_ in pred_smplx.vertices]
        )

        # pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        # pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
        audio_array = batch["audio_array"][0].cpu().numpy()
        audio_fps = 18000
        audio_clip = AudioArrayClip(audio_array[:, None].repeat(2, 1), audio_fps)
        if self.save_feats:
            encoder_inputs = {
                "smpl_params_w": {
                    k: v.unsqueeze(0) for k, v in outputs[smpl_key].items()
                },
            }
            feats = self.endecoder.encode_humanml3d(encoder_inputs)
            self.feats_arr.append(feats)
            # self.text_arr.append(text)
        else:
            # Visualize
            if trainer.global_rank == 0 and self.num_val % self.vis_every_n_val == 0:
                wandb_dict = visualize_smpl_scene_mp4(
                    f"vis_text_global_{data_name}",
                    batch_idx,
                    vid,
                    pred_ay_verts,
                    target_w_j3d,
                    transform_mode="global",
                    faces_smpl=self.faces_smpl,
                    J_regressor=self.J_regressor,
                    audio_clip=audio_clip,
                )
                self.wandb_html_dict.update(wandb_dict)
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wandb_html_dict = {}
        if self.save_feats:
            self.feats_arr = []
            self.text_arr = []
            print(
                f"start generating text-to-motion features which will be saved at {self.save_dir}\n"
            )
            print(
                "#### saving a dump feature first to check the correctness of the path #### \n"
            )
            dump_feats = torch.randn(1, 1, 1, 1)
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(dump_feats, self.save_dir + "/dump.pt")
            print(f"dump feature saved to {self.save_dir}/dump.pt\n")

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        self.num_val += 1
        if len(self.wandb_html_dict) > 0:
            pl_module.logger.log_metrics(self.wandb_html_dict)
        if self.save_feats:
            feats_arr = torch.cat(self.feats_arr, dim=0).cpu()
            results = {
                "feats": feats_arr,
                "text": self.text_arr,
            }
            os.makedirs(self.save_dir, exist_ok=True)
            if self.dataset_part_ind >= 0:
                fname = (
                    self.save_dir
                    + f"/new_feats_part{self.dataset_part_ind}_len{self.text_len}_{self.trial_ind}.pt"
                )
            else:
                fname = (
                    self.save_dir + f"/new_feats_len{self.text_len}_{self.trial_ind}.pt"
                )
            torch.save(results, fname)
            os.chmod(fname, 0o755)
            print(f"text-to-motion features saved to {fname}")
