import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
from os.path import join as pjoin

import clip
# from transformers import AutoTokenizer, CLIPTextModelWithProjection

from motiondiff.utils.mdm_modules import (
    MotionEncoderBiGRUCo,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)
from motiondiff.utils.motionx_modules import (
    MotionDecoderWithDropout,
    MotionEncoderBiGRUCoWithDropout,
    MovementConvEncoderWithDropout,
)

import argparse


def build_motionx_evaluators(opt):
    movement_enc = MovementConvEncoder(
        opt["dim_pose"], opt["dim_movement_enc_hidden"], opt["dim_movement_latent"]
    )

    text_enc = TextEncoderMLP(input_dim=512, hidden_dim=512, output_dim=512)

    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt["dim_movement_latent"],
        hidden_size=opt["dim_motion_hidden"],
        output_size=opt["dim_coemb_hidden"],
        device=opt["device"],
    )

    ckpt_dir = "./t2m_checkpoints/motionx"

    checkpoint = torch.load(
        pjoin(
            opt["checkpoints_dir"],
            ckpt_dir,
            "text_mot_match/model_10.0_clip_fullset",
            "finest.tar",
        ),
        map_location=opt["device"],
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(
        "Loading Evaluation Model Wrapper (Epoch %d) Completed!!"
        % (checkpoint["epoch"])
    )
    return text_enc, motion_enc, movement_enc

def build_humanml3d_evaluators(opt):
    movement_enc = MovementConvEncoder(
        opt["dim_pose"] - 4, opt["dim_movement_enc_hidden"], opt["dim_movement_latent"]
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=opt["dim_word"],
        pos_size=opt["dim_pos_ohot"],
        hidden_size=opt["dim_text_hidden"],
        output_size=opt["dim_coemb_hidden"],
        device=opt["device"],
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt["dim_movement_latent"],
        hidden_size=opt["dim_motion_hidden"],
        output_size=opt["dim_coemb_hidden"],
        device=opt["device"],
    )

    ckpt_dir = "./inputs/t2m"
    checkpoint = torch.load(
        pjoin(
            opt["checkpoints_dir"], ckpt_dir, "text_mot_match", "model", "finest.tar"
        ),
        map_location=opt["device"],
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(
        "Loading Evaluation Model Wrapper (Epoch %d) Completed!!"
        % (checkpoint["epoch"])
    )
    return text_enc, motion_enc, movement_enc

def build_evaluators(args, opt):
    if args.dataset == 'humanml3d':
        """
            Our self-trained text-motion encoder by pairing the latent representation
            as in t2m: https://github.com/EricGuo5513/text-to-motion
        """
        text_enc, motion_enc, movement_enc = build_humanml3d_evaluators(opt)
    elif args.dataset == 'motionx':
        """
            The text encoder used by text-to-motion:
            https://github.com/EricGuo5513/text-to-motion
            provided by training on paired text-motion data on humanml3d 
        """
        text_enc, motion_enc, movement_enc = build_motionx_evaluators(opt)
    else:
        assert False, "Invalid text encoder"

    return text_enc, motion_enc, movement_enc


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


# from action2motion
def evaluate_matching_score(text_embeddings_all, motion_embeddings_all):
    # match_score_dict = OrderedDict({})
    # R_precision_dict = OrderedDict({})
    # activation_dict = OrderedDict({})

    all_motion_embeddings = []
    score_list = []
    all_size = 0
    matching_score_sum = 0
    top_k_count = 0
    # print(motion_loader_name)
    batch_size = 64
    batch_num = len(text_embeddings_all) // batch_size
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            text_embeddings = text_embeddings_all[i * batch_size : (i + 1) * batch_size]
            motion_embeddings = motion_embeddings_all[
                i * batch_size : (i + 1) * batch_size
            ]
            dist_mat = euclidean_distance_matrix(
                text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
            )
            matching_score_sum += dist_mat.trace()

            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)

            all_size += text_embeddings.shape[0]

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size

    return matching_score, R_precision, all_motion_embeddings


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(
        statistics_1[0], statistics_1[1], statistics_2[0], statistics_2[1]
    )


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def evaluate_diversity(feats, diversity_times=300):
    diversity = calculate_diversity(feats, diversity_times)
    return diversity


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def evaluate_multimodality(feats, mm_num_times=10):
    mm_motion_embeddings = feats
    score = calculate_multimodality(mm_motion_embeddings, mm_num_times)
    return score


def compute_fid(feats1, feats2):
    mu1, sigma1 = calculate_activation_statistics(feats1)
    mu2, sigma2 = calculate_activation_statistics(feats2)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def get_text_embedding(text_encoder, word_embs, pos_ohot, cap_lens):
    with torch.no_grad():
        word_embs = word_embs.detach().to("cuda").float()
        pos_ohot = pos_ohot.detach().to("cuda").float()
        cap_lens = cap_lens.detach().to("cuda").float()
        text_embedding = text_encoder(word_embs, pos_ohot, cap_lens)
    return text_embedding



def get_motion_embedding(motions, m_lens):
    with torch.no_grad():
        motions = motions.detach().to("cuda").float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        """Movement Encoding"""
        movements = movement_enc(motions[..., :-4]).detach()
        m_lens = m_lens // opt["unit_length"]
        motion_embedding = motion_enc(movements, m_lens)

    return motion_embedding



# Function to load weights
def load_weights(checkpoint_path, movement_enc, motion_enc, motion_decoder):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        movement_enc.load_state_dict(checkpoint["movement_enc_state_dict"])
        motion_enc.load_state_dict(checkpoint["motion_enc_state_dict"])
        motion_decoder.load_state_dict(checkpoint["motion_decoder_state_dict"])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")


def get_motion_embeds(all_motions):
    batch_size = 64
    total_samples = all_motions.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceil division

    # List to store embeddings
    all_embeddings = []

    # Process in batches
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Get the batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch = all_motions[start_idx:end_idx]  # Shape: (batch_size, 196, 143)

            # Convert to PyTorch tensor if necessary
            if isinstance(batch, np.ndarray):  # If it's a NumPy array
                batch = torch.tensor(batch).float().to(opt["device"])
            else:  # If already a PyTorch tensor
                batch = batch.to(opt["device"])

            # Forward pass through the encoder
            movements = movement_enc(batch)  # Shape: (batch_size, seq_len, latent_dim)
            # movements = movements.permute(0, 2, 1)  # Ensure proper shape for GRU: (batch_size, seq_len, input_dim)

            # Compute sequence lengths
            m_lens = (
                torch.tensor([batch.shape[1]])
                .float()
                .to(opt["device"])
                .repeat(batch.shape[0])
            )
            m_lens = (
                (m_lens // opt["unit_length"]).cpu().long()
            )  # Move to CPU and convert to int64

            # Forward pass through the motion encoder
            embeddings = motion_enc(
                movements, m_lens
            )  # Shape: (batch_size, latent_dim)

            # Append to the list
            all_embeddings.append(embeddings)

    # Concatenate all embeddings into a single tensor
    final_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: (4444, latent_dim)
    return final_embeddings


DEMO_FEAT_PATH = "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/jinkunc/gvhmr/mocap_mixed_v1/unimfm/unimfm_est_st_norm_di_lg_g8/version_0/text_feats_ts10_humanml3d/feats_seed0_263d.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--pred_feat", type=str, 
                    default=DEMO_FEAT_PATH, help="path to the generated features")
    parser.add_argument("--dataset", type=str, default='humanml3d', help='dataset name')

    args = parser.parse_args()

    device = 'cuda'

    if args.dataset == "humanml3d":
        mean = np.load("outputs/humanml3d_mean.npy")
        std = np.load("outputs/humanml3d_std.npy")
        mean = torch.tensor(mean).float().to(device)
        std = torch.tensor(std).float().to(device)
        # the embedding is generated by gen_humanml_text_tokens.py
        text_embeds = pickle.load(open("inputs/humanml_text_embs.pkl", "rb"))
        gt_feat_path = 'outputs/humanml3d_feats_gt/feats_test_humanml3d_format.pt.npy'
        gt_motions = np.load(gt_feat_path)
        gt_motions = torch.tensor(gt_motions).float().to(device)
    elif args.dataset == "motionx":
        mean = np.load("inputs/motion_x_mean_train.npy")
        std = np.load("inputs/motion_x_std_train.npy")
        mean = torch.tensor(mean).float().to(device)
        std = torch.tensor(std).float().to(device)
        text_embeds = torch.load("inputs/motionx_text_clip_embds.pth") # text embeddings

        gt_feat_path = "inputs/motionx_test_gt_feats.pkl"
        gt_features = pickle.load(open(gt_feat_path, "rb"))
        gt_motions = gt_features["motions"]
        gt_texts = gt_features["texts"] # raw text

    # denormalize features 
    gt_motions = gt_motions * std + mean

    # gt_features = pickle.load(args.gt_feat, "rb")
    # gt_motions = gt_features["motions"]
    # gt_texts = gt_features["texts"]

    # feats_path = "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/jinkunc/gvhmr/mocap_mixed_v1/unimfm/unimfm_est_st_norm_di_lg_mx2_cp1_g8/version_0/text_feats_196_motionx"
    # feats_ckpt = "feats_len196_1.pt"
    feats = pickle.load(open(args.pred_feat, "rb"))
    # feats = torch.load(args.pred_feat)
    test_motions = feats["feats"]
    test_texts = feats["text"]

    if args.dataset == 'humanml3d':
        test_motions = test_motions[:, :119]


    POS_enumerator = {
        "VERB": 0,
        "NOUN": 1,
        "DET": 2,
        "ADP": 3,
        "NUM": 4,
        "AUX": 5,
        "PRON": 6,
        "ADJ": 7,
        "ADV": 8,
        "Loc_VIP": 9,
        "Body_VIP": 10,
        "Obj_VIP": 11,
        "Act_VIP": 12,
        "Desc_VIP": 13,
        "OTHER": 14,
    }
    opt = {
        "dataset_name": args.dataset,
        "device": device,
        "dim_word": 300,
        "max_motion_length": 196,
        "dim_pos_ohot": len(POS_enumerator),
        "dim_motion_hidden": 1024,
        "max_text_len": 20,
        "dim_text_hidden": 512 if args.dataset == 'humanml3d' else 256,
        "dim_coemb_hidden": 512,
        "dim_pose": 263 if args.dataset == "humanml3d" else 251,
        "dim_movement_enc_hidden": 512,
        "dim_movement_latent": 512,
        "checkpoints_dir": ".",
        "unit_length": 4,
    }


    text_enc, motion_enc, movement_enc = build_evaluators(args, opt)
    text_enc = text_enc.to(opt['device'])
    motion_enc = motion_enc.to(opt['device'])
    movement_enc = movement_enc.to(opt['device'])


    gt_m_lens = (
        torch.tensor([gt_motions.shape[1]]).float().cuda().repeat(gt_motions.shape[0])
    )
    test_m_lens = (
        torch.tensor([test_motions.shape[1]]).float().cuda().repeat(test_motions.shape[0])
    )

    gt_motion_embs = get_motion_embedding(gt_motions, gt_m_lens)
    test_motion_embs = get_motion_embedding(test_motions, test_m_lens)

    all_word_embs = []
    all_pos_ohots = []
    all_sent_lens = []
    all_text_embeddings = []

    print("Extrating pre-saved text embeddings")
    for i in tqdm(range(len(test_texts))):
        text = test_texts[i]
        saved_embds = text_embeds[text]
        word_embs = saved_embds["word_embeddings"]
        pos_ohots = saved_embds["pose_one_hots"]
        sent_len = saved_embds["sent_len"]
        all_word_embs.append(word_embs[None])
        all_pos_ohots.append(pos_ohots[None])
        all_sent_lens.append(sent_len)

        word_embs = torch.tensor(word_embs).float().cuda()[None]
        pos_ohots = torch.tensor(pos_ohots).float().cuda()[None]
        sent_len = torch.tensor(sent_len).float().cuda()[None]
        text_embedding = get_text_embedding(text_enc, word_embs, pos_ohots, sent_len)
        all_text_embeddings.append(text_embedding)

    all_text_embedding = torch.cat(all_text_embeddings, dim=0)
    
    print("######### Start evaluating the generated features.... ######")

    matching_score, R_precision, all_motion_embeddings = evaluate_matching_score(
        all_text_embedding, test_motion_embs
    )
    fid = compute_fid(gt_motion_embs.cpu().numpy(), test_motion_embs.cpu().numpy())
    div = evaluate_diversity(all_motion_embeddings)

    print("Matching score on motionx is: ", matching_score)
    print("R-precision on motionx is: ", R_precision)
    print("FID on motionx is: ", fid)
    print("Diversity on motionx is: ", div)


