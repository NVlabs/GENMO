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
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from motiondiff.utils.mdm_modules import (
    MotionEncoderBiGRUCo,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)

# encoder = EnDecoder(stats_name="DEFAULT_01", encode_type="humanml3d").cuda()

dataset_name = "humanml"
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
    "dataset_name": dataset_name,
    "device": "cuda",
    "dim_word": 300,
    "max_motion_length": 196,
    "dim_pos_ohot": len(POS_enumerator),
    "dim_motion_hidden": 1024,
    "max_text_len": 20,
    "dim_text_hidden": 512,
    "dim_coemb_hidden": 512,
    "dim_pose": 263 if dataset_name == "humanml" else 251,
    "dim_movement_enc_hidden": 512,
    "dim_movement_latent": 512,
    "checkpoints_dir": ".",
    "unit_length": 4,
}


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

Loc_list = (
    "left",
    "right",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "forward",
    "back",
    "backward",
    "up",
    "down",
    "straight",
    "curve",
)

Body_list = (
    "arm",
    "chin",
    "foot",
    "feet",
    "face",
    "hand",
    "mouth",
    "leg",
    "waist",
    "eye",
    "knee",
    "shoulder",
    "thigh",
)

Obj_List = (
    "stair",
    "dumbbell",
    "chair",
    "window",
    "floor",
    "car",
    "ball",
    "handrail",
    "baseball",
    "basketball",
)

Act_list = (
    "walk",
    "run",
    "swing",
    "pick",
    "bring",
    "kick",
    "put",
    "squat",
    "throw",
    "hop",
    "dance",
    "jump",
    "turn",
    "stumble",
    "dance",
    "stop",
    "sit",
    "lift",
    "lower",
    "raise",
    "wash",
    "stand",
    "kneel",
    "stroll",
    "rub",
    "bend",
    "balance",
    "flap",
    "jog",
    "shuffle",
    "lean",
    "rotate",
    "spin",
    "spread",
    "climb",
)

Desc_list = (
    "slowly",
    "carefully",
    "fast",
    "careful",
    "slow",
    "quickly",
    "happy",
    "angry",
    "sad",
    "happily",
    "angrily",
    "sadly",
)

VIP_dict = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_List,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec


# mdm version of encoder
def build_evaluators(opt):
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

    ckpt_dir = opt["dataset_name"]
    if opt["dataset_name"] == "humanml":
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

            # word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
            # text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
            #     word_embs=word_embeddings,
            #     pos_ohot=pos_one_hots,
            #     cap_lens=sent_lens,
            #     motions=motions,
            #     m_lens=m_lens
            # )
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
    #     match_score_dict[motion_loader_name] = matching_score
    #     R_precision_dict[motion_loader_name] = R_precision
    #     activation_dict[motion_loader_name] = all_motion_embeddings

    # print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
    # print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

    # line = f'---> [{motion_loader_name}] R_precision: '
    # for i in range(len(R_precision)):
    #     line += '(top %d): %.4f ' % (i+1, R_precision[i])
    # print(line)
    # print(line, file=file, flush=True)

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
    -- mu: dim_featcccccbkvecgrfrjgrctkkhtnnjcbctdvhktfgbergbnh
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
    # eval_dict = OrderedDict({})
    # print('========== Evaluating Diversity ==========')
    # for model_name, motion_embeddings in activation_dict.items():
    #     diversity = calculate_diversity(motion_embeddings, diversity_times)
    #     eval_dict[model_name] = diversity
    #     print(f'---> [{model_name}] Diversity: {diversity:.4f}')
    #     print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
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
    # eval_dict = OrderedDict({})
    # print('========== Evaluating MultiModality ==========')
    # for model_name, mm_motion_loader in mm_motion_loaders.items():
    #     mm_motion_embeddings = []
    #     with torch.no_grad():
    #         for idx, batch in enumerate(mm_motion_loader):
    #             # (1, mm_replications, dim_pos)
    #             motions, m_lens = batch
    #             motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
    #             mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
    #     if len(mm_motion_embeddings) == 0:
    #         multimodality = 0
    #     else:
    #         mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
    #         multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
    #     print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
    #     print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
    #     eval_dict[model_name] = multimodality
    mm_motion_embeddings = feats
    score = calculate_multimodality(mm_motion_embeddings, mm_num_times)
    return score


def compute_fid(feats1, feats2):
    mu1, sigma1 = calculate_activation_statistics(feats1)
    mu2, sigma2 = calculate_activation_statistics(feats2)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


t2m_path = "inputs/t2m"
kit_path = "inputs/kit"

feats_path = "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/jinkunc/gvhmr/mocap_mixed_v1/unimfm/unimfm_est_st_norm_di_lg_g8/version_0/text_feats_ts10_humanml3d"
feats_file = os.path.join(feats_path, "feats_seed{}_263d.pkl".format(0))
feats = pickle.load(open(feats_file, "rb"))
motions = feats["feats"]
texts = feats["text"]
# breakpoint()

# feats_1_path = os.path.join(feats_path, 'feats_part0_len196_0.pt')
# feats_2_path = os.path.join(feats_path, 'feats_part1_len196_0.pt')
# feats1 = torch.load(feats_1_path)
# # texts = feats1['text']
# motions_1 = feats1['feats']
# feats2 = torch.load(feats_2_path)
# motions_2 = feats2['feats']

# motions = torch.cat([motions_1, motions_2], dim=0)
# motions_padded = torch.zeros(motions.shape[0], motions.shape[1], 263)
# motions_padded[:, :, :motions.shape[2]] = motions
# model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# # Choice #1: we use CLIP to get text embeddings from raw texts
# bs = 128
# total_text_num = len(texts)
# batch_num = total_text_num // bs
# all_text_embeds = []
# print("Generating text embbedings...")
# for i in tqdm(range(batch_num+1)):
#     # texts = clip.tokenize(raw_text, context_length=196, truncate=True).to('cuda')
#     # breakpoint()
#     text_batch = texts[i*bs:(i+1)*bs]
#     inputs = tokenizer(text_batch, padding='max_length', truncation=True,max_length=77, return_tensors="pt")
#     outputs = model(**inputs)
#     text_embeds = outputs.text_embeds
#     all_text_embeds.append(text_embeds)

# text_embeds = torch.cat(all_text_embeds, dim=0)
# torch.save(text_embeds, 'inputs/humanml3d_part0_text_clip_embds.pth')

# motions_smpl_format = encoder.decode_humanml3d(motions)

# breakpoint()

humanml3d_text_embeds = torch.load("inputs/humanml3d_text_clip_embds.pth")

text_encoder, motion_encoder, movement_encoder = build_evaluators(opt)
# motion encoders used by MDM for evaluation, we follow this convention.
motion_encoder = motion_encoder.to("cuda")
movement_encoder = movement_encoder.to("cuda")
# Choice #2: the vectorizer and text embedding model used by MDM for evaluation, tokens required
w_vectorizer = WordVectorizer(pjoin("./", "glove"), "our_vab")
text_encoder = text_encoder.to("cuda")


def encode_tokens(tokens, w_vectorizer):
    if len(tokens) < opt.max_text_len:
        # pad with "unk"
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (opt.max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[: opt.max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    return word_embeddings, pos_one_hots, sent_len


def get_text_embedding(word_embs, pos_ohot, cap_lens):
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
        movements = movement_encoder(motions[..., :-4]).detach()
        m_lens = m_lens // opt["unit_length"]
        motion_embedding = motion_encoder(movements, m_lens)

    return motion_embedding


# smpl_data = encoder.decode_humanml3d(raw_data)
m_lens = torch.tensor([motions.shape[1]]).float().cuda().repeat(motions.shape[0])

mean = np.load("outputs/humanml3d_mean.npy")
std = np.load("outputs/humanml3d_std.npy")

mean = torch.tensor(mean).cuda()
std = torch.tensor(std).cuda()
# motions = (motions - torch.tensor(mean).cuda()) / torch.tensor(std).cuda()
# motions = motions * std + mean

motion_embs = get_motion_embedding(motions, m_lens)

humanml_gt_motions_path = "outputs/humanml3d_feats_gt/feats_test.pt"
# Pred_vectors_path = 'outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats.pt'

# texts = 'outputs/humanml3d_test_texts.pt'
# GT_vectors_path = 'outputs/humanml3d_feats_gt/feats_test_humanml3d_format.pt.npy'
# Pred_vectors_path = 'outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats_humanml3d_format.npy'
# GT_vectors_path = 'outputs/ground_truth_motions.npy'
# Pred_vectors_path = 'outputs/mdm_vald_motions.npy'


# gt_motions = torch.load(humanml_gt_motions_path).cuda()
# gt_m_lens = torch.tensor([gt_motions.shape[1]]).float().cuda().repeat(gt_motions.shape[0])
# gt_motion_embs = get_motion_embedding(gt_motions, gt_m_lens)

# matching_score, R_precision, all_motion_embeddings = evaluate_matching_score(humanml3d_text_embeds, motion_embs)

humanml_text_embs = pickle.load(open("inputs/humanml_text_embs.pkl", "rb"))

batch_size = 64
batch_num = len(texts) // batch_size

all_word_embs = []
all_pos_ohots = []
all_sent_lens = []
all_text_embeddings = []

print("Extrating pre-saved text embeddings")
for i in tqdm(range(len(texts))):
    text = texts[i]
    saved_embds = humanml_text_embs[text]
    word_embs = saved_embds["word_embeddings"]
    pos_ohots = saved_embds["pose_one_hots"]
    sent_len = saved_embds["sent_len"]
    all_word_embs.append(word_embs[None])
    all_pos_ohots.append(pos_ohots[None])
    all_sent_lens.append(sent_len)

    word_embs = torch.tensor(word_embs).float().cuda()[None]
    pos_ohots = torch.tensor(pos_ohots).float().cuda()[None]
    sent_len = torch.tensor(sent_len).float().cuda()[None]
    text_embedding = get_text_embedding(word_embs, pos_ohots, sent_len)
    all_text_embeddings.append(text_embedding)

# all_word_embs = torch.tensor(np.concatenate(all_word_embs, axis=0))
# all_pos_ohots = torch.tensor(np.concatenate(all_pos_ohots, axis=0))
# all_sent_lens = torch.tensor(np.array(all_sent_lens))

# text_embedding = get_text_embedding(all_word_embs, all_pos_ohots, all_sent_lens)
all_text_embedding = torch.cat(all_text_embeddings, dim=0)

matching_score, R_precision, all_motion_embeddings = evaluate_matching_score(
    all_text_embedding, motion_embs
)

# GT_feats = (GT_feats - mean) / std
# Pred_feats = (Pred_feats - mean) / std


GT_vectors_path = "outputs/humanml3d_feats_gt/feats_test_humanml3d_format.pt.npy"
GT_feats = np.load(GT_vectors_path)
gt_m_lens = torch.tensor([GT_feats.shape[1]]).float().cuda().repeat(GT_feats.shape[0])
# pred_m_lens = torch.tensor([Pred_feats.shape[1]]).float().cuda().repeat(Pred_feats.shape[0])
GT_feats = torch.tensor(GT_feats).float().cuda()
# Pred_feats = torch.tensor(Pred_feats).float().cuda()
# raw_texts = torch.load(texts)
gt_motion_embs = get_motion_embedding(GT_feats, gt_m_lens)
# pred_motion_embs = get_motion_embedding(Pred_feats, pred_m_lens)
# fid = compute_fid(gt_motion_embs.cpu().numpy(), all_motion_embeddings)
breakpoint()
# GT_feats = GT_feats * std + mean
# Pred_feats = Pred_feats * std + mean

multimodal_dist = matching_score

fid = compute_fid(gt_motion_embs.cpu().numpy(), all_motion_embeddings)

div = evaluate_diversity(all_motion_embeddings)

print("Multimodal Distance: ", multimodal_dist)
print("FID: ", fid)
print("Diversity: ", div)

# GT_diversity = evaluate_diversity(gt_motion_embs.cpu().numpy())
# Pred_diversity = evaluate_diversity(pred_motion_embs.cpu().numpy())

# GT_multi_modality = evaluate_multimodality(gt_motion_embs.cpu().numpy())
# Pred_multi_modality = evaluate_multimodality(pred_motion_embs.cpu().numpy())

# GT_multi_modality = evaluate_multimodality(GT_feats)
# Pred_multi_modality = evaluate_multimodality(Pred_feats)


# print(f'GT_multi_modality: {GT_multi_modality}')
# print(f'Pred_multi_modality: {Pred_multi_modality}')

# feat_dim = GT_feats.shape[-1]
# GT_feats_flat = GT_feats.reshape(-1, feat_dim)
# Pred_feats_flat = Pred_feats.reshape(-1, feat_dim)
# fid = compute_fid(GT_feats_flat, Pred_feats_flat)
# print(f'FID: {fid}')

# GT_diversity = evaluate_diversity(GT_feats_flat)
# Pred_diversity = evaluate_diversity(Pred_feats_flat)
# print(f'GT_diversity: {GT_diversity}')
# print(f'Pred_diversity: {Pred_diversity}')


# FID:
# mdm: 1.7330507772884864
# ours: 0.13503613421584149

# texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
