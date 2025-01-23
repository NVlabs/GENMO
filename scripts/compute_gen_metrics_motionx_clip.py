import numpy as np
from scipy import linalg
import torch 
from collections import OrderedDict
import torch.nn as nn
import sys 
import os 
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from motiondiff.utils.mdm_modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from motiondiff.utils.motionx_modules import  MovementConvEncoderWithDropout, MotionEncoderBiGRUCoWithDropout, MotionDecoderWithDropout
import clip
import pickle
from os.path import join as pjoin
from transformers import AutoTokenizer, CLIPTextModelWithProjection


device = 'cuda'
# clip_model = CLIPTextModelWithProjection.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", do_sample=False).to(device).eval()
# clip_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

import clip
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
# encoder = EnDecoder(stats_name='DEFAULT_01', encode_type='humanml3d').cuda()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

dataset_name = 'motionx'
POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}
opt = {
    'dataset_name': dataset_name,
    'device': 'cuda',
    'dim_word': 300,
    'max_motion_length': 196,
    'dim_pos_ohot': len(POS_enumerator),
    'dim_motion_hidden': 1024,
    'max_text_len': 20,
    'dim_text_hidden': 256,
    'dim_coemb_hidden': 512,
    'dim_pose': 143 if dataset_name == 'motionx' else 263,
    'dim_movement_enc_hidden': 512,
    'dim_movement_latent': 512,
    'checkpoints_dir': '.',
    'unit_length': 4,
}


# POS_enumerator = {
#     'VERB': 0,
#     'NOUN': 1,
#     'DET': 2,
#     'ADP': 3,
#     'NUM': 4,
#     'AUX': 5,
#     'PRON': 6,
#     'ADJ': 7,
#     'ADV': 8,
#     'Loc_VIP': 9,
#     'Body_VIP': 10,
#     'Obj_VIP': 11,
#     'Act_VIP': 12,
#     'Desc_VIP': 13,
#     'OTHER': 14,
# }

# Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
#             'up', 'down', 'straight', 'curve')

# Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

# Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

# Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
#             'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
#             'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

# Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
#              'angrily', 'sadly')

# VIP_dict = {
#     'Loc_VIP': Loc_list,
#     'Body_VIP': Body_list,
#     'Obj_VIP': Obj_List,
#     'Act_VIP': Act_list,
#     'Desc_VIP': Desc_list,
# }


class TextEncoderMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super(TextEncoderMLP, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # Define an activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.layer1(x)))
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.bn3(self.layer3(x))  # No activation on the final layer (linear output)
        return x

# mdm version of encoder
def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose'], opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    # text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
    #                               pos_size=opt['dim_pos_ohot'],
    #                               hidden_size=opt['dim_text_hidden'],
    #                               output_size=opt['dim_coemb_hidden'],
    #                               device=opt['device'])
    text_enc = TextEncoderMLP(input_dim=512, hidden_dim=512, output_dim=512)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    # if opt['dataset_name'] == 'motionx':
    #     ckpt_dir = './inputs/t2m'
    ckpt_dir = './t2m_checkpoints/motionx'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match/model_10.0_clip_fullset', 'finest.tar'), map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
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
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
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
    all_motion_embeddings = []
    score_list = []
    all_size = 0
    matching_score_sum = 0
    top_k_count = 0
    # print(motion_loader_name)
    batch_size = 8
    batch_num = len(text_embeddings_all) // batch_size
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            text_embeddings = text_embeddings_all[i*batch_size:(i+1)*batch_size]
            motion_embeddings = motion_embeddings_all[i*batch_size:(i+1)*batch_size]
            dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                    motion_embeddings.cpu().numpy())
            matching_score_sum += dist_mat.trace()

            
            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)
            # print(top_k_mat.sum(axis=0))

            all_size += text_embeddings.shape[0]

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size

    return matching_score, R_precision, all_motion_embeddings


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


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

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


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


gt_features = pickle.load(open('inputs/motionx_test_gt_feats.pkl', 'rb'))
gt_motions = gt_features['motions']
gt_texts = gt_features['texts']


save_parent_dir = '/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/jinkunc/gvhmr/mocap_mixed_v1/unimfm/'
exp_path = 'unimfm_est_st_norm_di_lg_mx3_cp1_g8/version_1/text_feats_196_motionx'
ckpt_name = 'new_feats_len196_0.pt'
feats_path = os.path.join(save_parent_dir, exp_path, ckpt_name)

feats = torch.load(feats_path)
test_motions = feats['feats']
test_texts = feats['text']

gt_motions = torch.tensor(gt_motions).float().to(device)
test_motions = torch.tensor(test_motions).float().to(device)

# test_motions[:, :, 126:136] = gt_motions[:, :, 126:136]

motionx_mean = np.load('inputs/motion_x_mean_train.npy')
motionx_std = np.load('inputs/motion_x_std_train.npy')

motionx_mean = torch.tensor(motionx_mean).float().to(device)
motionx_std = torch.tensor(motionx_std).float().to(device)

gt_motions = (gt_motions - motionx_mean) / motionx_std
test_motions = (test_motions - motionx_mean) / motionx_std


text_enc, motion_enc, movement_enc = build_evaluators(opt)
text_enc = text_enc.to('cuda')
motion_enc = motion_enc.to('cuda')
movement_enc = movement_enc.to('cuda')

total_num = len(gt_texts)



batch_size = 64
batch_num = total_num // batch_size
# batch_num = math.ceil(total_num / batch_size)  # Use ceiling for all items

def get_co_embeds(motions, texts, movement_encoder, motion_encoder, text_enc, opt):
    m_lens = torch.tensor([motions.shape[1]]).float().to(motions.device).repeat(motions.shape[0]).long()
    # Sort the length of motions in descending order, (length of text has been sorted)
    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
    # print(self.align_idx)
    # print(m_lens[self.align_idx])
    motions = motions[align_idx]
    m_lens = m_lens[align_idx]

    '''Movement Encoding'''
    movements = movement_encoder(motions).detach()
    m_lens = m_lens // 4
    motion_embedding = motion_encoder(movements, m_lens // 4)

    '''Text Encoding'''
    with torch.no_grad():
        # texts = [t[0] for t in texts]
        # tokenized_inputs = clip_tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt")
        
        # # Remove 'token_type_ids' (not used by CLIP models)
        # tokenized_inputs.pop("token_type_ids", None)
        # for key in tokenized_inputs:
        #     tokenized_inputs[key] = tokenized_inputs[key].to(device)
        # clip_text_feat = clip_model(**tokenized_inputs)
        # text_embeds_batch = clip_text_feat.text_embeds

        text = clip.tokenize(texts, truncate=True).to(device)
        text_embeds_clip = clip_model.encode_text(text).float()
    
    text_embeds_ours = text_enc(text_embeds_clip)[align_idx]

    return motion_embedding, text_embeds_ours

all_text_embeds = []
all_motion_embeds = []
print("Generating embeddings for GTs...")
for i in tqdm(range(batch_num)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_num)  # Avoid index overflow
    text_batch = gt_texts[start_idx:end_idx]
    text_batch = [t[0] for t in text_batch]
    motion_batch = gt_motions[start_idx:end_idx]
    motion_embeds, text_embeds = get_co_embeds(motion_batch, text_batch, movement_enc, motion_enc, text_enc, opt)
    all_motion_embeds.append(motion_embeds)
    all_text_embeds.append(text_embeds)

all_text_embeds = torch.cat(all_text_embeds, dim=0)
all_motion_embeds = torch.cat(all_motion_embeds, dim=0).detach()

all_text_embeds_test = []
all_motion_embeds_test = []
print("Generating embeddings for test...")
for i in tqdm(range(batch_num)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_num)  # Avoid index overflow
    text_batch = test_texts[start_idx:end_idx]
    text_batch = [t[0] for t in text_batch]
    motion_batch = test_motions[start_idx:end_idx]
    motion_embeds, text_embeds = get_co_embeds(motion_batch, text_batch, movement_enc, motion_enc, text_enc, opt)
    all_motion_embeds_test.append(motion_embeds)
    all_text_embeds_test.append(text_embeds)

all_text_embeds_test = torch.cat(all_text_embeds_test, dim=0)
all_motion_embeds_test = torch.cat(all_motion_embeds_test, dim=0).detach()


matching_score, R_precision, all_motion_embeddings = evaluate_matching_score(all_text_embeds, all_motion_embeds)

fid = compute_fid(all_motion_embeds_test.detach().cpu().numpy(), all_motion_embeds.detach().cpu().numpy())

matching_score_test, R_precision_test, all_motion_embeddings_test = evaluate_matching_score(all_text_embeds_test, all_motion_embeds_test)

div_gt = evaluate_diversity(all_motion_embeds.detach().cpu().numpy())
div_test = evaluate_diversity(all_motion_embeds_test.detach().cpu().numpy())

print("feature path: ", feats_path)
print("FID: ", fid)

print("GT Matching Score: ", matching_score)
print("GT R-precision: ", R_precision)
print("GT Diversity: ", div_gt)

print("Test Matching Score: ", matching_score_test)
print("Test R-precision: ", R_precision_test)
print("Test Diversity: ", div_test)
