import torch
import numpy as np 
from os.path import join as pjoin
import pickle
from tqdm import tqdm 

raw_data_path = 'inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_test.pth'
raw_data = torch.load(raw_data_path)

seqs = list(raw_data.keys())
max_text_len = 20

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

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')


VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
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
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec
    
w_vectorizer = WordVectorizer(pjoin('./', 'glove'), 'our_vab')

text_embs_dict = dict()

for seq in tqdm(seqs):
    print('Processing seq: ', seq)
    data = raw_data[seq]

    # global_orient = data['global_orient_w'].cuda()
    # betas = data['betas'].cuda()
    # body_pose = data['body_pose'].cuda()
    # transl = data['transl_w'].cuda()

    text_data_list = data['text_data']

    for text_data in text_data_list:
        caption = text_data['caption']
        tokens = text_data['tokens']
        real_tokens = []
        for token in tokens:
            token_list = token.split(" ")
            real_tokens += token_list 
        tokens = real_tokens[:-2] # throw away the last two `0.0`
        
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        # breakpoint()
        # token_list = tokens.split(' ')
        for token in tokens:
            token_list = token
            word_emb, pos_oh = w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        text_embs_dict[caption] = {'pose_one_hots': pos_one_hots,
                                    'word_embeddings': word_embeddings,
                                    'sent_len': sent_len}   

with open('inputs/humanml_text_embs.pkl', 'wb') as f: 
    pickle.dump(text_embs_dict, f)
print("done!")
