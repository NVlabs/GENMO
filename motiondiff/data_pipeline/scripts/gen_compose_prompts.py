import numpy as np
import argparse
import pickle
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--out_file_path', default='assets/compose_prompts/230715_%s.txt')
parser.add_argument('--max_num', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--filter_max_num', type=int, default=2)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

text_mapping = pickle.load(open('data_aug/HumanML3D_gpt_aug/part1.pkl', 'rb'))
orig_texts = list(text_mapping.keys())

def num_in_dataset(keywords):
    num = 0
    for text in orig_texts:
        all_keywords = sum(keywords, [])
        all_exist = True
        for kw in all_keywords:
            if kw not in text:
                all_exist = False
                break
        if all_exist:
            num += 1
    return num


lowerbody_action_lists = [
    ('walks', 'walking', ['walk']),   # 3rd person, present participle, keywords
    ('runs', 'running', ['run']),
    ('jogs', 'jogging', ['jog']),
    ('jumps', 'jumping', ['jump']),
    ('crouches', 'crouching', ['crouch']),
    ('stands', 'standing', ['stand']),
    ('sits', 'sitting', ['sit']),
    ('turns left', 'turning left', ['turn', 'left']),
    ('turns right', 'turning right', ['turn', 'right'])
]

upperbody_action_lists = [
    ('waves', 'waving', ['wave']),
    ('points forward', 'pointing forward', ['point', 'forward']),
    ('lifts left hand', 'lifting left hand', ['lift', 'left hand']),
    ('lifts right hand', 'lifting right hand', ['lift', 'right hand']),
    ('lifts both hands', 'lifting both hands', ['lift', 'both hands']),
    ('punches', 'punching', ['punch']),
    ('throws', 'throwing', ['throw']),
    ('catches', 'catching', ['catch']),
    ('claps', 'clapping', ['clap']),
    ('scratches head', 'scratching head', ['scratch', 'head']),
    ('picks up a cell phone', 'picking up a cell phone', ['pick up', 'cell phone']),
    ('carrys a bag', 'carrying a bag', ['carry', 'bag']),
]


atom_sentence_templates = {
    'lower': ['<lower_body>'],
    'upper': ['<upper_body>'],
    'lower_while_upper': ['<lower_body>', 'while', '<upper_body_present>']
}

temporal_compose_templates = {
    'lower_upper': ['a person', '<lower_while_upper>'],
    'lower_upper+lower_upper': ['a person', '<lower_while_upper>', ',', 'and then', '<lower_while_upper>'],
    'lower+lower_upper': ['a person', '<lower>', ',', 'and then', '<lower_while_upper>'],
    'lower+lower': ['a person', '<lower>', ',', 'and then', '<lower>'],
    'lower+upper': ['a person', '<lower>', ',', 'and then', '<upper>'],
    'upper+upper': ['a person', '<upper>', ',', 'and then', '<upper>'],
}

def compose_atom_sentence(all_gen_prompts, cur_prompt, template, keywords, all_keywords_list):
    if len(template) == 0:
        all_gen_prompts.append(cur_prompt)
        all_keywords_list.append(keywords)
        return 
    
    cur_subsentence = template[0]
    if cur_subsentence in ['<lower_body>', '<lower_body_present>']:
        for (lower_body_verb, lower_body_verb_present, lower_body_keywords) in lowerbody_action_lists:
            if cur_subsentence == '<lower_body>':
                new_prompt = (cur_prompt + ' ' + lower_body_verb).strip()
            else:
                new_prompt = (cur_prompt + ' ' + lower_body_verb_present).strip()
            new_keywords = keywords.copy()
            new_keywords.append(lower_body_keywords)
            compose_atom_sentence(all_gen_prompts, new_prompt, template[1:], new_keywords, all_keywords_list)
    elif cur_subsentence in ['<upper_body>', '<upper_body_present>']:
        for (upper_body_verb, upper_body_verb_present, upper_body_keywords) in upperbody_action_lists:
            if cur_subsentence == '<upper_body>':
                new_prompt = (cur_prompt + ' ' + upper_body_verb).strip()
            else:
                new_prompt = (cur_prompt + ' ' + upper_body_verb_present).strip()
            new_keywords = keywords.copy()
            new_keywords.append(upper_body_keywords)
            compose_atom_sentence(all_gen_prompts, new_prompt, template[1:], new_keywords, all_keywords_list)
    elif cur_subsentence in [',', '.']:
        compose_atom_sentence(all_gen_prompts, (cur_prompt + cur_subsentence).strip(), template[1:], keywords, all_keywords_list)
    else:
        compose_atom_sentence(all_gen_prompts, (cur_prompt + ' ' + cur_subsentence).strip(), template[1:], keywords, all_keywords_list)

    return


def compose_temporal_sentence(all_gen_prompts, cur_prompt, template, atom_dict, used_sentences, random_n=0):
    if len(template) == 0:
        all_gen_prompts.append(cur_prompt)
        return 
    
    cur_subsentence = template[0]
    if cur_subsentence[0] == '<' and cur_subsentence[-1] == '>':
        key = cur_subsentence[1:-1]
        if random_n > 0:
            index = np.random.permutation(len(atom_dict[key]))[:min(random_n, len(atom_dict[key]))]
        else:
            index = np.arange(len(atom_dict[key]))
        for i in index:
            atom_sentence = atom_dict[key][i]
            if atom_sentence in used_sentences:
                continue
            used_sentences.add(atom_sentence)
            new_prompt = (cur_prompt + ' ' + atom_sentence).strip()
            compose_temporal_sentence(all_gen_prompts, new_prompt, template[1:], atom_dict, used_sentences, random_n)
            used_sentences.remove(atom_sentence)
    elif cur_subsentence in [',', '.']:
        compose_temporal_sentence(all_gen_prompts, (cur_prompt + cur_subsentence).strip(), template[1:], atom_dict, used_sentences, random_n)
    else:
        compose_temporal_sentence(all_gen_prompts, (cur_prompt + ' ' + cur_subsentence).strip(), template[1:], atom_dict, used_sentences, random_n)

    return


# generate atom sentences using sptial composition
atom_sentences_dict = {}
atom_sentences_filter_dict = {}
atom_keywords_dict = {}
atom_sentences_num_in_dataset = {}
for key, template in atom_sentence_templates.items():
    atom_sentences_dict[key] = []
    atom_sentences_filter_dict[key] = []
    atom_keywords_dict[key] = []
    compose_atom_sentence(atom_sentences_dict[key], '', template, [], atom_keywords_dict[key])
    assert len(atom_sentences_dict[key]) == len(atom_keywords_dict[key])
           
    for prompt, keywords in zip(atom_sentences_dict[key], atom_keywords_dict[key]):
        atom_sentences_num_in_dataset[prompt] = num_in_dataset(keywords)
        if key in ['lower', 'upper']:
            atom_sentences_filter_dict[key].append(prompt)
        elif atom_sentences_num_in_dataset[prompt] <= args.filter_max_num:
            atom_sentences_filter_dict[key].append(prompt)
            print(prompt, keywords, atom_sentences_num_in_dataset[prompt])
    print(f'[Atom] "{key}" total prompts: {len(atom_sentences_dict[key])}, filtered total prompts: {len(atom_sentences_filter_dict[key])}')




# generate temporal sentences using temporal composition
temporal_sentences_dict = {}
for key, template in temporal_compose_templates.items():
    temporal_sentences_dict[key] = []
    compose_temporal_sentence(temporal_sentences_dict[key], '', template, atom_sentences_filter_dict, set(), random_n=0)

    for prompt in temporal_sentences_dict[key]:
        print(prompt)
    print(f'[Temporal] "{key}" total prompts: {len(temporal_sentences_dict[key])}')

    out_prompts = temporal_sentences_dict[key].copy()
    fkey = key
    if len(out_prompts) > args.max_num:
        out_prompts = np.random.permutation(out_prompts)[:args.max_num]
        fkey += '_seed%d' % args.seed

    # below code to dump a list to a file
    fname = args.out_file_path % fkey
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        for prompt in out_prompts:
            f.write(prompt + '\n')


