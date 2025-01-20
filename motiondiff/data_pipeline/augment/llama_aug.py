from openai import OpenAI
import os
import sys
sys.path.append('./')
import argparse
import pandas as pd
import time
from motiondiff.utils.tools import write_list_to_file


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

def call_llama(prompt):
    prefix = "When answering the following questions, please always give the answers directly and start each answer with an asterisk (*). Each answer is one line. Don't use any asterisk except for the start.\n"
    completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[{"role":"user","content":prefix + prompt}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
    )

    results = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            results += chunk.choices[0].delta.content
    return results


paraphrase_templates = {
    'part1': [
        'Paraphase this sentence in plain language in three ways:\n"%s"',
        'Paraphase this sentence as a command in three ways:\n"%s"',
        'Paraphase this sentence as short as possible in plain language in three ways:\n"%s"',
        'Paraphase this sentence under 5 words in plain language in three ways:\n"%s"',
    ],
}


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=3)
parser.add_argument('--template_key', default='part1')
args = parser.parse_args()

out_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_aug_texts/v1'
meta_file = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.1/meta_240416_v3.csv'
# meta_file = 'out/meta1.csv'
meta = pd.read_csv(meta_file)

os.makedirs(out_dir, exist_ok=True)


def paraphrase_text(text):
    try:
        all_para_texts = []
        templates_arr = sum([paraphrase_templates[x] for x in args.template_key.split(',')], [])
        for template in templates_arr:
            prompt = template % text
            reply = call_llama(prompt)
            reply = reply.replace('"', '').replace("'", '').replace('!', '.')
            para_texts = []
            for x in reply.split('\n'):
                if len(x) <= 4 or x[0] != '*':
                    continue
                if '*' in x[:4]:
                    text_p = x.split('*', 1)[1].strip()
                    if '*' in text_p:
                        raise Exception('Additional asterisk found!')
                    para_texts.append(text_p)
            para_texts = [x for x in para_texts if len(x) > 1]
            all_para_texts += para_texts
        if len(all_para_texts) > len(templates_arr) * 3:
            raise Exception('Too many paraphrases. Something is possibly wrong!')
        # print('num of para texts:', len(all_para_texts))
        # print(all_para_texts)
    except KeyboardInterrupt:
        sys.exit()
    except:
        return None
    return all_para_texts


# text = """a person moves backwards then forwards then jumps."""
# all_para_texts = paraphrase_text(text)
# exit()


for i in range(args.start, args.end):
    t0 = time.time()
    
    row = meta.iloc[i]
    texts = [
        str(row['natural_desc_1']),
        str(row['natural_desc_2']),
        str(row['natural_desc_3']),
        str(row['technical_description']),
        str(row['short_description'])
    ]
    
    for k, text in enumerate(texts):
        if len(text) <= 2 or text == 'nan':
            continue
        out_path = f'{out_dir}/{i:06d}-{k}.txt'
        if os.path.exists(out_path):
            print(f'[Exists] {out_path}')
            continue
        for _ in range(5):
            para_texts = paraphrase_text(text)
            if para_texts is not None:
                break
        if para_texts is None:
            print('Warning: paraphrase_text failed!')
            continue
        
        try:
            text_list = [text] + para_texts
            write_list_to_file(out_path, text_list)
        except:
            print('Warning: write_list_to_file failed!')
            print(text_list)
            continue
    
    
    print(f'{i}/({args.start}, {args.end}) processed, time: {time.time() - t0:.2f}')
    
    