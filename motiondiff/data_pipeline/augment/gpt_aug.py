import os
import sys
sys.path.append('./')
import argparse
from tqdm import tqdm
from gpt import GPTSession
from motiondiff.data_pipeline.get_data import get_dataset_loader, get_dataset
from motiondiff.utils.tools import write_list_to_file


paraphrase_templates = {
	'part1': [
		'Paraphase this sentence in plain language in three ways:\n"%s"',
		'Paraphase this sentence under 5 words in plain language in three ways:\n"%s"'
	],
	'part2': [
		# 'Paraphase this sentence in plain language in three ways:\n"%s"',
		# 'Paraphase this sentence under 5 words in plain language in three ways:\n"%s"',
		'Paraphase this sentence with 5 to 10 words in plain language in three ways:\n"%s"',
		'Paraphase this sentence with 10 to 15 words in plain language in three ways:\n"%s"',
		'Paraphase this sentence with 15 to 20 words in plain language in three ways:\n"%s"',
		'Paraphase this sentence with 20 to 25 words in plain language in three ways:\n"%s"',
		'Paraphase this sentence to add some detail (not too much) in plain language in three ways:\n"%s"',
		'Paraphase this sentence to make it more succinct in plain language in three ways:\n"%s"',
		'Paraphase this sentence to make it as succinct as possible in plain language in three ways:\n"%s"'
	]
}


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train')
parser.add_argument('--out_dir', default='dataset/HumanML3D_gpt_aug')
parser.add_argument('--template_key', default='part1')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--start_i', type=int, default=0)
parser.add_argument('--end_i', type=int, default=1e8)
args = parser.parse_args()

session = GPTSession()
out_dir = f'{args.out_dir}/{args.template_key}'
os.makedirs(out_dir, exist_ok=True)

# text = """a person moves backwards then forwards then jumps."""

def paraphrase_text(text):
	try:
		all_para_texts = []
		templates_arr = sum([paraphrase_templates[x] for x in args.template_key.split(',')], [])
		for template in templates_arr:
			prompt = template % text
			reply = session.chat(prompt)
			reply = reply.replace('"', '').replace("'", '')
			para_texts = []
			for x in reply.split('\n'):
				if len(x) <= 5:
					continue
				if '-' in x[:4]:
					para_texts.append(x.split('-', 1)[1].strip())
				elif '*' in x[:4]:
					para_texts.append(x.split('*', 1)[1].strip())
				else:
					para_texts.append(x.split('.', 1)[1].strip())
			para_texts = [x for x in para_texts if len(x) > 1]
			if len(reply.split('\n')) != len(para_texts):
				print('Warning: reply and para_texts have different lengths!')
				print('Reply:', reply)
			# para_texts_str = '\n'.join(para_texts)
			# print(f'* Prompt:\n{prompt}\n* Reply:\n{para_texts_str}\n')
			all_para_texts += para_texts
		# print('num of para texts:', len(all_para_texts))
		# print(all_para_texts)
	except KeyboardInterrupt:
		sys.exit()
	except:
		return None
	return all_para_texts


# text = """a person moves backwards then forwards then jumps."""
# paraphrase_text(text)
# exit()


dataset = get_dataset(name='humanml', num_frames=196, split=args.split, hml_mode='train', debug=args.debug)
print(f'{args.split} dataset size:', len(dataset))

for i, (seq, data) in enumerate(tqdm(dataset.t2m_dataset.data_dict.items())):
	if not (args.start_i <= i < args.end_i):
		continue
	
	for j, text_dict in enumerate(data['text']):
		out_path = f'{out_dir}/{seq}-{j}.txt'
		if args.skip and os.path.exists(out_path):
			continue
		text = text_dict['caption']
		print(f'Processing {i}/{len(dataset)} [{seq}.txt], text: {text}')
		for _ in range(5):
			para_texts = paraphrase_text(text)
			if para_texts is not None:
				break
		if para_texts is None:
			print('Warning: paraphrase_text failed!')
			continue
		text_list = [text] + para_texts
		write_list_to_file(out_path, text_list)





