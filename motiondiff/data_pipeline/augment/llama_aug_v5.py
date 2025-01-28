import os
import sys

from openai import OpenAI

sys.path.append("./")
import argparse
import time

import pandas as pd

from motiondiff.utils.tools import write_list_to_file

NUM_EXPECTED = 10  # ten replies per prompt

prefix = "Answer the following questions DIRECTLY without any introduction, and start each answer with an asterisk (*). \
Each answer is one line. DO NOT use the asterisk (*) except at the start of an answer.\n"

preamble = "I will give you four descriptions that all describe the exact same human motion. \
Think about how the person is moving then write exactly TEN (10) more descriptions of the motion. "
# Make sure to keep the meaning the same as the original descriptions, but leave out small details:\n"%s"',

postfix = "Here are the descriptions for you. Write the descriptions DIRECTLY without any introduction and write each description on a separate line starting with an asterisk:\n%s"

post_details = "make sure the new descriptions accurately describe the motion as much as possible. "
post_brief = " leave out small details like starting/ending foot, left/rights, exact angles, etc... "

# rephrase with roughly the same length
part_paraphrase = (
    "Please use plain language and "
    + post_details
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* person reaching down at a 45 degree angle towards an unattainable object on the floor with their left hand\n\
* trying to grasp something located far below on the floor with the left hand\n\
* someone extends their left hand as if trying to reach something far away down on the floor from a 45 degree angle\n\
* reaching down\n\
Output:\n\
* Reaching toward the ground with the left hand for an object that's out of reach.\n\
* Stretches out the left arm to grab something off the floor. \n\
* The person extends their left hand downwards to try and retrieve an object from the ground.\n\
Input:\n\
* reacting strongly to a kidney punch from the left side while in a fighting stance\n\
* exhibit a heavy reaction as if hit by a kidney punch from the left side\n\
* character in fighting stance reacts heavily to a kidney punch from the left side\n\
* reacting strongly to a punch to the kidney\n\
Output:\n\
* A person reacts to getting punched on the left side.\n\
* Doubling over after taking a hit from the left. \n\
* Starting in a fighting stance, the character receives a blow on the left side.\n\
"
)
# as a command to the character
part_command = (
    "Write the new descriptions as commands to the person and "
    + post_details
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* reacting strongly to a kidney punch from the left side while in a fighting stance\n\
* exhibit a heavy reaction as if hit by a kidney punch from the left side\n\
* character in fighting stance reacts heavily to a kidney punch from the left side\n\
* reacting strongly to a punch to the kidney\n\
Output:\n\
* React to getting punched on the left side.\n\
* Double over after taking a hit from the left. \n\
* Starting in a fighting stance, receive a blow on the left side.\n\
Input:\n\
* an old person making a small, maximum effort jump forward\n\
* very old person making a slow small forward jump\n\
* an old and tired performer makes a small forward jump with maximum effort\n\
* old person slowly jump forward with maximum effort\n\
Output:\n\
* Take a small jump forward like an old person.\n\
* Acting like an old person, leap forward. \n\
* Make a slow and tired jump forward.\n\
"
)
# as short as possible with same detail
part_brief_detail = (
    "Be as short and simple as possible but "
    + post_details
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* person reaching down at a 45 degree angle towards an unattainable object on the floor with their left hand\n\
* trying to grasp something located far below on the floor with the left hand\n\
* someone extends their left hand as if trying to reach something far away down on the floor from a 45 degree angle\n\
* reaching down\n\
Output:\n\
* Reaching to the ground with the left hand.\n\
* Grabbing something off the floor with left hand. \n\
* The person reaches with their left for an object on the ground.\n\
Input:\n\
* reacting strongly to a kidney punch from the left side while in a fighting stance\n\
* exhibit a heavy reaction as if hit by a kidney punch from the left side\n\
* character in fighting stance reacts heavily to a kidney punch from the left side\n\
* reacting strongly to a punch to the kidney\n\
Output:\n\
* A person reacts to a punch.\n\
* Taking a hit from the left. \n\
* The character receives a blow in a fighting stance.\n\
"
)
# as short as possible as commands
part_brief_command = (
    "Write the descriptions as commands and be as short and simple as possible, "
    + post_details
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* person reaching down at a 45 degree angle towards an unattainable object on the floor with their left hand\n\
* trying to grasp something located far below on the floor with the left hand\n\
* someone extends their left hand as if trying to reach something far away down on the floor from a 45 degree angle\n\
* reaching down\n\
Output:\n\
* Reach to the ground.\n\
* Grab something from the floor. \n\
* Reach for an object on the ground.\n\
Input:\n\
* an old person making a small, maximum effort jump forward\n\
* very old person making a slow small forward jump\n\
* an old and tired performer makes a small forward jump with maximum effort\n\
* old person slowly jump forward with maximum effort\n\
Output:\n\
* Take a small jump forward like an old person.\n\
* Act like an old person and leap forward. \n\
* Jump forward slowly.\n\
"
)
# very short, lose detail
part_short = (
    "Be as short and simple as possible describing the MAIN action of the motion, "
    + post_brief
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* person reaching down at a 45 degree angle towards an unattainable object on the floor with their left hand\n\
* trying to grasp something located far below on the floor with the left hand\n\
* someone extends their left hand as if trying to reach something far away down on the floor from a 45 degree angle\n\
* reaching down\n\
Output:\n\
* Reaching down.\n\
* Grab from the floor.\n\
* Reach to the ground.\n\
Input:\n\
* reacting strongly to a kidney punch from the left side while in a fighting stance\n\
* exhibit a heavy reaction as if hit by a kidney punch from the left side\n\
* character in fighting stance reacts heavily to a kidney punch from the left side\n\
* reacting strongly to a punch to the kidney\n\
Output:\n\
* Reacting to a punch.\n\
* Taking a hit from the left.\n\
* Receives a punch.\n\
"
)
# very short as a command
part_short_command = (
    "Describe the MAIN action of the motion as a command and be as short and simple as possible, "
    + post_brief
    + "Here are few examples with only 3 outputs:\n \
Input:\n\
* person reaching down at a 45 degree angle towards an unattainable object on the floor with their left hand\n\
* trying to grasp something located far below on the floor with the left hand\n\
* someone extends their left hand as if trying to reach something far away down on the floor from a 45 degree angle\n\
* reaching down\n\
Output:\n\
* Reach to the ground.\n\
* Grab from the floor. \n\
* Reach for an object.\n\
Input:\n\
* an old person making a small, maximum effort jump forward\n\
* very old person making a slow small forward jump\n\
* an old and tired performer makes a small forward jump with maximum effort\n\
* old person slowly jump forward with maximum effort\n\
Output:\n\
* Small jump forward.\n\
* Leap forward slowly. \n\
* Tired jump forward.\n\
"
)

# index into part1
ITER_TEMPLATES = {4, 5}  # short descriptions

paraphrase_templates = {
    "part1": [
        preamble + part_paraphrase + postfix,
        preamble + part_command + postfix,
        preamble + part_brief_detail + postfix,
        preamble + part_brief_command + postfix,
        preamble + part_short + postfix,
        preamble + part_short_command + postfix,
    ],
    "old2": [
        'I will give you several descriptions of a motion. Write exactly THREE (3) more descriptions of the motion in plain language. Make sure to keep the meaning the same as the original descriptions, but leave out small details:\n"%s"',
        'I will give you several descriptions of a motion. Write exactly THREE (3) more descriptions of the motion in the form of a command. Make sure to keep the meaning the same as the original descriptions, but leave out small details:\n"%s"',
        'I will give you several descriptions of a motion. Write exactly THREE (3) more descriptions of the motion being as brief and succinct as possible. Keep the overall meaning the same as the original descriptions, but leave out small details:\n"%s"',
        'I will give you several descriptions of a motion. Write exactly THREE (3) more descriptions of the motion, each using less than FIVE (5) words. Keep the overall meaning the same as the original descriptions, but leave out the details:\n"%s"',
    ],
    "old": [
        'Paraphase this sentence in plain language in three ways:\n"%s"',
        'Paraphase this sentence as a command in three ways:\n"%s"',
        'Paraphase this sentence as short as possible in plain language in three ways:\n"%s"',
        'Paraphase this sentence under 5 words in plain language in three ways:\n"%s"',
    ],
}

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    #   api_key = "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", # ye
    api_key="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",  # davis
)


def call_llama(prompt, iterate=False):
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True,
    )

    results = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            results += chunk.choices[0].delta.content

    # print("RESULTS 1")
    # print(results)

    if iterate:
        iterate_prompt = "Do these descriptions really have the same meaning as the original motion? Please try again and make sure to keep the overall meaning the same. Write the descriptions DIRECTLY without any introduction and write each description on a separate line starting with an asterisk:"
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": results},
                {"role": "user", "content": iterate_prompt},
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True,
        )

        results = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                results += chunk.choices[0].delta.content

        # print("RESULTS 2")
        # print(results)

    return results


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=3)
parser.add_argument("--template_key", default="part1")
args = parser.parse_args()

out_dir = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_aug_texts_v5_new"
meta_file = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full353_v2.0/meta_240527_v014.csv"
# out_dir = './out/bones_aug_texts_v5_dev'
# meta_file = '../bones_data/bones_full353_v2.0/meta_240527_v014.csv'

# meta_file = 'out/meta1.csv'
meta = pd.read_csv(meta_file)

os.makedirs(out_dir, exist_ok=True)


def paraphrase_text(text):
    try:
        all_para_texts = []
        templates_arr = sum(
            [paraphrase_templates[x] for x in args.template_key.split(",")], []
        )
        for ti, template in enumerate(templates_arr):
            prompt = template % text
            # print(prompt)
            do_iterate = ti in ITER_TEMPLATES
            reply = call_llama(prompt, iterate=do_iterate)
            # print(reply)
            reply = reply.replace('"', "").replace("'", "").replace("!", ".")
            para_texts = []
            for x in reply.split("\n"):
                if len(x) <= 4 or x[0] != "*":
                    continue
                if "*" in x[:4]:
                    text_p = x.split("*")[1].strip()
                    # if '*' in text_p:
                    #     raise Exception('Additional asterisk found!')
                    para_texts.append(text_p)
            para_texts = [x for x in para_texts if len(x) > 1]
            all_para_texts += para_texts
        if len(all_para_texts) != len(templates_arr) * NUM_EXPECTED:
            raise Exception(
                "Not the expected number of paraphrases. Something is possibly wrong!"
            )
        # print('num of para texts:', len(all_para_texts))
        # print(all_para_texts)
    except KeyboardInterrupt:
        sys.exit()
    except:
        return None
    return all_para_texts


# missing_idx = [9922, 9958, 10020, 10031, 10043, 10197, 10200, 10206, 10237, 10242, 10244, 10263, 11972, 11985, 11994, 12018, 12050, 12064, 12076, 12085, 12088, 12100, 12114, 12116, 81846, 81865, 81926, 81972, 98986, 135454, 135476, 135484, 135485, 135493, 135498, 135505, 135515, 135523, 135531, 135533, 135540, 135545, 135555, 135565, 137694, 137757, 137761, 137791, 137793, 137802, 137818, 137829, 137841, 137869, 137876, 137884, 137897, 137899, 137964, 150347, 150348, 150351, 150352, 150353, 150354, 158054, 169251, 169262, 169308, 169318, 169322, 169328, 169330, 169332, 169336, 169340, 169342, 169344, 169580, 169582, 170589, 170610, 170614, 170622, 170627, 170633, 170647, 170652, 170697, 170844, 198653, 215620, 215636, 215656, 215692, 215699, 215705, 215724, 215725, 215727, 215732, 215739, 215743, 215789, 215815, 215926, 218020, 218050, 218374, 218382, 218384, 218398, 218404, 218414, 218417, 218428, 218430, 218445, 218450, 218519, 218524, 228889, 228897, 228911, 228929, 228933, 228953, 228986, 229017, 229038, 229050, 229052, 229063, 229066, 229078, 229079, 230137, 230145, 230201, 230274, 230301, 230306, 230318, 230352, 230356, 230378, 230385, 230420, 230530, 230635, 230872, 243939, 243985, 244016, 244042, 244819, 244823, 244838, 244839, 244874, 244888, 244891, 244910, 244912, 244969, 245018, 246875, 247011, 247064, 247093, 247096, 247132, 247145, 247161, 247163, 247195, 247206, 247216, 247219, 247245, 247295, 264152, 264225, 264237, 264238, 264242, 264273, 264300, 264320, 264325, 264328, 264332, 264333, 264348, 264380, 264390, 265622, 266437, 266461, 266540, 266567, 266574, 266613, 266633, 266645, 266657, 266669, 266693, 267552, 267557, 267561, 282810, 282811, 282819, 282831, 282840, 282843, 282847, 282854, 282859, 282867, 282871, 282885, 282891, 282903, 282905, 285267, 285304, 285339, 285350, 285352, 285355, 285356, 285364, 285374, 285375, 285378, 285383, 285388, 285395, 285401, 329095, 329098, 329102, 329103, 329106, 329114, 329126, 329140, 329756, 332667, 332669, 332674, 332675, 332677, 332681, 332683, 332689, 332691, 332699, 332705, 332714, 332723, 332729, 332735, 334134, 334154, 334155, 334170, 334171, 334176, 334178, 334179, 334186, 334192, 334194, 334200, 334203, 334206, 334224, 339562, 339563, 349202]
for i in range(args.start, args.end):  # missing_idx
    t0 = time.time()

    row = meta.iloc[i]
    texts = [
        (str(row["natural_desc_1"]), 0),
        (str(row["natural_desc_2"]), 1),
        (str(row["natural_desc_3"]), 2),
        # (str(row['technical_description']), 3), # technical descriptions are not accurately paraphrased
        (str(row["short_description"]), 4),
    ]

    text_in = [text for text in texts if len(text[0]) > 2 and text[0] != "nan"]
    if len(text_in) > 0:
        text_str = ""
        for ti, text in enumerate(text_in):
            text_str += "* %s\n" % (text[0])

        # text_str = '. '.join([text[0] for text in text_in])
        # print(text_str)

        out_path = f"{out_dir}/{i:06d}-0.txt"
        if os.path.exists(out_path):
            print(f"[Exists] {out_path}")
            continue

        for _ in range(20):  # try 5 times to get something valid
            para_texts = paraphrase_text(text_str)
            if para_texts is not None:
                break

        if para_texts is None:
            print("Warning: paraphrase_text failed!")
            continue

        # write a different file for each annotation even though augmentations are the same
        #       for compatibility with data loader later
        for text_k, k in text_in:
            out_path = f"{out_dir}/{i:06d}-{k}.txt"
            try:
                text_list = [text_k] + para_texts
                write_list_to_file(out_path, text_list)
            except:
                print("Warning: write_list_to_file failed!")
                print(text_list)
                continue

    print(f"{i}/({args.start}, {args.end}) processed, time: {time.time() - t0:.2f}")
