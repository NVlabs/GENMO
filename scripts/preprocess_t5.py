import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

from hmr4d.dataset.pure_motion.humanml3d import Humanml3dDataset


def load_and_freeze_llm(llm_version):
    tokenizer = T5Tokenizer.from_pretrained(llm_version)
    model = T5EncoderModel.from_pretrained(llm_version)
    # Freeze llm weights
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def encode_text(raw_text, has_text):
    # raw_text - list (batch_size length) of strings with input text prompts
    no_text = ~torch.tensor(has_text)
    device = "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            max_text_len = 50

            encoded = tokenizer.batch_encode_plus(
                raw_text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_text_len,
                truncation=True,
            )
            # We expect all the processing is done in GPU.
            input_ids = encoded.input_ids.to(device)
            attn_mask = encoded.attention_mask.to(device)

            with torch.no_grad():
                output = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
                encoded_text = output.last_hidden_state.detach()

            encoded_text = encoded_text[:, :max_text_len]
            attn_mask = attn_mask[:, :max_text_len]
            encoded_text *= attn_mask.unsqueeze(-1)
            # for bnum in range(encoded_text.shape[0]):
            #     nvalid_elem = attn_mask[bnum].sum().item()
            #     encoded_text[bnum][nvalid_elem:] = 0
    encoded_text[no_text] = 0
    return encoded_text


text_encoder, tokenizer = load_and_freeze_llm("t5-3b")
text_encoder.cuda()


torch.autograd.set_grad_enabled(False)

dataset = Humanml3dDataset(cam_augmentation="v11", split="test")
output_dir = "inputs/HumanML3D_SMPL/t5_embeddings_v1"
os.makedirs(output_dir, exist_ok=True)

text_embed_dict = {}

for i, (mid, data) in enumerate(dataset.motion_files.items()):
    text = [x["caption"] for x in data["text_data"]]
    has_text = [x != "" for x in text]
    text_embed = encode_text(text, has_text).cpu()
    # torch.save(text_embed, os.path.join(output_dir, f"{mid}.pth"))
    text_embed_dict[mid] = text_embed
    print(f"{i}/{len(dataset)} {mid}")

torch.save(text_embed_dict, os.path.join(output_dir, f"test_text_embed.pth"))
print(len(dataset))
