import torch


def load_and_freeze_llm(llm_version):
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(llm_version)
    model = T5EncoderModel.from_pretrained(llm_version)
    # Freeze llm weights
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def encode_text_batch(raw_text, text_encoder, tokenizer, device="cuda"):
    # raw_text - list (batch_size length) of strings with input text prompts

    with torch.no_grad():
        max_text_len = 50

        encoded = tokenizer.batch_encode_plus(
            raw_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_len,
            truncation=True,
        )
        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)

        output = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

        encoded_text = encoded_text[:, :max_text_len]
        attn_mask = attn_mask[:, :max_text_len]
        encoded_text *= attn_mask.unsqueeze(-1)

    return encoded_text
