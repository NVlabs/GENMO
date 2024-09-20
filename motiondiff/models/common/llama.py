import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class LLAMADecoder(nn.Module):
    
    def __init__(self, config):
        super(LLAMADecoder, self).__init__()
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if 'pretrained_cp_path' in config:
            self.load_pretrained(config.pretrained_cp_path, config)
        
    def forward(self, hidden_states, src_key_padding_mask=None, attention_mask=None, position_ids=None, **kwargs):
        hidden_states = hidden_states.transpose(0, 1)
        if src_key_padding_mask is not None:
            attention_mask = torch.zeros(hidden_states.shape[0], 1, hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
            attention_mask += src_key_padding_mask[:, None, None, :] * -1e9
            attention_mask += src_key_padding_mask[:, None, :, None] * -1e9
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long).repeat(hidden_states.shape[0], 1)
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask, position_ids)
            hidden_states = layer_outputs[0]
        hidden_states = hidden_states.transpose(0, 1)
        return hidden_states
    
    def load_pretrained(self, cp_path, config):
        state_dict = torch.load(cp_path, map_location='cpu')
        print('Loading pretrained LLAMA weights...')
        for layer_idx, layer in enumerate(self.layers):
            layer_dict = {
                'self_attn.q_proj.weight': state_dict[f'layers.{layer_idx}.attention.wq.weight'],
                'self_attn.k_proj.weight': state_dict[f'layers.{layer_idx}.attention.wk.weight'],
                'self_attn.v_proj.weight': state_dict[f'layers.{layer_idx}.attention.wv.weight'],
                'self_attn.o_proj.weight': state_dict[f'layers.{layer_idx}.attention.wo.weight'],
                'input_layernorm.weight': state_dict[f'layers.{layer_idx}.attention_norm.weight'],
                'post_attention_layernorm.weight': state_dict[f'layers.{layer_idx}.ffn_norm.weight'],
            }
            if config.pretrained_load_ffn:
                layer_dict.update({
                    'mlp.gate_proj.weight': state_dict[f'layers.{layer_idx}.feed_forward.w1.weight'],
                    'mlp.up_proj.weight': state_dict[f'layers.{layer_idx}.feed_forward.w3.weight'],
                    'mlp.down_proj.weight': state_dict[f'layers.{layer_idx}.feed_forward.w2.weight']
                })
            missing_keys, unexpected_keys = layer.load_state_dict(layer_dict, strict=False)
            if layer_idx == 0:
                if len(missing_keys) > 0:
                    print(f'[LLAMADecoder] layer{layer_idx} Missing keys: {missing_keys}')
                if len(unexpected_keys) > 0:
                    print(f'[LLAMADecoder] layer{layer_idx} Unexpected keys: {unexpected_keys}')
        return