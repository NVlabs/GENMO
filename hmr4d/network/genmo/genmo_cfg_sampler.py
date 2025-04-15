from copy import deepcopy

import torch


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel:
    def __init__(self, model):
        self.model = model  # model is the actual model to run

    def __call__(self, x, timesteps, y=None, **kwargs):
        y_uncond = deepcopy(y)
        y_uncond["encoded_text"] = torch.zeros_like(y["encoded_text"])
        y_uncond["f_cond"] = y["f_uncond"]
        if "multi_text_data" in y:
            y_uncond["multi_text_data"]["text_embed"] = torch.zeros_like(
                y["multi_text_data"]["text_embed"]
            )

        out = self.model(x, timesteps, y, **kwargs)
        out_uncond = self.model(x, timesteps, y_uncond, **kwargs)
        outputs = dict()
        for k in out:
            outputs[k] = out_uncond[k] + y["scale"] * (out[k] - out_uncond[k])
        return outputs

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
