import numpy
import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf.dictconfig import DictConfig



def update_scheduled_params(obj, scheduler_dict, step):
    for target, cfg in scheduler_dict.items():
        sch_type = cfg['type']
        target_attr = target
        target_obj = obj
        if '.' in target:
            target_obj_str, target_attr = target.rsplit('.', 1)
            for x in target_obj_str.split('.'):
                target_obj = getattr(target_obj, x)
        if sch_type == 'linear':
            i = len(cfg['seg_vals']) - 1
            while step < cfg['seg_steps'][i]:
                i -= 1
            if i == len(cfg['seg_vals']) - 1:
                val = cfg['seg_vals'][i]
            else:
                t = (step - cfg['seg_steps'][i]) / (cfg['seg_steps'][i + 1] - cfg['seg_steps'][i])
                t = max(0.0, min(1.0, t))
                val = (1.0 - t) * cfg['seg_vals'][i] + t * cfg['seg_vals'][i + 1]
        elif sch_type == 'segment':
            i = len(cfg['seg_vals']) - 1
            while step < cfg['seg_steps'][i]:
                i -= 1
            val = cfg['seg_vals'][i]

        if type(val) is DictConfig or type(val) is dict:
            tmp_obj = getattr(target_obj, target_attr)
            if cfg.get('overwrite_dict', False):
                setattr(target_obj, target_attr, val)
            else:
                for k, v in val.items():
                    if type(tmp_obj) is dict:
                        tmp_obj[k] = v
                    else:
                        setattr(tmp_obj, k, v)
        else:
            setattr(target_obj, target_attr, val)

        if 'trigger_func' in cfg and step == cfg['seg_steps'][i]:
            target_obj = obj
            target_func = cfg['trigger_func']
            print(f'Triggering function: {target_func}')
            target_obj_str, target_func = target_func.rsplit('.', 1)
            for x in target_obj_str.split('.'):
                target_obj = getattr(target_obj, x)
            getattr(target_obj, target_func)()




class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, final_lr: float = 0.0, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.final_lr = final_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.num_warmup_steps:
            return [base_lr * float(current_step) / float(max(1, self.num_warmup_steps)) for base_lr in self.base_lrs]
        else:
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return [self.final_lr + (base_lr - self.final_lr) * cosine_decay for base_lr in self.base_lrs]



if __name__ == '__main__':

    class YourModel(torch.nn.Module):
        def __init__(self):
            super(YourModel, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = YourModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    

    num_warmup_steps = 1000
    num_training_steps = 10000
    final_lr = 0.0001

    scheduler = WarmupCosineScheduler(optimizer, num_warmup_steps, num_training_steps, final_lr)

    lrs = []
    for step in range(num_training_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    # Plotting the learning rate vs training steps
    import matplotlib.pyplot as plt

    plt.plot(range(num_training_steps), lrs)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Training Steps')
    # plt.show()
    plt.savefig('out/lr_vs_steps.png')