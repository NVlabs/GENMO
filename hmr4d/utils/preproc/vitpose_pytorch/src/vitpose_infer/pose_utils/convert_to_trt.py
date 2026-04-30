import torch
from builder import build_model
from torch2trt import TRTModule, torch2trt

pose = build_model("ViTPose_base_coco_256x192", "./models/vitpose-b.pth")
pose.cuda().eval()

x = torch.ones(1, 3, 256, 192).cuda()
net_trt = torch2trt(pose, [x], max_batch_size=10, fp16_mode=True)
torch.save(net_trt.state_dict(), "vitpose_trt.pth")
