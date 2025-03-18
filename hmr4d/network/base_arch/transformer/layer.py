import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class BasicBlock(nn.Module):
    """Basic 1D residual block."""

    def __init__(
        self,
        inplanes,
        planes,
        ker_size,
        stride=1,
        dropout=0.1,
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.LeakyReLU,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            planes,
            kernel_size=ker_size,
            stride=stride,
            padding=ker_size // 2,
            dilation=1,
            bias=True,
        )
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size // 2, bias=True
        )
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    inplanes,
                    planes,
                    stride=stride,
                    kernel_size=ker_size,
                    padding=ker_size // 2,
                    bias=True,
                ),
                norm_layer(planes),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x
