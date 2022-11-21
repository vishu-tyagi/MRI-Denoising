from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctorch.utils.constants import (HEIGHT, WIDTH)


class Mask(nn.Module):
    def __init__(self, pad: tuple[int]):
        super().__init__()
        mask = F.pad(torch.ones(HEIGHT, WIDTH), pad, mode="constant", value=0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        return torch.mul(x, self.mask)
