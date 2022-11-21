import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return F.relu(x.real) + (1j * F.relu(x.imag))


class ComplexSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return F.sigmoid(x.real) + (1j * F.sigmoid(x.imag))
