import torch
import torch.nn as nn

from ctorch.config import ComplexTorchConfig
from ctorch.complexnn import (DoubleConv, Down)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: int, **kw):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.params = kw
        self.layers = nn.ModuleList()
        self._layers_init()

    def _layers_init(self):
        current_dim = self.in_channels
        for i, hdim in enumerate(self.hidden_dims):
            if not i:
                self.layers.append(DoubleConv(current_dim, hdim, **self.params))
                current_dim = hdim
                continue
            self.layers.append(Down(current_dim, hdim, **self.params))
            current_dim = hdim

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = list()
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out
