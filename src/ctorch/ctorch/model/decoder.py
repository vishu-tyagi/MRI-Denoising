import torch
import torch.nn as nn

from ctorch.config import ComplexTorchConfig
from ctorch.complexnn import (Up, OutConv)


class Decoder(nn.Module):
    def __init__(self, out_channel, hidden_dims, **kw):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.out_channels = out_channel
        self.layers = nn.ModuleList()
        current_dim = self.hidden_dims[-1]
        for hdim in self.hidden_dims[-2::-1]:
            self.layers.append(Up(current_dim, hdim, **kw))
            current_dim = hdim
        self.layers.append(OutConv(current_dim, self.out_channels, **kw))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode output of encoder

        Args:
            x (list[torch.Tensor]): Output of encoder

        Returns:
            torch.Tensor: Decoded output
        """
        n = len(self.hidden_dims)
        out = x[-1]
        for i, layer in enumerate(self.layers[:-1]):
            y = x[n - (i + 2)]
            out = layer(out, y)
        out = self.layers[-1](out)
        return out
