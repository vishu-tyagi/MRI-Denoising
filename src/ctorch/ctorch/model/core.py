import torch
import torch.nn as nn

from ctorch.model.encoder import Encoder
from ctorch.model.decoder import Decoder
from ctorch.model.mask import Mask


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: list[int],
        pad: tuple[int],
        **kw
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, **kw)
        self.decoder = Decoder(out_channels, hidden_dims, **kw)
        self.mask = Mask(pad)

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.mask(out)
        return out
