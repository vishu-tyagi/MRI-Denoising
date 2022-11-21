import torch
import torch.nn as nn

from ctorch.complexnn.conv import (ComplexConv2d, ComplexConvTranspose2d)
from ctorch.complexnn.maxpool import (ComplexMaxPool2d)
from ctorch.complexnn.activation import ComplexRelu
from ctorch.utils.constants import (CONV2D, MAXPOOL2D, CONVTRANSPOSE2D, OUTCONV2D)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self._model = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, **kw[CONV2D]),
            ComplexRelu(),
            ComplexConv2d(out_channels, out_channels, **kw[CONV2D]),
            ComplexRelu()
        )

    def forward(self, x: torch.Tensor):
        return self._model(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self._model = nn.Sequential(
            ComplexMaxPool2d(**kw[MAXPOOL2D]),
            DoubleConv(in_channels, out_channels, **kw)
        )

    def forward(self, x: torch.Tensor):
        return self._model(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self.up_conv2d = ComplexConvTranspose2d(in_channels, out_channels, **kw[CONVTRANSPOSE2D])
        self.doubleconv = DoubleConv(out_channels * 2, out_channels, **kw)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.up_conv2d(x)
        out = torch.cat([x, y], dim=1)
        out = self.doubleconv(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self._model = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, **kw[OUTCONV2D])
        )

    def forward(self, x: torch.Tensor):
        return self._model(x)
