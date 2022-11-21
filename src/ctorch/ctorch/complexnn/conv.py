import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kw)

    def forward(self, x: torch.Tensor):
        out_real = self.conv2d(x.real) - self.conv2d(x.imag)
        out_imag = self.conv2d(x.real) + self.conv2d(x.imag)
        return out_real + (1j * out_imag)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kw):
        super().__init__()
        self.convtranspose2d = nn.ConvTranspose2d(in_channels, out_channels, **kw)

    def forward(self, x: torch.Tensor):
        out_real = self.convtranspose2d(x.real) - self.convtranspose2d(x.imag)
        out_imag = self.convtranspose2d(x.real) + self.convtranspose2d(x.imag)
        return out_real + (1j * out_imag)
