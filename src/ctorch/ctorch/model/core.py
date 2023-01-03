import torch
import torch.nn as nn

from ctorch.config import ComplexTorchConfig
from ctorch.model.unet import UNet
from ctorch.utils.constants import (
    IN_CHANNELS,
    OUT_CHANNELS,
    HIDDEN_DIMENSIONS,
    PARAMETERS,
    HEIGHT,
    WIDTH
)


class Model(nn.Module):
    def __init__(self, config: ComplexTorchConfig):
        super().__init__()
        self.config = config
        self.params = self.config.MODEL_PARAMETERS
        H_difference = self.config.PREPROCESSING_NEW_HEIGHT - HEIGHT
        W_difference = self.config.PREPROCESSING_NEW_WIDTH - WIDTH
        self.pad = (
            round(W_difference / 2),
            W_difference - round(W_difference / 2),
            round(H_difference / 2),
            H_difference - round(H_difference / 2)
        )
        self.unet = UNet(
            in_channels=self.params[IN_CHANNELS],
            out_channels=self.params[OUT_CHANNELS],
            hidden_dims=self.params[HIDDEN_DIMENSIONS],
            pad=self.pad,
            **self.params[PARAMETERS]
        )

    def forward(self, x: torch.Tensor):
        return self.unet(x)