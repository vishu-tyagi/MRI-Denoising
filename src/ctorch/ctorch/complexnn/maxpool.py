import torch
import torch.nn as nn


class ComplexMaxPool2d(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(**kw, return_indices=True)

    def forward(self, x: torch.Tensor):
        x_abs = torch.abs(x)
        _, indices = self.max_pool2d(x_abs)
        x_flat = torch.flatten(x, start_dim=-2)
        indices_flat = torch.flatten(indices, start_dim=-2)
        out = torch.gather(x_flat, index=indices_flat, dim=-1)
        out = out.reshape(indices.shape)
        return out
