"""DTM-Guided Fusion Module (DGFM)."""
import torch
from torch import nn


class DGFM(nn.Module):
    """DTM-Guided Fusion Module (DGFM)."""

    def __init__(self, config):
        super().__init__()
        self.guide = nn.Sequential(nn.Linear(112, config.dgfm_hidden_dim, bias=config.bias), nn.Mish(),
                                   nn.Linear(config.dgfm_hidden_dim, 112, bias=config.bias), nn.Sigmoid())

    def forward(self, dt, at):
        return torch.cat((dt, at * self.guide(dt)), dim=1)
