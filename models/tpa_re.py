"""Temporal-Physical Attention Radar Encoder (TPA-RE)."""
import torch
from torch import nn


class TemporalPhysicalAttention(nn.Module):
    """Temporal-Physical Attention (TPA)."""

    def __init__(self, channels, hidden, config):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=config.bias),
                                    nn.BatchNorm2d(hidden, eps=config.batch_norm_eps,
                                                   momentum=config.batch_norm_momentum), nn.Mish())
        self.time_head = nn.Conv2d(hidden, channels, 1, bias=config.bias)
        self.physical_head = nn.Conv2d(hidden, channels, 1, bias=config.bias)

    def forward(self, t3):
        height, width = t3.shape[-2:]
        temporal = t3.mean(dim=3, keepdim=True)
        physical = t3.mean(dim=2, keepdim=True).transpose(2, 3)
        combined = self.shared(torch.cat((temporal, physical), dim=2))
        temporal, physical = torch.split(combined, (height, width), dim=2)
        at = self.time_head(temporal).sigmoid()
        ap = self.physical_head(physical.transpose(2, 3)).sigmoid()
        return t3 + at * t3 + ap * t3  # TPA residual and axis-wise attention aggregation.


class TPAResBlock(nn.Module):
    """TPA-Res block."""

    def __init__(self, channels, attention_channels, config):
        super().__init__()
        self.convolutions = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(channels, channels, config.convolution_kernel,
                                     padding=config.convolution_kernel // 2, bias=config.bias),
                          nn.BatchNorm2d(channels, eps=config.batch_norm_eps,
                                         momentum=config.batch_norm_momentum), nn.Mish())
            for _ in range(config.convolutions_per_block)])
        self.attention = TemporalPhysicalAttention(channels, attention_channels, config)

    def forward(self, x):
        t2 = self.convolutions(x)
        return self.attention(t2 + x)  # Apply TPA to the residual feature T3.


class TPARE(nn.Module):
    """Temporal-Physical Attention Radar Encoder (TPA-RE)."""

    output_dim = 112

    def __init__(self, config):
        super().__init__()
        c = config.channels
        self.mapping = nn.Sequential(nn.Conv2d(1, c, config.convolution_kernel,
                                               padding=config.convolution_kernel // 2, bias=config.bias),
                                     nn.BatchNorm2d(c, eps=config.batch_norm_eps,
                                                    momentum=config.batch_norm_momentum), nn.Mish(),
                                     nn.MaxPool2d(config.mapping_pool_size, config.mapping_pool_size))
        self.blocks = nn.ModuleList([TPAResBlock(c, config.attention_channels, config)
                                     for _ in range(config.block_count)])
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(c, dim, bias=config.bias),
                          nn.BatchNorm1d(dim, eps=config.batch_norm_eps,
                                         momentum=config.batch_norm_momentum), nn.Mish())
            for dim in config.projection_dims])
        self.level_weights = nn.Parameter(torch.empty(len(config.projection_dims))
                                          if config.lambda_init is None else
                                          torch.tensor(config.lambda_init, dtype=torch.float32))

    def forward(self, x):
        x = self.mapping(x)
        levels = []
        for block, projection, weight in zip(self.blocks, self.projections, self.level_weights):
            x = block(x)
            levels.append(weight * projection(x.mean(dim=(2, 3))))
        return torch.cat(levels, dim=1)  # Concatenate three feature levels into a 112-dimensional embedding.
