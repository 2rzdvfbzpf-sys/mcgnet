"""Multi-stream Cross-Gated Network (MCG-Net)."""
import torch
from torch import nn
from .config import ModelConfig
from .tpa_re import TPARE
from .dgfm import DGFM


class MCGNet(nn.Module):
    """Multi-stream Cross-Gated Network (MCG-Net)."""

    output_dim = 224

    def __init__(self, config, *, initialize=True):
        super().__init__()
        self.config = config
        if initialize:
            for name in ('lambda_init', 'batch_norm_momentum', 'linear_init_std', 'scale_init'):
                if getattr(config, name) is None:
                    raise ValueError('Training requires an explicit model.' + name)
        self.encoder_dt = TPARE(self.config)
        self.encoder_at = (None if self.config.share_modal_encoders
                           else TPARE(self.config))
        self.fusion = DGFM(self.config)
        if initialize:
            self.apply(self._initialize)
        self.scale = nn.Parameter(torch.empty(()) if self.config.scale_init is None
                                  else torch.tensor(float(self.config.scale_init)))

    def _initialize(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=self.config.linear_init_std)
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, dt, at):
        if dt.shape != at.shape or dt.ndim != 4 or tuple(dt.shape[1:]) != (1, 25, 64):
            raise ValueError('DTM and ATM must both have shape (batch,1,25,64)')
        if self.encoder_at is None:
            # Shared modality weights also share BN; process both modalities together.
            dt_feature, at_feature = self.encoder_dt(torch.cat((dt, at))).chunk(2)
        else:
            dt_feature, at_feature = self.encoder_dt(dt), self.encoder_at(at)
        return self.fusion(dt_feature, at_feature)
