"""MCG-Net configuration."""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import torch


@dataclass(frozen=True)
class ModelConfig:
    """MCG-Net architecture configuration."""

    channels: int
    projection_dims: Tuple[int, int, int]
    attention_channels: int
    lambda_init: Optional[Tuple[float, float, float]]
    share_modal_encoders: bool
    convolution_kernel: int
    mapping_pool_size: int
    block_count: int
    convolutions_per_block: int
    dgfm_hidden_dim: int
    bias: bool
    batch_norm_eps: float
    batch_norm_momentum: Optional[float]
    normalization_eps: float
    linear_init_std: Optional[float]
    scale_init: Optional[float]

    @classmethod
    def from_inference(cls, settings):
        values = dict(settings)
        for name in ('lambda_init', 'batch_norm_momentum', 'linear_init_std', 'scale_init'):
            values[name] = None
        return cls(**values)

    def __post_init__(self):
        object.__setattr__(self, 'projection_dims', tuple(self.projection_dims))
        if self.lambda_init is not None:
            object.__setattr__(self, 'lambda_init', tuple(self.lambda_init))
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError('channels must be a positive integer')
        if type(self.attention_channels) is not int or self.attention_channels < 1:
            raise ValueError('attention_channels must be a positive integer')
        if (len(self.projection_dims) != 3 or sum(self.projection_dims) != 112
                or any(type(x) is not int or x < 1 for x in self.projection_dims)):
            raise ValueError('three positive projection dimensions must sum to 112')
        if self.lambda_init is not None and (len(self.lambda_init) != 3 or not all(
                isinstance(x, (int, float)) and torch.isfinite(torch.tensor(float(x)))
                for x in self.lambda_init)):
            raise ValueError('lambda_init must contain three finite numbers')
        if type(self.share_modal_encoders) is not bool:
            raise ValueError('share_modal_encoders must be boolean')
        for name, expected in (('convolution_kernel', 3), ('mapping_pool_size', 2),
                               ('block_count', 3), ('convolutions_per_block', 3)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError('%s must be %d for TPA-RE' % (name, expected))
        if type(self.dgfm_hidden_dim) is not int or self.dgfm_hidden_dim < 1:
            raise ValueError('dgfm_hidden_dim must be a positive integer')
        if type(self.bias) is not bool:
            raise ValueError('bias must be boolean')
        for name in ('batch_norm_eps', 'normalization_eps', 'linear_init_std', 'scale_init'):
            value = getattr(self, name)
            if value is None and name in ('linear_init_std', 'scale_init'):
                continue
            if type(value) not in (int, float) or not 0 < value < float('inf'):
                raise ValueError(name + ' must be finite and positive')
        if self.batch_norm_momentum is not None and (type(self.batch_norm_momentum) not in (int, float)
                                                    or not 0 < self.batch_norm_momentum <= 1):
            raise ValueError('batch_norm_momentum must be in (0,1]')

    def to_dict(self):
        return asdict(self)
