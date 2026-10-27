"""Encoder, fusion and input configurations for MCG-Net ablation."""
import _bootstrap
import numpy as np
import torch
from torch import nn

from models import MCGNet, ModelConfig
from models.tpa_re import TPARE
from models.cpc import cosine_prototype_logits


VARIANTS = {
    'resnet12_concat': ('ResNet-12', False, 'concat', 'dt_at'),
    'tpa_res_concat': ('TPA-Res', False, 'concat', 'dt_at'),
    'tpa_re_concat': ('TPA-RE', True, 'concat', 'dt_at'),
    'tpa_re_weighted': ('TPA-RE', True, 'weighted', 'dt_at'),
    'tpa_re_cca': ('TPA-RE', True, 'cca', 'dt_at'),
    'tpa_re_dgfm': ('TPA-RE', True, 'dgfm', 'dt_at'),
    'dtm_only': ('TPA-RE', True, 'none', 'dt'),
    'atm_only': ('TPA-RE', True, 'none', 'at'),
}
DEFAULT_VARIANTS = list(VARIANTS)


class ResidualBlock(nn.Module):
    """Three-convolution residual block with Mish and one spatial pooling."""
    def __init__(self, cin, cout, config):
        super().__init__()
        layers = []
        for index in range(3):
            layers.extend([nn.Conv2d(cin if index == 0 else cout, cout, 3,
                                     padding=1, bias=config.bias),
                           nn.BatchNorm2d(cout, eps=config.batch_norm_eps,
                                          momentum=config.batch_norm_momentum)])
            if index < 2:
                layers.append(nn.Mish())
        self.main = nn.Sequential(*layers)
        self.skip = (nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=config.bias),
            nn.BatchNorm2d(cout, eps=config.batch_norm_eps,
                           momentum=config.batch_norm_momentum)))
        self.activation = nn.Mish()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.activation(self.main(x) + self.skip(x)))


class ResNet12(nn.Module):
    """Four residual stages with twelve main-branch convolutions."""
    output_dim = 112

    def __init__(self, config, widths):
        super().__init__()
        cin, blocks = 1, []
        for width in widths:
            blocks.append(ResidualBlock(cin, width, config))
            cin = width
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Sequential(nn.Linear(cin, 112, bias=config.bias),
            nn.BatchNorm1d(112, eps=config.batch_norm_eps,
                            momentum=config.batch_norm_momentum), nn.Mish())

    def forward(self, x):
        return self.projection(self.blocks(x).mean((2, 3)))


class LastLevelEncoder(TPARE):
    """Feature Mapping and three TPA-Res blocks with a last-level embedding."""
    def __init__(self, config):
        super().__init__(config)
        del self.projections
        del self.level_weights
        self.projection = nn.Sequential(nn.Linear(config.channels, 112, bias=config.bias),
            nn.BatchNorm1d(112, eps=config.batch_norm_eps,
                            momentum=config.batch_norm_momentum), nn.Mish())

    def forward(self, x):
        x = self.mapping(x)
        for block in self.blocks:
            x = block(x)
        return self.projection(x.mean((2, 3)))


class AblationNetwork(nn.Module):
    """CPC embedding network for one encoder/fusion/input configuration."""
    _initialize = MCGNet._initialize

    def __init__(self, variant, config, settings):
        super().__init__()
        self.config, self.variant = config, variant
        encoder, _, self.fusion_kind, self.inputs = VARIANTS[variant]
        if variant == 'tpa_re_cca':
            raise ValueError('CCA requires a fitted Concatenation encoder')
        def make_encoder():
            if encoder == 'ResNet-12':
                return ResNet12(config, settings['resnet12_channels'])
            return TPARE(config) if encoder == 'TPA-RE' else LastLevelEncoder(config)
        self.encoder_dt = None if self.inputs == 'at' else make_encoder()
        self.encoder_at = None if self.inputs == 'dt' else make_encoder()
        if self.fusion_kind == 'weighted':
            self.fusion_weights = nn.Parameter(torch.tensor(settings['fusion_weight_logits'], dtype=torch.float32))
        self.apply(self._initialize)
        self.scale = nn.Parameter(torch.tensor(float(config.scale_init)))

    def paired(self, dt, at):
        return self.encoder_dt(dt), self.encoder_at(at)

    def forward(self, dt, at):
        if self.inputs == 'dt':
            return self.encoder_dt(dt)
        if self.inputs == 'at':
            return self.encoder_at(at)
        left, right = self.paired(dt, at)
        if self.fusion_kind == 'weighted':
            weights = self.fusion_weights.softmax(0)
            return torch.cat((weights[0] * left, weights[1] * right), 1)
        return torch.cat((left, right), 1)


class CCAClassifier(nn.Module):
    """Frozen paired encoders followed by fitted linear CCA projections."""
    def __init__(self, base, dimension):
        super().__init__()
        self.base, self.config = base, base.config
        self.dimension = dimension
        self.register_buffer('mean_dt', torch.zeros(112))
        self.register_buffer('mean_at', torch.zeros(112))
        self.register_buffer('projection_dt', torch.zeros(112, dimension))
        self.register_buffer('projection_at', torch.zeros(112, dimension))
        self.requires_grad_(False)

    @property
    def scale(self):
        return self.base.scale

    def forward(self, dt, at):
        left, right = self.base.paired(dt, at)
        return torch.cat(((left - self.mean_dt) @ self.projection_dt,
                          (right - self.mean_at) @ self.projection_at), 1)


def build_model(variant, config, settings, cca_dimension=None):
    config = ModelConfig(**config) if isinstance(config, dict) else config
    if variant == 'tpa_re_dgfm':
        return MCGNet(config)
    if variant == 'tpa_re_cca':
        if cca_dimension is None:
            raise ValueError('A fitted CCA dimension is required')
        return CCAClassifier(AblationNetwork('tpa_re_concat', config, settings), cca_dimension)
    return AblationNetwork(variant, config, settings)


def scores(model, dt, at, way, shot, query):
    z = model(dt, at).reshape(way, shot + query, -1)
    return model.scale * cosine_prototype_logits(z[:, :shot], z[:, shot:].reshape(way * query, -1),
                                                 model.config.normalization_eps)


def fit_cca(left, right, regularization):
    """Regularized linear CCA fitted on paired training features only."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or len(left) < 2:
        raise ValueError('CCA requires at least two aligned feature pairs')
    means = left.mean(0), right.mean(0)
    x, y = left - means[0], right - means[1]
    def whitening(value):
        cov = value.T @ value / (len(value) - 1)
        ridge = regularization * max(float(np.trace(cov) / len(cov)), 1e-12)
        eigenvalues, vectors = np.linalg.eigh(cov + ridge * np.eye(len(cov)))
        return (vectors * np.maximum(eigenvalues, 1e-12) ** -0.5) @ vectors.T
    wx, wy = whitening(x), whitening(y)
    u, correlations, vh = np.linalg.svd(wx @ (x.T @ y / (len(x) - 1)) @ wy, full_matrices=False)
    return means, (wx @ u, wy @ vh.T), correlations
