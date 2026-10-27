"""ML-HGR-Net: episodic radar gesture classification."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
METHOD = 'ml_hgr_net'
TITLE = 'ML-HGR-Net'
PROVENANCE = {'status': 'paper_checked_radar_input_adaptation', 'input': 'DTM/ATM;25x64;independent per-map min-max [0,1]', 'protocol': 'seeded30/10/10 class split;no subject filtering;no within-episode sample replacement', 'scope': 'paper-informed radar input adaptation;not a reproduction of the original dataset or reported accuracy', 'method': 'ML-HGR-Net', 'paper_title': 'ML-HGR-Net: A Meta-Learning Network for FMCW Radar Based Hand Gesture Recognition', 'paper': 'https://doi.org/10.1109/JSEN.2022.3169231', 'inspected_sections': ['IV-B/C feature cubes and four-convolution extractor,p.10811', 'IV-C/D/E,figures4-6 and time connection,p.10812', 'IV-F,Algorithm1,equations9-10,p.10813', 'V-C experimental settings,p.10814', 'V-F fivefold5-shot relation cost,p.10816'], 'mechanism': 'Two four-convolution feature branches;Conv-ReLU-BN-maxpool per feature layer;Hadamard product;support/query connection along time;two relation convolution blocks and two FC layers;sigmoid scores;MSE over individual support/query pairs;inter-task batch normalization of feature-extractor inputs', 'input_adaptation': 'Original RDM and ADM32x32x32 feature cubes are replaced by native2D DTM/ATM25x64 maps;four Conv2d layers per branch and two Conv2d relation layers;no synthetic third dimension', 'connection': 'Time is the first spatial axis in cached25x64 maps;feature shape13x52,connected shape26x52;each support example is compared separately;class prediction averages itsK relation scores', 'normalization': 'Training feature BN sees all examples of a multi-task batch in one forward;evaluation uses accumulated training BN statistics without query-label fitting', 'inter_task_normalization_reference': {'paper': 'https://proceedings.mlr.press/v97/li19c.html', 'section': '3.6', 'cited_as': 'Reference26 in ML-HGR-Net'}, 'dimension_interpretation': 'Original text states32^3 input and20^3 output with four Conv3/kernel2-pool layers;valid convolution and pool stride1 give32->29->26->23->20.This geometry-matching choice is inferred because padding/pool strides are not reported.', 'implementation_choices_not_specified_by_paper': {'independent_branch_weights': True, 'encoder_channels': [16, 32, 64, 64], 'relation_channels': [64, 64], 'relation_hidden': 64, 'encoder_padding': 0, 'encoder_pool_stride': 1, 'relation_padding': 0, 'relation_pool_stride': 2, 'feature_order': 'Conv-ReLU-BN-pool as listed in IV-C', 'relation_order': 'Conv-ReLU-BN-pool', 'meta_batch_size': 4, 'pair_chunk_size': 64, 'class_score_reduction': 'mean_of_support_relation_scores', 'convolution_bias': True, 'batch_norm_eps': 1e-05, 'batch_norm_momentum': 0.1}, 'memory_management': 'Relation pairs are processed in bounded batches;training relation BN uses the configured pair batches;non-reentrant activation checkpoint recomputation suppresses a second running-statistics update', 'native_cube_model': 'Model also accepts spatial_dims3/model_input_shape[32,32,32]/time_axis2 tensors for architecture checks;the provided cache CLI explicitly requires2D DTM/ATM', 'training_adaptation': 'Paper reports Adam and10000 training/100 testing tasks but does not give the learning rate;this benchmark uses the manuscript-specified Adam1e-4 and common training/validation/testing budgets', 'paper_pdf': {'filename': 'ML-HGR-Net_A_Meta-Learning_Network_for_FMCW_Radar_Based_Hand_Gesture_Recognition.pdf', 'sha256': '88392840797e637b26b0658468db70be45ae168367cb5b912e8424ce85b8fb5e'}}
SPLIT_KEYS = ('train_classes', 'validation_classes', 'test_classes')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_save(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_runtime(cfg):
    rt = cfg['runtime']
    workspace = rt['cublas_workspace_config']
    if workspace not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported CUBLAS workspace configuration')
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
        raise RuntimeError('Start a fresh Python process for the CUBLAS workspace setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    deterministic = rt['deterministic_algorithms']
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if rt['cpu_threads'] is not None:
        torch.set_num_threads(rt['cpu_threads'])
    name = rt['device']
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


class RadarCache:
    """Aligned DTM/ATM maps and original gesture identifiers."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.labels = np.load(self.path / 'labels.npy', allow_pickle=False)
        if self.labels.ndim != 1 or self.labels.dtype.kind not in 'iu' or not len(self.labels):
            raise ValueError('labels.npy must be a nonempty integer vector')
        self.arrays = [np.load(self.path / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in ('dt', 'at')]
        for arr in self.arrays:
            if arr.shape != (len(self.labels), 25, 64) or arr.dtype.kind != 'f':
                raise ValueError('dt.npy and at.npy must be floating point [samples,25,64]')
            for start in range(0, len(arr), 512):
                if not np.isfinite(arr[start:start + 512]).all():
                    raise ValueError('Nonfinite radar input')
        mapping = read_json(self.path / 'label_mapping.json')
        if not isinstance(mapping, dict) or any(type(v) is not int for v in mapping.values()):
            raise ValueError('label_mapping.json must map original IDs to integer cache labels')
        self.original = {v: int(k) for k, v in mapping.items()}
        if len(self.original) != len(mapping) or not set(np.unique(self.labels)).issubset(self.original):
            raise ValueError('Incomplete or nonunique original-ID mapping')
        self.signature = {name: file_hash(self.path / name) for name in
                          ('dt.npy', 'at.npy', 'labels.npy', 'label_mapping.json')}

    def batch(self, indices, device):
        x = np.stack([np.asarray(a[indices], dtype=np.float32) for a in self.arrays], axis=1)
        low = x.min(axis=(-2, -1), keepdims=True)
        span = x.max(axis=(-2, -1), keepdims=True) - low
        x = np.divide(x - low, span, out=np.zeros_like(x), where=span > 0)
        return torch.from_numpy(x).to(device)


def split_classes(labels, seed, counts):
    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    if len(classes) != sum(counts):
        raise ValueError('Cache class count must equal the sum of class_counts')
    np.random.RandomState(seed).shuffle(classes)
    a, b = counts[0], counts[0] + counts[1]
    return dict(zip(SPLIT_KEYS, (classes[:a].tolist(), classes[a:b].tolist(), classes[b:].tolist())))


def plan_seed(base, shot, stage, epoch=0):
    return int(np.random.SeedSequence([base, 0, shot, stage, epoch]).generate_state(1)[0])


def episode_plan(labels, classes, way, shot, query, count, seed):
    pools = {int(c): np.flatnonzero(labels == c) for c in classes}
    if len(classes) < way or any(len(p) < shot + query for p in pools.values()):
        raise ValueError('Insufficient classes or distinct support/query samples')
    rng = np.random.default_rng(seed)
    return np.asarray([[rng.choice(pools[int(c)], shot + query, replace=False)
                        for c in rng.choice(classes, way, replace=False)] for _ in range(count)], dtype=np.int64)


def stats(values, unit='episode'):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
        raise ValueError('Accuracies must be a finite nonempty vector in [0,1]')
    mean = float(v.mean())
    var = float(v.var(ddof=1)) if len(v) > 1 else None
    sd = math.sqrt(var) if var is not None else None
    half = 1.96 * sd / math.sqrt(len(v)) if sd is not None else None
    return {'statistical_unit': unit, 'episodes' if unit == 'episode' else 'runs': len(v),
            'mean_accuracy': mean, 'mean_accuracy_percent': mean * 100,
            'sample_variance_accuracy_0_1': var,
            'sample_variance_pp2': None if var is None else var * 10000,
            'sample_sd_pp': None if sd is None else sd * 100, 'ddof': 1,
            'ci95_method': 'normal_approximation_1.96_sd_over_sqrt_n',
            'ci95_half_width_pp': None if half is None else half * 100,
            'ci95_low_percent': None if half is None else (mean - half) * 100,
            'ci95_high_percent': None if half is None else (mean + half) * 100}


def print_stats(label, s):
    print('%s | Episodes: %d | Mean accuracy: %.4f%%' % (label, s['episodes'], s['mean_accuracy_percent']), flush=True)
    if s['sample_variance_pp2'] is None:
        print('  Sample variance: N/A | Sample SD: N/A | Mean accuracy +/- 95% CI: N/A', flush=True)
    else:
        print('  Sample variance: %.6f pp^2 (ddof=1) | Sample SD: %.6f pp' %
              (s['sample_variance_pp2'], s['sample_sd_pp']), flush=True)
        print('  Mean accuracy +/- 95%% CI: %.4f%% +/- %.4f pp | Interval: [%.4f%%, %.4f%%]' %
              (s['mean_accuracy_percent'], s['ci95_half_width_pp'], s['ci95_low_percent'], s['ci95_high_percent']), flush=True)


def conv_block(cin, cout, pool, padding, leaky, momentum, bias):
    blocks = [nn.Conv2d(cin, cout, 3, padding=padding, bias=bias),
              nn.BatchNorm2d(cout, momentum=momentum), nn.LeakyReLU(.2) if leaky else nn.ReLU()]
    if pool:
        blocks.append(nn.MaxPool2d(2))
    return nn.Sequential(*blocks)


def squared_distance(query, prototypes):
    return (query[:, None] - prototypes[None]).square().sum(-1)


def local_scores(support, query, neighbors):
    way, shot, channels, height, width = support.shape
    bank = F.normalize(support.permute(0, 2, 1, 3, 4).reshape(way, channels, -1), dim=1, eps=1e-12)
    descriptors = F.normalize(query.flatten(2).transpose(1, 2), dim=-1, eps=1e-12)
    if neighbors > bank.shape[-1]:
        raise ValueError('neighbor_k exceeds the support descriptor bank')
    return torch.stack([(descriptors @ bank[c]).topk(neighbors, dim=-1).values.sum((1, 2))
                        for c in range(way)], dim=1)


def episode_scores(model, support, query, target, training):
    return model(support, query)


def episode_loss(model, scores, target):
    return F.cross_entropy(scores, target)


def clip_gradients(model, options):
    if options.get('clip_norm') is not None:
        nn.utils.clip_grad_norm_(model.parameters(), options['clip_norm'])


def pretrain_model(model, cache, split, cfg, seed, shot, directory, device):
    return None


class FeatureBranch(nn.Sequential):
    """Four Conv-ReLU-BN-pool layers for one radar feature view."""
    def __init__(self, options):
        dims = options['spatial_dims']
        convolution = nn.Conv2d if dims == 2 else nn.Conv3d
        batch_norm = nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d
        pooling = nn.MaxPool2d if dims == 2 else nn.MaxPool3d
        channels = options['encoder_channels']
        super().__init__(*[nn.Sequential(convolution(1 if i == 0 else channels[i - 1], c, 3,
                          padding=options['encoder_padding'], bias=options['convolution_bias']),
                          nn.ReLU(), batch_norm(c), pooling(2, stride=options['encoder_pool_stride']))
                          for i, c in enumerate(channels)])


class RelationModule(nn.Sequential):
    """Two convolution blocks and two fully connected layers with sigmoid output."""
    def __init__(self, options, shape):
        dims = options['spatial_dims']
        convolution = nn.Conv2d if dims == 2 else nn.Conv3d
        batch_norm = nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d
        pooling = nn.MaxPool2d if dims == 2 else nn.MaxPool3d
        previous = options['encoder_channels'][-1]
        layers = []
        for channels in options['relation_channels']:
            layers.extend([convolution(previous, channels, 3, padding=options['relation_padding'],
                                       bias=options['convolution_bias']), nn.ReLU(), batch_norm(channels),
                           pooling(2, stride=options['relation_pool_stride'])])
            previous = channels
            shape = [(n + 2 * options['relation_padding'] - 3) // 1 + 1 for n in shape]
            shape = [(n - 2) // options['relation_pool_stride'] + 1 for n in shape]
        if min(shape) < 1:
            raise ValueError('Input is too small for the relation module')
        fc_input = previous * math.prod(shape)
        layers.extend([nn.Flatten(), nn.Linear(fc_input, options['relation_hidden']), nn.ReLU(),
                       nn.Linear(options['relation_hidden'], 1), nn.Sigmoid()])
        super().__init__(*layers)
        self.fc_input = fc_input


class Model(nn.Module):
    """Dual feature branches, Hadamard fusion and per-support relation scores."""
    def __init__(self, options, way, train_classes):
        super().__init__()
        self.options = options
        self.encoder_dt = FeatureBranch(options)
        self.encoder_at = FeatureBranch(options)
        shape = list(options['model_input_shape'])
        for _ in options['encoder_channels']:
            shape = [n + 2 * options['encoder_padding'] - 2 for n in shape]
            shape = [(n - 2) // options['encoder_pool_stride'] + 1 for n in shape]
        if min(shape) < 1:
            raise ValueError('Input is too small for the feature extractor')
        self.feature_shape = tuple(shape)
        shape[options['time_axis']] *= 2
        self.relation = RelationModule(options, shape)
        self._pair_scores = None

    def encode(self, x):
        return self.encoder_dt(x[:, :1]) * self.encoder_at(x[:, 1:])

    def _relation_chunk(self, support, query, pair_indices):
        n_support = len(support)
        si = torch.remainder(pair_indices, n_support)
        qi = torch.div(pair_indices, n_support, rounding_mode='floor')
        pair = torch.cat((support.index_select(0, si), query.index_select(0, qi)),
                         dim=2 + self.options['time_axis'])
        return self.relation(pair)

    def _recompute_context(self):
        from contextlib import contextmanager
        @contextmanager
        def context():
            batch_norms = [m for m in self.relation.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d))]
            states = [(m, m.momentum, m.num_batches_tracked.detach().clone()) for m in batch_norms]
            try:
                for m, _, _ in states:
                    m.momentum = 0.
                yield
            finally:
                for m, momentum, count in states:
                    m.momentum = momentum
                    m.num_batches_tracked.copy_(count)
        return context()

    def score_features(self, support, query, way, shot):
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint
        chunks = []
        pair_count = len(support) * len(query)
        for first in range(0, pair_count, self.options['pair_chunk_size']):
            indices = torch.arange(first, min(first + self.options['pair_chunk_size'], pair_count), device=query.device)
            if self.training and torch.is_grad_enabled() and self.options['checkpoint_relations']:
                value = checkpoint(self._relation_chunk, support, query, indices, use_reentrant=False,
                                   context_fn=lambda: (nullcontext(), self._recompute_context()))
            else:
                value = self._relation_chunk(support, query, indices)
            chunks.append(value)
        self._pair_scores = torch.cat(chunks).reshape(len(query), way, shot)
        return self._pair_scores.mean(2)

    def forward(self, support, query):
        way, shot = support.shape[:2]
        z = self.encode(torch.cat((support.flatten(0, 1), query)))
        return self.score_features(z[:way * shot], z[way * shot:], way, shot)


def episode_loss(model, scores, target):
    labels = F.one_hot(target, scores.shape[1]).float()[:, :, None].expand_as(model._pair_scores)
    return F.mse_loss(model._pair_scores, labels)


def training_task_batch(model, cache, tasks, shot, device):
    """Inter-task normalization of all feature-extractor inputs in a task batch."""
    way, total = tasks.shape[1:]
    x = torch.cat([cache.batch(indices.reshape(-1), device) for indices in tasks])
    encoded = model.encode(x)
    z = encoded.reshape(len(tasks), way, total, *encoded.shape[1:])
    losses, accuracies = [], []
    target = torch.arange(way, device=device).repeat_interleave(total - shot)
    for features in z:
        support = features[:, :shot].flatten(0, 1)
        query = features[:, shot:].flatten(0, 1)
        scores = model.score_features(support, query, way, shot)
        loss = episode_loss(model, scores, target)
        losses.append(loss)
        accuracies.append(float((scores.detach().argmax(1) == target).float().mean()))
    return torch.stack(losses).mean(), [float(loss.detach()) for loss in losses], accuracies



def prepare_episode(cache, indices, shot, device):
    way, total = indices.shape
    x = cache.batch(indices.reshape(-1), device).reshape(way, total, 2, 25, 64)
    support = x[:, :shot]
    query = x[:, shot:].flatten(0, 1)
    target = torch.arange(way, device=device).repeat_interleave(total - shot)
    return support, query, target


def evaluate(model, cache, plan, shot, device, directory=None, metadata=None):
    model.eval()
    losses, accuracies, logits, predictions = [], [], [], []
    for indices in plan:
        support, query, target = prepare_episode(cache, indices, shot, device)
        with torch.set_grad_enabled(METHOD == 'maml'):
            scores = episode_scores(model, support, query, target, False)
            loss = episode_loss(model, scores, target)
        if not torch.isfinite(scores).all() or not torch.isfinite(loss):
            raise RuntimeError('Nonfinite evaluation scores/loss')
        pred = scores.detach().argmax(1)
        losses.append(float(loss.detach()))
        accuracies.append(float((pred == target).float().mean()))
        if directory is not None:
            logits.append(scores.detach().cpu().numpy())
            predictions.append(pred.cpu().numpy())
    values = np.asarray(accuracies, dtype=np.float64)
    result = {'loss': float(np.mean(losses)), 'accuracy': float(values.mean()),
              'episode_accuracies': values.tolist(), 'episode_statistics': stats(values)}
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        scores = np.stack(logits)
        pred = np.stack(predictions)
        classes = cache.labels[plan[:, :, 0]]
        raw_classes = np.vectorize(cache.original.__getitem__, otypes=[np.int64])(classes)
        q = plan.shape[2] - shot
        truth = np.broadcast_to(np.repeat(np.arange(plan.shape[1]), q)[None], pred.shape).copy()
        true_raw = np.take_along_axis(raw_classes, truth, axis=1)
        pred_raw = np.take_along_axis(raw_classes, pred, axis=1)
        query_indices = plan[:, :, shot:].reshape(len(plan), -1)
        np.savez_compressed(directory / 'queries.npz', scores=scores, score_class_order_original_ids=raw_classes,
                            true_local=truth, predicted_local=pred, true_original_ids=true_raw,
                            predicted_original_ids=pred_raw, source_indices=query_indices,
                            episode_plan=plan, episode_accuracy=values)
        write_csv(directory / 'episode_accuracies.csv', [{'episode': i + 1, 'accuracy': float(v), 'accuracy_percent': float(v * 100)} for i, v in enumerate(values)])
        with (directory / 'query_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'query', 'source_index', 'true_original_id', 'predicted_original_id', 'correct'])
            for e in range(len(plan)):
                writer.writerows((e + 1, j + 1, int(query_indices[e, j]), int(true_raw[e, j]), int(pred_raw[e, j]), int(true_raw[e, j] == pred_raw[e, j])) for j in range(pred.shape[1]))
        original_ids = sorted(cache.original.values())
        cm = np.zeros((len(original_ids), len(original_ids)), dtype=np.int64)
        positions = {v: i for i, v in enumerate(original_ids)}
        np.add.at(cm, (np.vectorize(positions.__getitem__)(true_raw.ravel()), np.vectorize(positions.__getitem__)(pred_raw.ravel())), 1)
        rows = []
        for i, raw in enumerate(original_ids):
            n = int(cm[i].sum())
            rows.append({'original_id': raw, 'query_count': n, 'correct': int(cm[i, i]),
                         'accuracy_percent': float(cm[i, i] * 100 / n) if n else None})
        write_csv(directory / 'per_class.csv', rows)
        write_json(directory / 'per_class.json', rows)
        with (directory / 'confusion_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['true/predicted'] + original_ids)
            writer.writerows([raw] + cm[i].tolist() for i, raw in enumerate(original_ids))
        with (directory / 'confusion_percent.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['true/predicted'] + original_ids)
            writer.writerows([raw] + (list(cm[i] * 100 / cm[i].sum()) if cm[i].sum() else [''] * len(original_ids)) for i, raw in enumerate(original_ids))
        write_json(directory / 'test_summary.json', {**result, **(metadata or {}), 'provenance': PROVENANCE,
                   'score_type': 'sigmoid_relation_score' if METHOD in ('rn', 'ml_hgr_net') else 'classification_logit'})
    return result


def optimizer_for(model, options):
    return torch.optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('method') != METHOD:
        raise ValueError('Checkpoint belongs to another method')
    return checkpoint


def checkpoint_payload(model, optimizer, scheduler, cfg, cache, split, seed, shot, epoch, best, stale, history):
    return {'method': METHOD, 'provenance': PROVENANCE, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': None if scheduler is None else scheduler.state_dict(),
            'config': cfg, 'cache_signature': cache.signature, 'split': split, 'seed': seed, 'shot': shot,
            'epoch': epoch, 'best_validation': best, 'stale_epochs': stale, 'history': history,
            'source_sha256': file_hash(__file__), 'rng': rng_state(),
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                            'device': str(next(model.parameters()).device)}}


def train_one(cache, split, cfg, seed, shot, directory, device, resume=None, stop_after=None):
    directory.mkdir(parents=True, exist_ok=True)
    options = cfg['methods'][METHOD]
    tr = cfg['training']
    way, query = cfg['episode']['way'], cfg['episode']['query_per_class']
    seed_all(seed)
    model = Model(options, way, len(split['train_classes'])).to(device)
    print('%s | seed=%d shot=%d | parameters=%d | %s' % (TITLE, seed, shot, sum(p.numel() for p in model.parameters()), PROVENANCE['status']), flush=True)
    pretraining_state = None
    if resume is None:
        pretraining_state = pretrain_model(model, cache, split, cfg, seed, shot, directory, device)
    optimizer = optimizer_for(model, options)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options['step_size'], options['gamma']) if options['step_size'] else None
    if pretraining_state is not None:
        optimizer.load_state_dict(pretraining_state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(pretraining_state['scheduler'])
    history, best, stale = [], -1., 0
    start = options['pretraining_epochs'] + 1
    if resume is not None:
        saved = load_checkpoint(resume)
        if saved['cache_signature'] != cache.signature or saved['source_sha256'] != file_hash(__file__):
            raise ValueError('Cache or script differs from checkpoint; use its saved source and data')
        expected_config, actual_config = copy.deepcopy(saved['config']), copy.deepcopy(cfg)
        for item in (expected_config, actual_config):
            item['runtime'].pop('device', None)
            item['data'].pop('cache_dir', None)
        if expected_config != actual_config or saved['seed'] != seed or saved['shot'] != shot or saved['split'] != split:
            raise ValueError('Resume configuration/split differs from checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        history, best, stale, start = saved['history'], saved['best_validation'], saved['stale_epochs'], saved['epoch'] + 1
        restore_rng(saved['rng'])
    for epoch in range(start, tr['epochs'] + 1):
        if stale >= tr['patience']:
            break
        model.train()
        plan = episode_plan(cache.labels, split['train_classes'], way, shot, query, tr['episodes_per_epoch'], plan_seed(seed, shot, 1, epoch))
        train_losses, train_accs = [], []
        meta_batch = options['meta_batch_size']
        for start_task in range(0, len(plan), meta_batch):
            optimizer.zero_grad(set_to_none=True)
            tasks = plan[start_task:start_task + meta_batch]
            loss, task_losses, task_accuracies = training_task_batch(model, cache, tasks, shot, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            train_losses.extend(task_losses)
            train_accs.extend(task_accuracies)
            clip_gradients(model, options)
            optimizer.step()
        val_epoch = epoch if cfg['validation']['sampling'] == 'per_epoch' else 0
        val_plan = episode_plan(cache.labels, split['validation_classes'], way, shot, query, cfg['validation']['episodes'], plan_seed(seed, shot, 2, val_epoch))
        val = evaluate(model, cache, val_plan, shot, device)
        improved = val['accuracy'] > best
        best = max(best, val['accuracy'])
        stale = 0 if improved else stale + 1
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'train_accuracy': float(np.mean(train_accs)),
               'validation_loss': val['loss'], 'validation_accuracy': val['accuracy'], 'best_validation': best,
               'validation_improved': improved, 'learning_rate': optimizer.param_groups[0]['lr']}
        print('shot=%d seed=%d epoch=%03d loss=%.4f val_loss=%.4f train=%.2f%% val=%.2f%% best=%.2f%%' %
              (shot, seed, epoch, row['train_loss'], row['validation_loss'], row['train_accuracy'] * 100, val['accuracy'] * 100, best * 100), flush=True)
        test_cfg = cfg['testing']
        every = test_cfg['every_epochs']
        if epoch >= test_cfg['start_epoch'] and ((every == 0 and improved) or (every > 0 and epoch % every == 0)):
            monitor_plan = episode_plan(cache.labels, split['test_classes'], way, shot, query, test_cfg['periodic_episodes'], plan_seed(seed, shot, 5, epoch))
            monitor = evaluate(model, cache, monitor_plan, shot, device)
            print_stats('epoch=%03d %s test' % (epoch, 'best_validation' if every == 0 else 'interval'), monitor['episode_statistics'])
            row.update({'monitor_' + k: v for k, v in monitor['episode_statistics'].items()})
            np.savez_compressed(directory / ('monitor_%03d.npz' % epoch), plan=monitor_plan,
                                episode_accuracy=np.asarray(monitor['episode_accuracies']),
                                episode_statistics=json.dumps(monitor['episode_statistics']))
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        payload = checkpoint_payload(model, optimizer, scheduler, cfg, cache, split, seed, shot, epoch, best, stale, history)
        atomic_save(directory / 'last.pt', payload)
        if improved:
            atomic_save(directory / 'best.pt', payload)
        write_csv(directory / 'history.csv', history)
        if stop_after is not None and epoch >= stop_after:
            return None
        if stale >= tr['patience']:
            print('Early stopping: %d epochs without validation improvement.' % stale, flush=True)
    best_path = directory / 'best.pt'
    if not best_path.exists():
        raise RuntimeError('Missing best checkpoint')
    saved = load_checkpoint(best_path)
    model.load_state_dict(saved['model'])
    plan = episode_plan(cache.labels, split['test_classes'], way, shot, query, cfg['testing']['episodes'], plan_seed(seed, shot, 4))
    report = evaluate(model, cache, plan, shot, device, directory / 'test',
                      {'seed': seed, 'shot': shot, 'checkpoint_epoch': saved['epoch'], 'best_validation': saved['best_validation'],
                       'episode_seed': plan_seed(seed, shot, 4), 'cache_signature': cache.signature})
    print_stats('Final best-validation test', report['episode_statistics'])
    return {'method': METHOD, 'reference_status': PROVENANCE['status'], 'seed': seed, 'shot': shot, 'best_epoch': saved['epoch'],
            'best_validation': saved['best_validation'], **report['episode_statistics']}


def update_batch_reports(batch, row, cfg):
    path = batch / 'run_results.json'
    rows = read_json(path) if path.exists() else []
    rows = [r for r in rows if (r['seed'], r['shot']) != (row['seed'], row['shot'])] + [row]
    order = {(seed, shot): (i, j) for i, seed in enumerate(cfg['experiment']['seeds'])
             for j, shot in enumerate(cfg['experiment']['shots'])}
    rows.sort(key=lambda r: order[(r['seed'], r['shot'])])
    write_json(path, rows)
    write_csv(batch / 'runs.csv', rows)
    write_json(batch / 'summary.json', {str(k): stats([r['mean_accuracy'] for r in rows if r['shot'] == k], 'independent_training')
               for k in cfg['experiment']['shots'] if any(r['shot'] == k for r in rows)})


def validate_config(cfg):
    if cfg.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema_version')
    exp = cfg['experiment']
    if type(exp['runs']) is not int or exp['runs'] < 1 or exp['runs'] != len(exp['seeds']):
        raise ValueError('experiment.runs must equal the number of explicit seeds')
    if len(set(exp['seeds'])) != len(exp['seeds']) or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in exp['seeds']):
        raise ValueError('Seeds must be unique uint32 integers')
    if not exp['shots'] or len(set(exp['shots'])) != len(exp['shots']) or any(type(k) is not int or k < 1 for k in exp['shots']):
        raise ValueError('shots must be distinct positive integers')
    for v in [*cfg['data']['class_counts'], cfg['episode']['way'], cfg['episode']['query_per_class'],
              cfg['training']['epochs'], cfg['training']['patience'], cfg['training']['episodes_per_epoch'],
              cfg['validation']['episodes'], cfg['testing']['episodes'], cfg['testing']['periodic_episodes']]:
        if type(v) is not int or v < 1:
            raise ValueError('Class and episode counts must be positive integers')
    if len(cfg['data']['class_counts']) != 3 or min(cfg['data']['class_counts']) < cfg['episode']['way']:
        raise ValueError('Each split needs at least way classes')
    if cfg['validation']['sampling'] not in ('per_epoch', 'fixed'):
        raise ValueError('validation.sampling must be per_epoch or fixed')
    for key in ('every_epochs', 'start_epoch'):
        if type(cfg['testing'][key]) is not int or cfg['testing'][key] < 0:
            raise ValueError('Invalid testing.' + key)
    options = cfg['methods'][METHOD]
    if options['spatial_dims'] != 2 or options['model_input_shape'] != cfg['data']['input_shape']:
        raise ValueError('DTM/ATM cache training requires spatial_dims=2 and model_input_shape=[25,64]')
    if len(options['encoder_channels']) != 4 or len(options['relation_channels']) != 2:
        raise ValueError('ML-HGR-Net requires four feature convolutions and two relation convolutions')
    if options['time_axis'] != 0:
        raise ValueError('The first spatial axis of cached DTM/ATM maps is time')
    if any(type(options[name]) is not int or options[name] < 1 for name in
           ('meta_batch_size', 'pair_chunk_size', 'encoder_pool_stride', 'relation_pool_stride', 'relation_hidden')):
        raise ValueError('Invalid ML-HGR-Net batch/stride/hidden setting')
    if not math.isfinite(options['learning_rate']) or options['learning_rate'] <= 0 or options['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if options['pretraining_epochs'] >= cfg['training']['epochs']:
        raise ValueError('Pretraining must leave at least one episodic training epoch')
    if cfg['data']['input_shape'] != [25, 64] or cfg['data']['normalization'] != 'per_map_minmax' or cfg['data']['subject_filtering'] or cfg['data']['sampling_replacement']:
        raise ValueError('This comparison uses 25x64 per-map min-max inputs and class-only, no-replacement sampling')
    if type(options['step_size']) is not int or options['step_size'] < 0 or not 0 < options['gamma'] <= 1:
        raise ValueError('Invalid learning-rate scheduler')
    if METHOD == 'maml':
        for name in ('train_inner_steps', 'evaluation_inner_steps', 'meta_batch_size', 'channels'):
            if type(options[name]) is not int or options[name] < 1:
                raise ValueError('Invalid MAML ' + name)
        if not math.isfinite(options['inner_learning_rate']) or options['inner_learning_rate'] <= 0:
            raise ValueError('Invalid MAML inner learning rate')
    if METHOD in ('dn4', 'channel_dn4') and (type(options['neighbor_k']) is not int or options['neighbor_k'] < 1):
        raise ValueError('neighbor_k must be a positive integer')
    if METHOD == 'channel_dn4' and not 1 <= options['se_reduction'] <= options['channels']:
        raise ValueError('Invalid SE reduction')


def main(argv=None):
    parser = argparse.ArgumentParser(description=TITLE + ' on DTM/ATM radar maps')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--shots', type=int, nargs='+')
    parser.add_argument('--device')
    parser.add_argument('--check-plan', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--resume', type=Path, help='A last.pt checkpoint')
    parser.add_argument('--test-only', type=Path, help='A best.pt checkpoint')
    parser.add_argument('--test-episodes', type=int)
    parser.add_argument('--episode-seed', type=int)
    parser.add_argument('--stop-after', type=int, help='Save an epoch checkpoint and return')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(PROVENANCE, ensure_ascii=False, indent=2))
        return
    if args.resume and args.test_only:
        parser.error('Use either --resume or --test-only')
    checkpoint_path = args.resume or args.test_only
    saved = load_checkpoint(checkpoint_path) if checkpoint_path else None
    cfg = copy.deepcopy(saved['config']) if saved else read_json(args.config)
    if saved and (args.seeds or args.shots):
        parser.error('Saved checkpoint determines seed and shot')
    if args.seeds:
        cfg['experiment'].update(seeds=args.seeds, runs=len(args.seeds))
    if args.shots:
        cfg['experiment']['shots'] = args.shots
    if args.device:
        cfg['runtime']['device'] = args.device
    if args.cache_dir:
        cfg['data']['cache_dir'] = str(args.cache_dir.resolve())
    validate_config(cfg)
    device = configure_runtime(cfg)
    cache_path = Path(cfg['data']['cache_dir'])
    if not cache_path.is_absolute():
        cache_path = HERE / cache_path
    cache = RadarCache(cache_path)
    if saved and saved['cache_signature'] != cache.signature:
        raise ValueError('Checkpoint data fingerprint differs from this cache')
    root = (HERE / cfg['output']['results_dir']).resolve()
    if root != HERE / 'results' and HERE not in root.parents:
        raise ValueError('Outputs must remain inside comparison_methods')
    if args.test_only:
        if saved['source_sha256'] != file_hash(__file__):
            raise ValueError('Use the saved source snapshot for this checkpoint')
        seed_all(saved['seed'])
        shot = saved['shot']
        model = Model(cfg['methods'][METHOD], cfg['episode']['way'], len(saved['split']['train_classes'])).to(device)
        model.load_state_dict(saved['model'])
        episode_seed = args.episode_seed if args.episode_seed is not None else plan_seed(saved['seed'], shot, 4)
        count = args.test_episodes if args.test_episodes is not None else cfg['testing']['episodes']
        if count < 1 or episode_seed < 0:
            raise ValueError('Invalid test episode count/seed')
        plan = episode_plan(cache.labels, saved['split']['test_classes'], cfg['episode']['way'], shot, cfg['episode']['query_per_class'], count, episode_seed)
        out = Path(tempfile.mkdtemp(prefix='t_', dir=make_dir(root / METHOD / 'manual_test')))
        result = evaluate(model, cache, plan, shot, device, out, {'checkpoint': str(args.test_only.resolve()), 'checkpoint_epoch': saved['epoch'],
                           'seed': saved['seed'], 'shot': shot, 'episode_seed': episode_seed})
        print_stats('Independent test', result['episode_statistics'])
        print('Results:', out, flush=True)
        return
    plans = []
    for seed in cfg['experiment']['seeds']:
        split = split_classes(cache.labels, seed, cfg['data']['class_counts'])
        raw = {key: [cache.original[c] for c in split[key]] for key in SPLIT_KEYS}
        for key in SPLIT_KEYS:
            print('seed=%d %s original IDs: %s' % (seed, key, raw[key]), flush=True)
            for shot in cfg['experiment']['shots']:
                episode_plan(cache.labels, split[key], cfg['episode']['way'], shot, cfg['episode']['query_per_class'], 1, plan_seed(seed, shot, 4))
        plans.append((seed, split, raw))
    if args.check_plan:
        print('Class splits and episode feasibility verified; no training started.', flush=True)
        return
    if args.resume:
        directory = args.resume.resolve().parent
        row = train_one(cache, saved['split'], cfg, saved['seed'], saved['shot'], directory, device, args.resume, args.stop_after)
        batch = directory.parent.parent
        if row is not None and (batch / 'config.json').is_file():
            update_batch_reports(batch, row, cfg)
        return
    batch = Path(tempfile.mkdtemp(prefix='b_', dir=make_dir(root / METHOD)))
    write_json(batch / 'config.json', cfg)
    write_json(batch / 'sources.json', PROVENANCE)
    source = Path(__file__).read_text(encoding='utf-8')
    (batch / (METHOD + '.py')).write_text(source, encoding='utf-8')
    write_json(batch / 'data.json', {'path': str(cache.path), 'sha256': cache.signature, 'samples': len(cache.labels),
                                   'normalization': 'per_map_minmax', 'subject_filtering': False})
    for run, (seed, split, raw) in enumerate(plans, 1):
        run_dir = batch / ('r%02d_%d' % (run, seed))
        write_json(run_dir / 'split.json', {'internal_ids': split, 'original_ids': raw,
                   'sample_counts': {key: int(np.isin(cache.labels, split[key]).sum()) for key in SPLIT_KEYS}})
        for shot in cfg['experiment']['shots']:
            row = train_one(cache, split, cfg, seed, shot, run_dir / ('k%d' % shot), device, stop_after=args.stop_after)
            if row is not None:
                update_batch_reports(batch, row, cfg)
    print('Results:', batch, flush=True)


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    main()
