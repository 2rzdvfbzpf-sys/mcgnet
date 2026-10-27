"""DSMFT-Net: episodic radar gesture classification."""
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
METHOD = 'dsmft_net'
TITLE = 'DSMFT-Net'
PROVENANCE = {'method': 'DSMFT-Net', 'paper': 'https://doi.org/10.1007/s40747-025-01848-2', 'status': 'paper_based_two_view_adaptation', 'input': 'DTM/ATM; 25x64; independent per-map min-max [0,1]', 'protocol': 'seeded 30/10/10 class split; no subject filtering; no within-episode sample replacement', 'scope': 'Episodic few-shot radar classification with DTM/ATM representations', 'mechanism': 'Shared residual-CBAM encoder; FCM cluster prototypes; intra-view cluster CE; cross-view prototype entropy minimization; target classifier mutual information maximization; source supervised CE', 'adaptation': 'DTM and ATM treated as two feature views, never labeled simulated/measured domains; training-class-only pretraining, followed by inductive support-prototype classification on unseen classes; concatenated normalized view embeddings', 'specified_implementation_choices': ['Figure7: stem Conv64 stride2 and max-pool2, two basic residual blocks per 64/128/256 stage, one CBAM after each stage; later stages preserve spatial size per prose; channel shortcuts use 1x1 projections', 'Global average -> FC256 -> classifier256/100/train-class-count; classifier used only in pretraining', 'FCM uses standard normalized inverse-distance memberships; printed Eq12 membership is not normalized', 'Temperatures0.1, fuzzy exponent2, iteration limit50/tolerance1e-5 and equal loss weights are configurable choices, not claimed paper constants', 'First20 of total300 epochs are pretraining;100 batches64 per pretraining epoch; remaining epochs episodic; optimizer and StepLR continue across phases', 'No validation/test data or labels used for clustering or pretraining; no target-query adaptation at evaluation', 'Original simulator/domain experiment and fixed six-class classifier are not reproduced by the DTM/ATM unseen-class adaptation'], 'paper_html_sha256': '6aa5278b2311088d76d5d8d7cf86b2cb408a6e0f5dbc84666fef3c2d144fff85', 'inspected_figures_sha256': {'5': 'e02558f075350f2d5f86372ff80c425ef58c3f30d603f9f7f5227917a6fe2857', '7': '51c54c01d583f15abf57ef1dba77b2319b0305dd6dfac0a4b33efb10a97d706d'}}
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


class CBAM(nn.Module):
    """Channel and spatial attention for radar spectrogram features."""
    def __init__(self, channels, reduction):
        super().__init__()
        self.channel = nn.Sequential(nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False), nn.ReLU(),
                                     nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        weight = torch.sigmoid(self.channel(x.mean((2, 3), keepdim=True)) + self.channel(x.amax((2, 3), keepdim=True)))
        x = x * weight
        weight = torch.sigmoid(self.spatial(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)))
        return x * weight


class BasicResidual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                    nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.branch(x) + self.skip(x))


class Model(nn.Module):
    """DSMFT prototype-contrast mechanism adapted to paired DTM/ATM views."""
    def __init__(self, options, way, train_classes):
        super().__init__()
        self.options = options
        widths = options['encoder_channels']
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(), nn.MaxPool2d(2))
        stages = []
        previous = widths[0]
        for width in widths:
            stages.append(nn.Sequential(BasicResidual(previous, width), BasicResidual(width, width), CBAM(width, options['cbam_reduction'])))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.projection = nn.Linear(widths[-1], options['embedding_dim'])
        self.classifier = nn.Sequential(nn.Linear(options['embedding_dim'], options['classifier_hidden']), nn.ReLU(),
                                         nn.Linear(options['classifier_hidden'], train_classes))

    def encode(self, x):
        return self.projection(self.stages(self.stem(x)).mean((2, 3)))

    def paired(self, x):
        n = len(x)
        z = self.encode(torch.cat((x[:, :1], x[:, 1:]), 0))
        return z[:n], z[n:]

    def forward(self, support, query):
        way, shot = support.shape[:2]
        za, zb = self.paired(torch.cat((support.flatten(0, 1), query)))
        z = F.normalize(torch.cat((F.normalize(za, dim=1), F.normalize(zb, dim=1)), 1), dim=1)
        prototypes = F.normalize(z[:way * shot].reshape(way, shot, -1).mean(1), dim=1)
        return (z[way * shot:] @ prototypes.T) / self.options['episodic_temperature']


def fuzzy_c_means(features, clusters, seed, fuzziness, max_iterations, tolerance):
    """FCM with row-normalized memberships and deterministic initialization."""
    x = np.asarray(features, dtype=np.float64)
    if len(x) < clusters or fuzziness <= 1:
        raise ValueError('FCM requires n>=clusters and fuzziness>1')
    rng = np.random.default_rng(seed)
    u = rng.random((len(x), clusters))
    u /= u.sum(1, keepdims=True)
    for _ in range(max_iterations):
        weights = u ** fuzziness
        centers = (weights.T @ x) / weights.sum(0)[:, None].clip(1e-15)
        distances = np.maximum((x * x).sum(1)[:, None] + (centers * centers).sum(1)[None] - 2 * (x @ centers.T), 1e-15)
        inv = distances ** (-1. / (fuzziness - 1.))
        updated = inv / inv.sum(1, keepdims=True)
        delta = np.max(np.abs(updated - u))
        u = updated
        if delta < tolerance:
            break
    weights = u ** fuzziness
    centers = (weights.T @ x) / weights.sum(0)[:, None].clip(1e-15)
    return centers.astype(np.float32), u.argmax(1).astype(np.int64), float((u ** 2).sum() / len(x))


def contrastive_losses(model, a, b, centers_a, centers_b, labels_a, labels_b):
    opt = model.options
    raw_b = b
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    ca, cb = F.normalize(centers_a, dim=1), F.normalize(centers_b, dim=1)
    ipc = F.cross_entropy((a @ ca.T) / opt['intra_temperature'], labels_a) + F.cross_entropy((b @ cb.T) / opt['intra_temperature'], labels_b)
    def entropy(p):
        return -(p * p.clamp_min(1e-12).log()).sum(-1)
    pab = ((a @ cb.T) / opt['cross_temperature']).softmax(1)
    pba = ((b @ ca.T) / opt['cross_temperature']).softmax(1)
    cpc = entropy(pab).mean() + entropy(pba).mean()
    # Target-view classifier predictions use no ATM ground-truth labels.
    predicted = model.classifier(raw_b).softmax(1)
    negative_mi = entropy(predicted).mean() - entropy(predicted.mean(0))
    return ipc, cpc, negative_mi


def pretrain_model(model, cache, split, cfg, seed, shot, directory, device):
    opt = cfg['methods'][METHOD]
    budget = opt['pretraining_epochs']
    if budget < 1:
        raise ValueError('DSMFT adaptation requires prototype-contrast pretraining')
    indices = np.flatnonzero(np.isin(cache.labels, split['train_classes']))
    class_positions = {cls: i for i, cls in enumerate(split['train_classes'])}
    targets = np.asarray([class_positions[int(cache.labels[i])] for i in indices], dtype=np.int64)
    optimizer = optimizer_for(model, opt)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt['step_size'], opt['gamma'])
    history = []
    size = opt['pretraining_batch_size']
    for epoch in range(1, budget + 1):
        model.eval()
        features_a, features_b = [], []
        with torch.no_grad():
            for start in range(0, len(indices), size):
                a, b = model.paired(cache.batch(indices[start:start + size], device))
                features_a.append(a.cpu().numpy())
                features_b.append(b.cpu().numpy())
        fa, fb = np.concatenate(features_a), np.concatenate(features_b)
        centers_a, labels_a, pc_a = fuzzy_c_means(fa, len(split['train_classes']), plan_seed(seed, shot, 6, epoch), opt['fcm_fuzziness'], opt['fcm_iterations'], opt['fcm_tolerance'])
        centers_b, labels_b, pc_b = fuzzy_c_means(fb, len(split['train_classes']), plan_seed(seed, shot, 7, epoch), opt['fcm_fuzziness'], opt['fcm_iterations'], opt['fcm_tolerance'])
        ca, cb = torch.from_numpy(centers_a).to(device), torch.from_numpy(centers_b).to(device)
        rng = np.random.default_rng(plan_seed(seed, shot, 8, epoch))
        model.train()
        records = []
        for _ in range(opt['pretraining_batches_per_epoch']):
            chosen = rng.choice(len(indices), min(size, len(indices)), replace=False)
            optimizer.zero_grad(set_to_none=True)
            a, b = model.paired(cache.batch(indices[chosen], device))
            ya = torch.from_numpy(labels_a[chosen]).to(device)
            yb = torch.from_numpy(labels_b[chosen]).to(device)
            ipc, cpc, negative_mi = contrastive_losses(model, a, b, ca, cb, ya, yb)
            ys = torch.from_numpy(targets[chosen]).to(device)
            supervised = F.cross_entropy(model.classifier(a), ys)
            loss = opt['ipc_weight'] * ipc + opt['cpc_weight'] * cpc + opt['mi_weight'] * negative_mi + opt['source_ce_weight'] * supervised
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite DSMFT pretraining loss')
            loss.backward()
            optimizer.step()
            records.append([float(v.detach()) for v in (loss, ipc, cpc, negative_mi, supervised)])
        means = np.mean(records, axis=0)
        history.append(dict(zip(('epoch', 'loss', 'ipc', 'cpc', 'negative_mi', 'source_ce', 'fcm_pc_dt', 'fcm_pc_at'),
                                [epoch, *means.tolist(), pc_a, pc_b])))
        write_csv(directory / 'pretraining.csv', history)
        np.savez_compressed(directory / 'clusters.npz', train_source_indices=indices, dt_centers=centers_a, at_centers=centers_b,
                            dt_cluster_labels=labels_a, at_cluster_labels=labels_b)
        scheduler.step()
        print('DSMFT pretrain %03d/%03d loss=%.4f IPC=%.4f CPC=%.4f -MI=%.4f source_CE=%.4f' %
              (epoch, budget, *means.tolist()), flush=True)
    atomic_save(directory / 'pretrained.pt', {'method': METHOD, 'model': model.state_dict(), 'config': cfg,
                'train_source_indices': indices, 'seed': seed, 'shot': shot, 'history': history, 'provenance': PROVENANCE,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
    return {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}



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
            for indices in tasks:
                support, queries, target = prepare_episode(cache, indices, shot, device)
                scores = episode_scores(model, support, queries, target, True)
                loss = episode_loss(model, scores, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                (loss / len(tasks)).backward()
                train_losses.append(float(loss.detach()))
                train_accs.append(float((scores.detach().argmax(1) == target).float().mean()))
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
