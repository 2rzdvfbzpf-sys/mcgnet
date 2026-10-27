"""Radar representations and episodic sampling for MCG-Net."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SPLIT_KEYS = ('train_classes', 'validation_classes', 'test_classes')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class RadarCache:
    def __init__(self, directory, normalization):
        self.directory = Path(directory).resolve()
        if normalization not in ('per_map_minmax', 'already_normalized'):
            raise ValueError('Unknown normalization mode')
        self.normalization = normalization
        self.labels = np.load(self.directory / 'labels.npy', allow_pickle=False)
        if self.labels.ndim != 1 or self.labels.dtype.kind not in 'iu' or len(self.labels) == 0:
            raise ValueError('labels.npy must be a one-dimensional integer array')
        self.arrays = {key: np.load(self.directory / (key + '.npy'), mmap_mode='r',
                                   allow_pickle=False) for key in ('dt', 'at')}
        self.ranges = {}
        for key, array in self.arrays.items():
            if array.shape != (len(self.labels), 25, 64) or array.dtype.kind != 'f':
                raise ValueError(key + '.npy must be floating point (samples,25,64)')
            low, high = float('inf'), float('-inf')
            for first in range(0, len(array), 512):
                part = array[first:first + 512]
                if not np.isfinite(part).all():
                    raise ValueError(key + ' contains non-finite values')
                low, high = min(low, float(part.min())), max(high, float(part.max()))
            self.ranges[key] = [low, high]
            if normalization == 'already_normalized' and not (0 <= low <= high <= 1):
                raise ValueError(key + ' is outside [0,1]; choose per_map_minmax explicitly')

    def describe(self, hashes=False):
        result = {'cache_dir': str(self.directory), 'samples': len(self.labels),
                  'classes': np.unique(self.labels).tolist(), 'shape_per_map': [25, 64],
                  'input_ranges': self.ranges, 'normalization': self.normalization,
                  'subject_filtering': False}
        if hashes:
            result['sha256'] = {name: file_hash(self.directory / (name + '.npy'))
                                for name in ('labels', 'dt', 'at')}
        return result

    def batch(self, indices, device):
        tensors = []
        for key in ('dt', 'at'):
            data = np.array(self.arrays[key][indices], dtype=np.float32, copy=True)
            if self.normalization == 'per_map_minmax':
                low = data.min(axis=(-2, -1), keepdims=True)
                span = data.max(axis=(-2, -1), keepdims=True) - low
                data = np.divide(data - low, span, out=np.zeros_like(data), where=span > 0)
            tensors.append(torch.from_numpy(data[:, None]).to(device))
        return tuple(tensors)


def class_split(labels, seed, manifest=None, *, counts, sample_classes=None):
    """Generate or validate mutually disjoint class groups for one experiment."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or not len(labels):
        raise ValueError('Class split requires nonempty one-dimensional integer labels')
    if seed < 0:
        raise ValueError('Class split seed must be nonnegative')
    present = set(int(x) for x in np.unique(labels))
    if manifest is not None:
        if sample_classes is not None:
            raise ValueError('Use either a frozen split manifest or --sample-classes, not both')
        payload = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
        split = {key: payload[key] for key in SPLIT_KEYS}
        selection_policy = payload.get('class_selection_policy', 'explicit_manifest')
    else:
        if sample_classes is not None and (type(sample_classes) is not int
                or sample_classes != sum(counts) or len(present) < sample_classes):
            raise ValueError('--sample-classes must equal the split total and not exceed available classes')
        if sample_classes is None and len(present) != sum(counts):
            raise ValueError('Cache has %d classes, but manuscript split needs %s. Supply --split-json '
                             'with the actual class IDs, or explicitly request --sample-classes 50 '
                             'for a new seeded subset.' % (len(present), counts))
        classes = np.array(sorted(present), dtype=np.int64)
        np.random.RandomState(seed).shuffle(classes)
        if sample_classes is not None:
            classes = classes[:sample_classes]
        selection_policy = ('seeded_random_subset_then_split' if len(classes) < len(present)
                            else 'all_cache_classes_then_seeded_split')
        a, b = counts[0], counts[0] + counts[1]
        split = dict(zip(SPLIT_KEYS, (classes[:a].tolist(), classes[a:b].tolist(), classes[b:].tolist())))
    seen = set()
    for key, count in zip(SPLIT_KEYS, counts):
        values = split[key]
        if (not isinstance(values, list) or len(values) != count
                or any(type(x) is not int for x in values) or len(set(values)) != len(values)):
            raise ValueError('%s must contain exactly %d distinct integer cache labels' % (key, count))
        if seen.intersection(values) or not set(values).issubset(present):
            raise ValueError('Class groups overlap or contain absent cache labels')
        seen.update(values)
    return {**split, 'class_counts': list(counts),
            'class_split_seed': seed if manifest is None else payload.get('class_split_seed'),
            'class_selection_policy': selection_policy,
            'selected_cache_classes': sorted(seen),
            'unused_cache_classes': sorted(present - seen),
            'subject_filtering': False, 'support_query_sample_overlap': False}


def episode_plan(labels, classes, way, shot, query, count, seed):
    if min(way, shot, query, count) < 1 or len(classes) < way:
        raise ValueError('Invalid episode counts or too few classes')
    pools = {int(c): np.flatnonzero(labels == c) for c in classes}
    for cls, pool in pools.items():
        if len(pool) < shot + query:
            raise ValueError('Class %s needs %d distinct samples, has %d; replacement is forbidden'
                             % (cls, shot + query, len(pool)))
    rng = np.random.default_rng(seed)
    return np.asarray([[rng.choice(pools[int(c)], shot + query, replace=False)
                        for c in rng.choice(classes, way, replace=False)] for _ in range(count)],
                      dtype=np.int64)


def plan_seed(base, run, shot, stage, epoch=0):
    # Derive episode seeds from the run, shot, stage and epoch.
    return int(np.random.SeedSequence([base, run, shot, stage, epoch]).generate_state(1)[0])
