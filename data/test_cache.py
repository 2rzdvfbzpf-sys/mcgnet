"""Test-set cache export and validation for MCG-Net."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from data.radar_cache import RadarCache, file_hash, read_json, write_json


def label_mapping(data):
    mapping = read_json(data.directory / 'label_mapping.json')
    reverse = {int(label): int(raw) for raw, label in mapping.items()}
    present = set(map(int, np.unique(data.labels)))
    if min(present) < 0 or len(reverse) != len(mapping) or set(reverse) != present:
        raise ValueError('label_mapping.json must map original IDs one-to-one to all cache labels')
    return reverse


def export_test_cache(data, split, directory, source_description=None):
    """Save all samples of the test classes in their source-cache row order."""
    directory = Path(directory)
    reverse = label_mapping(data)
    classes = list(split['test_classes'])
    indices = np.flatnonzero(np.isin(data.labels, classes)).astype(np.int64)
    description = source_description or data.describe(hashes=True)
    identity = {'class_split_seed': split['class_split_seed'], 'class_order': classes,
                'original_id_order': [reverse[label] for label in classes],
                'normalization': data.normalization, 'source_cache_sha256': description['sha256'],
                'source_mapping_sha256': file_hash(data.directory / 'label_mapping.json'),
                'sample_count': len(indices)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if directory.exists():
        _, metadata = load_test_cache(directory, data.normalization)
        if metadata.get('export_identity_sha256') != digest:
            raise ValueError('Existing test cache does not match this class split and source cache')
        return metadata
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.test_cache_', dir=str(directory.parent)))
    try:
        for name, source in data.arrays.items():
            output = np.lib.format.open_memmap(temporary / (name + '.npy'), mode='w+',
                                               dtype=source.dtype, shape=(len(indices), 25, 64))
            for first in range(0, len(indices), 512):
                output[first:first + 512] = source[indices[first:first + 512]]
            output.flush()
            del output
        np.save(temporary / 'labels.npy', data.labels[indices], allow_pickle=False)
        np.save(temporary / 'source_indices.npy', indices, allow_pickle=False)
        write_json(temporary / 'label_mapping.json', {str(reverse[label]): label for label in classes})
        files = ('dt.npy', 'at.npy', 'labels.npy', 'label_mapping.json', 'source_indices.npy')
        metadata = {'format_version': 1, 'dataset_type': 'mcg_net_test_cache',
                    **identity, 'export_identity_sha256': digest,
                    'storage': 'source_cache_values_before_loader_normalization',
                    'sha256': {name: file_hash(temporary / name) for name in files}}
        write_json(temporary / 'dataset.json', metadata)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            if temporary.resolve().parent != directory.parent.resolve() or not temporary.name.startswith('.test_cache_'):
                raise ValueError('Unexpected temporary cache path')
            shutil.rmtree(temporary)
    return metadata


def load_test_cache(directory, normalization):
    directory = Path(directory).resolve()
    metadata = read_json(directory / 'dataset.json') if (directory / 'dataset.json').is_file() else {}
    if metadata:
        if metadata.get('format_version') != 1 or metadata.get('dataset_type') != 'mcg_net_test_cache':
            raise ValueError('Unsupported test-cache metadata')
        if metadata['normalization'] != normalization:
            raise ValueError('Test-cache normalization differs from checkpoint normalization')
        required = {'dt.npy', 'at.npy', 'labels.npy', 'label_mapping.json', 'source_indices.npy'}
        if set(metadata['sha256']) != required:
            raise ValueError('Incomplete test-cache file hashes')
        for name in sorted(required):
            if file_hash(directory / name) != metadata['sha256'][name]:
                raise ValueError('Test-cache file checksum differs: ' + name)
    data = RadarCache(directory, normalization)
    reverse = label_mapping(data)
    if metadata:
        classes = metadata['class_order']
        if (len(classes) != len(set(classes)) or set(classes) != set(reverse)
                or [reverse[label] for label in classes] != metadata['original_id_order']
                or metadata['sample_count'] != len(data.labels)):
            raise ValueError('Test-cache class metadata differs from labels/mapping')
        indices = np.load(directory / 'source_indices.npy', allow_pickle=False)
        if (indices.shape != data.labels.shape or indices.dtype.kind not in 'iu'
                or (indices < 0).any() or (np.diff(indices.astype(np.int64)) <= 0).any()):
            raise ValueError('Invalid source-cache row indices')
        data.source_indices = indices
    return data, metadata
