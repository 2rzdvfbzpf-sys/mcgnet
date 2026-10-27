"""Ablation experiment configuration and source provenance."""
import _bootstrap
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from models import ModelConfig
from networks import VARIANTS

HERE = _bootstrap.HERE
LOCAL_SOURCES = ('_bootstrap.py', 'networks.py', 'settings.py', 'reporting.py',
                 'runner.py', 'train.py', 'test.py')
EXTERNAL_SOURCES = tuple(name for folder in ('models', 'data', 'analysis')
                         for name in sorted(str(p.relative_to(_bootstrap.EXTERNAL_ROOT)).replace('\\', '/')
                                            for p in (_bootstrap.EXTERNAL_ROOT / folder).glob('*.py')))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def source_hashes():
    return {**{name: file_hash(HERE / name) for name in LOCAL_SOURCES},
            **{'external/' + name: file_hash(_bootstrap.EXTERNAL_ROOT / name) for name in EXTERNAL_SOURCES}}


def snapshot_sources(directory, config):
    directory = Path(directory)
    for name in LOCAL_SOURCES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    for name in EXTERNAL_SOURCES:
        target = directory / 'external' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_bootstrap.EXTERNAL_ROOT / name, target)
    write_json(directory / 'config.json', config)


def own_path(value, root=HERE):
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise ValueError('Output paths must remain inside the ablation directory')
    return resolved


def positive(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(name + ' must be a ' + ('nonnegative' if allow_zero else 'positive') + ' integer')


def validate_config(config, check_output_paths=True):
    required = {'schema_version', 'experiment', 'data', 'model', 'ablation', 'episode',
                'training', 'validation', 'testing', 'runtime', 'output'}
    if (not isinstance(config, dict) or set(config) != required
            or type(config['schema_version']) is not int or config['schema_version'] != 1):
        raise ValueError('Invalid ablation configuration schema')
    exp = config['experiment']
    positive(exp['runs'], 'experiment.runs')
    if (not isinstance(exp['seeds'], list) or len(exp['seeds']) != exp['runs']
            or len(set(exp['seeds'])) != len(exp['seeds'])
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in exp['seeds'])):
        raise ValueError('seeds must be distinct uint32 integers; count must equal runs')
    for values, name in ((exp['shots'], 'shots'), (config['ablation']['variants'], 'variants')):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(name + ' must be a nonempty distinct list')
    for shot in exp['shots']:
        positive(shot, 'shot')
    if any(value not in VARIANTS for value in config['ablation']['variants']):
        raise ValueError('Unknown ablation variant')
    data = config['data']
    if (data['num_classes'] != 50 or data['input_shape'] != [25, 64]
            or data['subject_filtering'] is not False or data['sampling_replacement'] is not False
            or data['normalization'] not in ('per_map_minmax', 'already_normalized')):
        raise ValueError('Expected 50-class aligned DTM/ATM, class-only, no-replacement protocol')
    counts = data['class_counts']
    if not isinstance(counts, list) or len(counts) != 3 or sum(counts) != 50:
        raise ValueError('Three class counts must sum to 50')
    for number in counts:
        positive(number, 'class count')
    ModelConfig(**config['model'])
    if config['model']['share_modal_encoders']:
        raise ValueError('Ablations use independent modality encoders')
    widths = config['ablation']['resnet12_channels']
    if not isinstance(widths, list) or len(widths) != 4:
        raise ValueError('ResNet-12 requires four stage widths')
    for width in widths:
        positive(width, 'resnet12 width')
    logits = config['ablation']['fusion_weight_logits']
    if len(logits) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
        raise ValueError('fusion_weight_logits requires two finite scalars')
    cca = config['ablation']['cca']
    candidates = cca['dimensions']
    if (not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates)
            or any(type(x) is not int or not 1 <= x <= 112 for x in candidates)):
        raise ValueError('CCA dimensions must be distinct integers in 1..112')
    if type(cca['regularization']) not in (int, float) or not 0 < cca['regularization'] < float('inf'):
        raise ValueError('CCA regularization must be finite and positive')
    positive(cca['feature_batch_size'], 'CCA feature batch size')
    positive(config['episode']['way'], 'way')
    positive(config['episode']['query_per_class'], 'query_per_class')
    if min(counts) < config['episode']['way']:
        raise ValueError('Every split must contain at least way classes')
    for key in ('epochs', 'episodes_per_epoch'):
        positive(config['training'][key], 'training.' + key)
    positive(config['training']['patience'], 'patience', True)
    optimizer, scheduler = config['training']['optimizer'], config['training']['scheduler']
    if optimizer['name'] != 'Adam' or scheduler['name'] != 'StepLR':
        raise ValueError('Expected Adam and StepLR')
    for value, name in ((optimizer['learning_rate'], 'learning_rate'),
                        (optimizer['eps'], 'optimizer.eps'), (scheduler['gamma'], 'scheduler.gamma')):
        if type(value) not in (int, float) or not 0 < value < float('inf'):
            raise ValueError(name + ' must be finite and positive')
    if (type(optimizer['weight_decay']) not in (int, float) or not 0 <= optimizer['weight_decay'] < float('inf')
            or len(optimizer['betas']) != 2 or any(not 0 <= x < 1 for x in optimizer['betas'])):
        raise ValueError('Invalid Adam weight decay/betas')
    positive(scheduler['step_size'], 'scheduler.step_size')
    positive(config['validation']['episodes'], 'validation.episodes')
    if config['validation']['sampling'] not in ('fixed', 'per_epoch') or config['validation']['selection_metric'] != 'accuracy':
        raise ValueError('Validation requires fixed/per_epoch tasks and accuracy selection')
    for key in ('episodes', 'start_epoch', 'periodic_episodes'):
        positive(config['testing'][key], 'testing.' + key)
    positive(config['testing']['every_epochs'], 'testing.every_epochs', True)
    rt = config['runtime']
    if type(rt['deterministic_algorithms']) is not bool or rt['cublas_workspace_config'] not in (':4096:8', ':16:8'):
        raise ValueError('Invalid deterministic runtime settings')
    if rt['cpu_threads'] is not None:
        positive(rt['cpu_threads'], 'cpu_threads')
    for value in config['output'].values():
        if not isinstance(value, str) or not value:
            raise ValueError('Output paths must be nonempty strings')
        if check_output_paths:
            own_path(value)
    return copy.deepcopy(config)
