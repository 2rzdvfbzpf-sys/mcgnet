"""MCG-Net experiment configuration."""
from dataclasses import fields
from pathlib import Path

from data.radar_cache import read_json
from models import ModelConfig


PROJECT = Path(__file__).resolve().parent


def _keys(value, expected, location):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError('%s keys differ: missing=%s, unknown=%s' %
                         (location, sorted(set(expected) - actual), sorted(actual - set(expected))))


def training_settings(config):
    training = config['training']
    optimizer, scheduler = training['optimizer'], training['scheduler']
    return {
        'epochs': training['epochs'], 'patience': training['patience'],
        'shots': config['experiment']['shots'], 'train_episodes': training['episodes_per_epoch'],
        'val_episodes': config['validation']['episodes'],
        'validation_plan': config['validation']['sampling'],
        'test_episodes': config['testing']['episodes'],
        'test_every_epochs': config['testing']['every_epochs'],
        'monitor_start_epoch': config['testing']['start_epoch'],
        'monitor_test_episodes': config['testing']['periodic_episodes'],
        'way': config['episode']['way'], 'query_per_class': config['episode']['query_per_class'],
        'learning_rate': optimizer['learning_rate'], 'weight_decay': optimizer['weight_decay'],
        'adam_betas': optimizer['betas'], 'adam_eps': optimizer['eps'],
        'scheduler_step_size': scheduler['step_size'], 'scheduler_gamma': scheduler['gamma'],
    }


def validate_config(config):
    _keys(config, ('schema_version', 'experiment', 'data', 'model', 'episode', 'training',
                   'validation', 'testing', 'runtime', 'output'), 'configuration')
    if type(config['schema_version']) is not int or config['schema_version'] != 2:
        raise ValueError('Unsupported configuration schema_version')
    sections = {
        'experiment': ('runs', 'seeds', 'shots'),
        'data': ('cache_dir', 'num_classes', 'class_counts', 'input_shape', 'normalization',
                 'subject_filtering', 'sampling_replacement'),
        'model': tuple(field.name for field in fields(ModelConfig)),
        'episode': ('way', 'query_per_class'),
        'training': ('epochs', 'patience', 'episodes_per_epoch', 'optimizer', 'scheduler'),
        'validation': ('episodes', 'sampling', 'selection_metric'),
        'testing': ('episodes', 'every_epochs', 'start_epoch', 'periodic_episodes'),
        'runtime': ('device', 'cpu_threads', 'deterministic_algorithms', 'cublas_workspace_config'),
        'output': ('results_dir', 'checkpoints_dir'),
    }
    for name, keys in sections.items():
        _keys(config[name], keys, name)
    _keys(config['training']['optimizer'], ('name', 'learning_rate', 'weight_decay', 'betas', 'eps'),
          'training.optimizer')
    _keys(config['training']['scheduler'], ('name', 'step_size', 'gamma'), 'training.scheduler')
    if config['training']['optimizer']['name'] != 'Adam':
        raise ValueError('training.optimizer.name must be Adam')
    if config['training']['scheduler']['name'] != 'StepLR':
        raise ValueError('training.scheduler.name must be StepLR')
    exp, data = config['experiment'], config['data']
    if type(exp['runs']) is not int or not 1 <= exp['runs'] <= 2**32:
        raise ValueError('experiment.runs must be a positive uint32-sized count')
    seeds = exp['seeds']
    if not isinstance(seeds, list) or len(seeds) != exp['runs']:
        raise ValueError('experiment.seeds must be a list with exactly experiment.runs entries')
    if any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds):
        raise ValueError('experiment.seeds entries must be integers in 0..4294967295')
    if len(set(seeds)) != len(seeds):
        raise ValueError('experiment.seeds must contain distinct seeds')
    if type(data['num_classes']) is not int or data['num_classes'] != 50:
        raise ValueError('data.num_classes must be 50')
    if data['input_shape'] != [25, 64]:
        raise ValueError('data.input_shape must be [25,64]')
    counts = data['class_counts']
    if (not isinstance(counts, list) or len(counts) != 3 or sum(counts) != data['num_classes']
            or any(type(value) is not int or value < 1 for value in counts)):
        raise ValueError('data.class_counts must contain three positive integers summing to 50')
    if data['subject_filtering'] is not False or data['sampling_replacement'] is not False:
        raise ValueError('Subject filtering and replacement sampling are not implemented')
    if data['normalization'] not in ('per_map_minmax', 'already_normalized'):
        raise ValueError('data.normalization must be per_map_minmax or already_normalized')
    if config['validation']['selection_metric'] != 'accuracy':
        raise ValueError('validation.selection_metric must be accuracy')
    for section, name in (('data', 'cache_dir'), ('output', 'results_dir'), ('output', 'checkpoints_dir')):
        value = config[section][name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(section + '.' + name + ' must be a nonempty path')
    runtime = config['runtime']
    if type(runtime['deterministic_algorithms']) is not bool:
        raise ValueError('runtime.deterministic_algorithms must be true or false')
    if runtime['cublas_workspace_config'] not in (':4096:8', ':16:8'):
        raise ValueError('runtime.cublas_workspace_config must be :4096:8 or :16:8')
    if not isinstance(runtime['device'], str) or not runtime['device'].strip():
        raise ValueError('runtime.device must be auto, cpu or a torch device string')
    if runtime['cpu_threads'] is not None and (type(runtime['cpu_threads']) is not int
                                               or runtime['cpu_threads'] < 1):
        raise ValueError('runtime.cpu_threads must be null or a positive integer')
    ModelConfig(**config['model'])
    from training.engine import TrainingConfig
    tc = TrainingConfig(**training_settings(config), runs=1,
                        seed=exp['seeds'][0], evaluation_seed=exp['seeds'][0])
    if min(counts) < tc.way:
        raise ValueError('Each class split must contain at least episode.way classes')
    return config


def load_config(path):
    return validate_config(read_json(path))


def resolve_paths(config):
    """Resolve experiment paths relative to the project containing train.py."""
    for section, name in (('data', 'cache_dir'), ('output', 'results_dir'), ('output', 'checkpoints_dir')):
        value = Path(config[section][name]).expanduser()
        config[section][name] = str((value if value.is_absolute() else PROJECT / value).resolve())
    return config
