"""MCG-Net checkpoint evaluation on a specified radar cache."""
import csv
from pathlib import Path
import tempfile

import numpy as np
import torch

from analysis.reports import save_query_artifacts
from data.radar_cache import episode_plan, file_hash, plan_seed, read_json, write_json
from data.test_cache import label_mapping, load_test_cache
from models import MCGNet, ModelConfig
from training import engine


def inference_settings(data, split, model_config, training_config, shot, run, spec):
    reverse = label_mapping(data)
    return {'format_version': 1, 'model': model_config.to_dict(),
            'shot': shot, 'way': training_config.way,
            'query_per_class': training_config.query_per_class,
            'test_episodes': training_config.test_episodes,
            'training_seed': training_config.seed,
            'episode_seed': plan_seed(training_config.evaluation_seed, run, shot, 4),
            'class_order_original_ids': [reverse[label] for label in split['test_classes']],
            'normalization': data.normalization,
            'reproducibility': spec['reproducibility'],
            'model_source_sha256': {name: digest for name, digest in spec['source_sha256'].items()
                                    if name.startswith('models/')},
            'source_cache_sha256': spec['data']['sha256']}


def checkpoint_settings(checkpoint, experiment_json=None, checkpoint_path=None):
    settings = checkpoint.get('inference')
    if settings is not None:
        if settings.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint inference metadata')
        return settings
    if experiment_json is None:
        raise ValueError('Checkpoint has no inference metadata; supply --experiment-json')
    spec = read_json(experiment_json)
    if engine.signature(spec) != checkpoint['spec_sha256']:
        raise ValueError('Checkpoint and experiment.json do not match')
    name = Path(checkpoint_path).parent.name
    try:
        shot_text, run_text = name.split('_r')
        shot, run = int(shot_text[1:]), int(run_text) - 1
        if not shot_text.startswith('s') or shot not in spec['training']['shots'] or run < 0:
            raise ValueError()
    except ValueError as error:
        raise ValueError('Checkpoint parent directory must be s<shot>_r<run>') from error
    data_info, split, tc = spec['data'], spec['split'], spec['training']
    originals = split.get('raw_gesture_ids', {}).get('test_classes')
    if originals is None:
        raise ValueError('experiment.json needs raw_gesture_ids.test_classes')
    return {'format_version': 1, 'model': spec['model'], 'shot': shot, 'way': tc['way'],
            'query_per_class': tc['query_per_class'], 'test_episodes': tc['test_episodes'],
            'training_seed': tc['seed'], 'episode_seed': plan_seed(tc['evaluation_seed'], run, shot, 4),
            'class_order_original_ids': originals, 'normalization': data_info['normalization'],
            'reproducibility': spec['reproducibility'],
            'model_source_sha256': {name: digest for name, digest in spec['source_sha256'].items()
                                    if name.startswith('models/')},
            'source_cache_sha256': data_info['sha256']}


def evaluate_checkpoint(data_dir, checkpoint_path, output_parent, *, device='auto', cpu_threads=None,
                        episodes=None, episode_seed=None, shot=None, way=None, query=None,
                        experiment_json=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'best.pt'
    checkpoint = engine.load_torch(checkpoint_path)
    settings = checkpoint_settings(checkpoint, experiment_json, checkpoint_path)
    current_sources = engine.source_hashes()
    for name, digest in settings['model_source_sha256'].items():
        if current_sources.get(name) != digest:
            raise ValueError('Checkpoint network source differs: ' + name)
    engine.configure_determinism(**settings['reproducibility'])
    if cpu_threads is not None:
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError('cpu_threads must be a positive integer')
        torch.set_num_threads(cpu_threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    data, metadata = load_test_cache(data_dir, settings['normalization'])
    reverse = label_mapping(data)
    forward = {original: label for label, original in reverse.items()}
    original_order = settings['class_order_original_ids']
    matching_classes = set(original_order) == set(forward)
    classes = ([forward[original] for original in original_order] if matching_classes
               else metadata.get('class_order', sorted(reverse)))
    values = {'episodes': settings['test_episodes'] if episodes is None else episodes,
              'shot': settings['shot'] if shot is None else shot,
              'way': settings['way'] if way is None else way,
              'query': settings['query_per_class'] if query is None else query}
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError('episodes, shot, way and query must be positive integers')
    seed = settings['episode_seed'] if episode_seed is None else episode_seed
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('episode seed must be an integer in 0..4294967295')
    plan = episode_plan(data.labels, classes, values['way'], values['shot'], values['query'],
                        values['episodes'], seed)
    model = MCGNet(ModelConfig.from_inference(settings['model']), initialize=False).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('Checkpoint:', checkpoint_path, flush=True)
    print('Test cache:', data.directory, '| samples:', len(data.labels), flush=True)
    print('Original IDs:', [reverse[label] for label in classes], flush=True)
    print('Device: %s | %d-way %d-shot | query/class=%d | episodes=%d | episode seed=%d' %
          (device, values['way'], values['shot'], values['query'], values['episodes'], seed), flush=True)
    details = {'logits': [], 'predicted_local': []}
    accuracies = engine.episode_accuracy(model, data, plan, values['shot'], device, details=details)
    output_parent = Path(output_parent).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='t_', dir=str(output_parent)))
    prefix = 's%d_r01' % values['shot']
    description = data.describe(hashes=True)
    description['label_mapping_sha256'] = file_hash(data.directory / 'label_mapping.json')
    context = {'seed': settings.get('training_seed'), 'episode_seed': seed,
               'checkpoint_epoch': checkpoint.get('epoch'), 'checkpoint_path': str(checkpoint_path),
               'checkpoint_sha256': file_hash(checkpoint_path),
               'learned_cosine_scale': float(model.scale.detach().cpu()),
               'spec_sha256': checkpoint.get('spec_sha256')}
    analysis = save_query_artifacts(data, plan, values['shot'], np.stack(details['logits']),
                                    np.stack(details['predicted_local']), output, prefix, context, classes)
    np.savez_compressed(output / (prefix + '_episodes.npz'), indices=plan, accuracies=accuracies)
    source_file = None
    if hasattr(data, 'source_indices'):
        source_file = prefix + '_source_indices.npz'
        np.savez_compressed(output / source_file, episode_source_indices=data.source_indices[plan],
                            query_source_indices=data.source_indices[plan[:, :, values['shot']:]].reshape(
                                values['episodes'], values['way'] * values['query']))
    with (output / (prefix + '.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode', 'accuracy_0_1'])
        writer.writerows(enumerate(accuracies.tolist(), 1))
    statistics = engine.test_episode_statistics(accuracies)
    result = {**context, **values, 'training_shot': settings['shot'],
              'episode_statistics': statistics, 'run_mean_accuracy': float(accuracies.mean()),
              'matching_checkpoint_test_classes': matching_classes,
              'data': description, 'normalization': settings['normalization'],
              'reproducibility': settings['reproducibility'], 'class_analysis': analysis,
              'original_id_order': [reverse[label] for label in classes],
              'source_indices_file': source_file, 'output_directory': str(output),
              'source_sha256_at_evaluation': current_sources}
    write_json(output / 'test_summary.json', result)
    engine.print_episode_statistics(statistics, 'Test accuracy:')
    print('Results:', output, flush=True)
    return result
