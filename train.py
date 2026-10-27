"""Multi-stream Cross-Gated Network (MCG-Net)."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sys
import tempfile

import numpy as np
import torch

from models import ModelConfig
from analysis.reports import aggregate_class_reports
from data.radar_cache import (RadarCache, SPLIT_KEYS, class_split, episode_plan,
                            file_hash, read_json, write_json)
from data.test_cache import export_test_cache
from training import engine as trainer
from configuration import load_config, resolve_paths, training_settings, validate_config


PROJECT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return number


def uint32_seed(value):
    number = nonnegative_int(value)
    if number > 2**32 - 1:
        raise argparse.ArgumentTypeError('seed must be in 0..4294967295')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Explicit experiment JSON')
    p.add_argument('--runs', '--n', type=positive_int,
                   help='Override experiment.runs')
    p.add_argument('--seeds', '--seed', dest='seeds', type=uint32_seed, nargs='+',
                   help='Explicit run seeds in order; count must match experiment.runs or --runs')
    p.add_argument('--shots', type=positive_int, nargs='+')
    p.add_argument('--epochs', type=positive_int)
    p.add_argument('--patience', type=nonnegative_int,
                   help='Override early stopping patience; 0 disables early stopping')
    p.add_argument('--train-episodes', type=positive_int)
    p.add_argument('--val-episodes', type=positive_int)
    p.add_argument('--test-episodes', type=positive_int)
    p.add_argument('--test-every-epochs', type=nonnegative_int,
                   help='0: test on validation improvement; positive: test every N epochs')
    p.add_argument('--monitor-start-epoch', type=positive_int,
                   help='First epoch eligible for training-time tests (inclusive)')
    p.add_argument('--monitor-test-episodes', type=positive_int)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--check-plan', '--check-split', dest='check_plan', action='store_true',
                   help='Print all configured seeds and class splits only; never construct a model')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--resume', type=Path, help='Continue a batch using its saved seeds/settings')
    mode.add_argument('--test-only', type=Path, help='Retest all completed best models in a saved batch')
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--device', help='Override runtime.device: auto, cuda or cpu')
    p.add_argument('--cpu-threads', type=positive_int)
    return p


def model_config(config):
    return ModelConfig(**config['model'])


def apply_overrides(config, args):
    """Explicit command-line values take precedence over the configuration file."""
    locations = {
        'runs': ('experiment', 'runs'), 'seeds': ('experiment', 'seeds'),
        'shots': ('experiment', 'shots'), 'epochs': ('training', 'epochs'),
        'patience': ('training', 'patience'), 'train_episodes': ('training', 'episodes_per_epoch'),
        'val_episodes': ('validation', 'episodes'), 'test_episodes': ('testing', 'episodes'),
        'test_every_epochs': ('testing', 'every_epochs'),
        'monitor_start_epoch': ('testing', 'start_epoch'),
        'monitor_test_episodes': ('testing', 'periodic_episodes'),
        'device': ('runtime', 'device'), 'cpu_threads': ('runtime', 'cpu_threads'),
    }
    for argument, (section, name) in locations.items():
        value = getattr(args, argument)
        if value is not None:
            config[section][name] = value
    for argument, section, name in (('cache_dir', 'data', 'cache_dir'),
                                     ('output_dir', 'output', 'results_dir'),
                                     ('checkpoint_dir', 'output', 'checkpoints_dir')):
        value = getattr(args, argument)
        if value is not None:
            config[section][name] = str(value.resolve())
    return resolve_paths(validate_config(config))


def validate_cache(data):
    if not np.array_equal(np.unique(data.labels), np.arange(50)):
        raise ValueError('Final entry requires 50 cache classes labeled 0..49')
    mapping = read_json(data.directory / 'label_mapping.json')
    reverse = {int(value): int(raw) for raw, value in mapping.items()}
    if len(mapping) != 50 or set(reverse) != set(range(50)):
        raise ValueError('label_mapping.json must map 50 original IDs one-to-one to 0..49')
    description = data.describe(hashes=True)
    description['label_mapping_sha256'] = file_hash(data.directory / 'label_mapping.json')
    return reverse, description


def make_run_plan(data, reverse, seeds, shots, counts, way, query):
    plans = []
    for index, seed in enumerate(seeds, 1):
        split = class_split(data.labels, seed, counts=counts)
        for key in SPLIT_KEYS:
            for shot in shots:
                episode_plan(data.labels, split[key], way, shot, query, 1, seed)
        split.update(
            raw_gesture_ids={key: [reverse[c] for c in split[key]] for key in SPLIT_KEYS},
            sample_counts={key: int(np.isin(data.labels, split[key]).sum()) for key in SPLIT_KEYS})
        plans.append({'repetition': index, 'seed': seed, 'split': split,
                      'directory': 'r%02d_%d' % (index, seed)})
    return plans


def print_plan(manifest):
    print('Independent runs per shot:', len(manifest['run_seeds']), flush=True)
    print('Configured run seeds:', manifest['run_seeds'], flush=True)
    print('Each run seed controls class split, initialization, training, validation and testing.', flush=True)
    for plan in manifest['plans']:
        print('Run %d: shared seed=%d' % (plan['repetition'], plan['seed']), flush=True)
        for key in SPLIT_KEYS:
            split = plan['split']
            print('  %s: %d classes, %d samples; original IDs=%s' %
                  (key, len(split[key]), split['sample_counts'][key], split['raw_gesture_ids'][key]),
                  flush=True)


def choose_device(args):
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return device


def check_resume_options(argv):
    allowed = {'--resume', '--test-only', '--device', '--cpu-threads', '--check-plan', '--check-split'}
    for argument in argv:
        if argument.startswith('--') and argument.split('=', 1)[0] not in allowed:
            raise ValueError('Resume/test-only uses saved seeds and settings; pass only its path, '
                             '--device or --cpu-threads (and optionally --check-plan)')


def checkpoint_hashes(directory, shots):
    root = Path(read_json(directory / 'experiment.json')['checkpoint_directory'])
    return {str(shot): file_hash(root / ('s%d_r01' % shot) / 'best.pt') for shot in shots}


def make_rows(plan, training, tested):
    trained = {row['shot']: row for row in training['runs']}
    rows = []
    for result in tested['runs']:
        run = trained[result['shot']]
        rows.append({'repetition': plan['repetition'], 'seed': plan['seed'], 'shot': result['shot'],
                     'class_split_seed': plan['split']['class_split_seed'],
                     'training_seed': run['training_seed'], 'evaluation_seed': run['evaluation_seed'],
                     'best_epoch': run['best_epoch'], 'epochs_completed': run['epochs_completed'],
                     'best_validation_accuracy_percent': run['best_validation_accuracy'] * 100,
                     'final_test_accuracy_percent': result['run_mean_accuracy'] * 100,
                     'run_mean_accuracy': result['run_mean_accuracy'],
                     'learned_cosine_scale': result['learned_cosine_scale'],
                     'test_episodes': result['episode_statistics']['n'],
                     **{'test_episode_' + name: result['episode_statistics'][name] for name in (
                         'sample_variance_pp2', 'sample_sd_pp', 'mean_95ci_half_width_pp',
                         'mean_95ci_lower_percent', 'mean_95ci_upper_percent')}})
    return rows


def save_batch_summary(directory, manifest, completed):
    import csv
    rows = [row for result in completed for row in result['rows']]
    summary = {'complete': len(completed) == len(manifest['plans']),
               'run_seeds': manifest['run_seeds'],
               'requested_repetitions': len(manifest['plans']), 'completed_repetitions': len(completed),
               'statistical_unit': 'one_class_split_and_independently_trained_model',
               'class_split_changes_between_runs': True, 'test_used_for_selection': False,
               'by_shot': {str(shot): trainer.accuracy_statistics(
                   [row['run_mean_accuracy'] for row in rows if row['shot'] == shot],
                   'mean_test_accuracy_per_split_and_training_run')
                   for shot in manifest['settings']['shots']} if rows else {},
               'runs': rows, 'class_analysis_by_shot': {}}
    for shot in manifest['settings']['shots']:
        analyses = [result['class_analyses'][str(shot)] for result in completed]
        if analyses:
            analysis = aggregate_class_reports(analyses, directory / 'analysis', 's%d' % shot)
            summary['class_analysis_by_shot'][str(shot)] = {
                'run_count': len(analyses), 'directory': 'analysis',
                'artifact_files': analysis['artifact_files']}
    write_json(directory / 'summary.json', summary)
    if rows:
        with (directory / 'runs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def execute_batch(data, manifest, directory, device, resume=False, test_only=False):
    manifest_id = trainer.signature(manifest)
    mc = ModelConfig(**manifest['model'])
    completed = []
    for plan in manifest['plans']:
        round_dir = directory / plan['directory']
        record_path = round_dir / 'round_result.json'
        print('\n>>> Repetition %d/%d | shared seed=%d' %
              (plan['repetition'], len(manifest['plans']), plan['seed']), flush=True)
        round_dir.mkdir(parents=True, exist_ok=True)
        export_test_cache(data, plan['split'], round_dir / 'test_cache', manifest['data'])
        print('Test cache:', round_dir / 'test_cache', flush=True)
        if record_path.exists() and not test_only:
            record = read_json(record_path)
            if (not resume or record['batch_sha256'] != manifest_id
                    or record['checkpoint_sha256'] != checkpoint_hashes(round_dir, manifest['settings']['shots'])):
                raise ValueError('Completed repetition exists or its saved record/checkpoint differs')
            completed.append(record)
            save_batch_summary(directory, manifest, completed)
            print('Completed repetition verified; reusing its saved final results.', flush=True)
            continue
        tc = trainer.TrainingConfig(**manifest['settings'], runs=1,
                                    seed=plan['seed'], evaluation_seed=plan['seed'])
        if test_only:
            training = read_json(round_dir / 'training_summary.json')
        else:
            training = trainer.train_experiment(data, plan['split'], mc, tc, round_dir, device,
                                                resume=resume, data_description=manifest['data'],
                                                checkpoint_directory=Path(manifest['checkpoint_root']) / plan['directory'],
                                                configuration=manifest['configuration'])
        test_dir = round_dir / ('t_' + secrets.token_hex(4))
        tested = trainer.test_experiment(data, round_dir, device, test_dir)
        record = {'complete': True, 'batch_sha256': manifest_id,
                  'repetition': plan['repetition'], 'seed': plan['seed'],
                  'checkpoint_sha256': checkpoint_hashes(round_dir, tc.shots),
                  'test_cache_directory': str((round_dir / 'test_cache').relative_to(directory)),
                  'test_summary_path': str((test_dir / 'test_summary.json').relative_to(directory)),
                  'class_analyses': {str(row['shot']): row['class_analysis'] for row in tested['runs']},
                  'rows': make_rows(plan, training, tested)}
        write_json(record_path, record)
        completed.append(record)
        save_batch_summary(directory, manifest, completed)
        for row in record['rows']:
            print('Run %d | seed=%d | %d-shot | best epoch=%d | final test=%.4f%%' %
                  (row['repetition'], row['seed'], row['shot'], row['best_epoch'],
                   row['final_test_accuracy_percent']), flush=True)
    return save_batch_summary(directory, manifest, completed)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    saved_path = args.resume or args.test_only
    if saved_path:
        check_resume_options(argv)
        directory = saved_path.resolve()
        manifest = read_json(directory / 'batch.json')
        if trainer.source_hashes() != manifest['source_sha256']:
            raise ValueError('Source files changed; resume/test with this batch sources snapshot')
        config = validate_config(manifest['configuration'])
        if trainer.signature(config) != manifest['configuration_sha256']:
            raise ValueError('Saved configuration checksum differs')
        if config['experiment']['seeds'] != manifest['run_seeds']:
            raise ValueError('Saved repetition seeds differ from the explicit configuration seeds')
        data = RadarCache(manifest['data']['cache_dir'], config['data']['normalization'])
        reverse, description = validate_cache(data)
        if description != manifest['data']:
            raise ValueError('Saved cache/normalization/mapping differ')
        if make_run_plan(data, reverse, manifest['run_seeds'], manifest['settings']['shots'],
                         config['data']['class_counts'], manifest['settings']['way'],
                         manifest['settings']['query_per_class']) != manifest['plans']:
            raise ValueError('Saved seed/class split plan no longer reproduces')
    else:
        if args.config is None:
            raise ValueError('Training requires --config with explicit settings and seeds')
        config = apply_overrides(load_config(args.config), args)
        exp = config['experiment']
        seeds = list(exp['seeds'])
        settings = training_settings(config)
        # Check experiment settings before preparing output directories.
        trainer.TrainingConfig(**settings, runs=1, seed=seeds[0], evaluation_seed=seeds[0])
        data = RadarCache(config['data']['cache_dir'], config['data']['normalization'])
        reverse, description = validate_cache(data)
        manifest = {'format_version': 3, 'run_seeds': seeds,
                    'seed_selection': 'explicit experiment.seeds in configured order',
                    'model': model_config(config).to_dict(), 'settings': settings, 'data': description,
                    'configuration': config, 'configuration_sha256': trainer.signature(config),
                    'configuration_source': str(args.config.resolve()),
                    'configuration_source_sha256': file_hash(args.config),
                    'plans': make_run_plan(data, reverse, seeds, settings['shots'],
                                           config['data']['class_counts'], settings['way'],
                                           settings['query_per_class']),
                    'source_sha256': trainer.source_hashes(),
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'protocol': 'class_disjoint_no_subject_restriction',
                    'statistical_unit': 'one_class_split_and_independently_trained_model',
                    'seed_selection_uses_test_results': False}
        directory = None
        print('Configuration:', args.config.resolve(), flush=True)
    for name in ('device', 'cpu_threads'):
        if getattr(args, name) is None:
            setattr(args, name, config['runtime'][name])
    print_plan(manifest)
    if args.check_plan:
        print('Plan check passed. No model construction, training, or testing.', flush=True)
        return manifest
    trainer.configure_determinism(config['runtime']['deterministic_algorithms'],
                                   config['runtime']['cublas_workspace_config'])
    device = choose_device(args)
    if directory is None:
        output_dir = Path(config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='b_', dir=str(output_dir)))
        manifest['checkpoint_root'] = str((Path(config['output']['checkpoints_dir']) / directory.name).resolve())
        write_json(directory / 'batch.json', manifest)
        write_json(directory / 'config.json', config)
        snapshots = directory / 'sources'
        snapshots.mkdir()
        trainer.snapshot_sources(snapshots, config)
        save_batch_summary(directory, manifest, [])
    print('Batch directory:', directory, flush=True)
    print('Checkpoint directory:', manifest['checkpoint_root'], flush=True)
    print('Device:', device, '| MCG-Net | channels=%d | scale init=%g | patience=%d' %
          (manifest['model']['channels'], manifest['model']['scale_init'],
           manifest['settings']['patience']), flush=True)
    print('Episodes: way=%d | query/class=%d | train=%d | validation=%d | final test=%d' %
          tuple(manifest['settings'][name] for name in
                ('way', 'query_per_class', 'train_episodes', 'val_episodes', 'test_episodes')), flush=True)
    print('Final tests always load each validation-best checkpoint; all run results are aggregated.', flush=True)
    try:
        summary = execute_batch(data, manifest, directory, device,
                                resume=bool(args.resume), test_only=bool(args.test_only))
    except KeyboardInterrupt:
        print('\nInterrupted. Resume this batch with: --resume "%s"' % directory, flush=True)
        raise
    for shot, stats in summary['by_shot'].items():
        print('%s-shot | runs=%d | mean=%.4f%% | sample SD(pp)=%s | sample variance(pp^2)=%s' %
              (shot, stats['n'], stats['mean_accuracy_percent'], stats['sample_sd_pp'],
               stats['sample_variance_pp2']), flush=True)
    print('Completed. Summary:', directory / 'summary.json', flush=True)
    print('Per-class accuracy, test coverage and confusion matrices:', directory / 'analysis', flush=True)
    return summary


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit('FAILED: ' + str(error))
