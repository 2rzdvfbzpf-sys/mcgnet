"""Train and evaluate encoder, fusion and input ablations."""
import _bootstrap
import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np

from data.radar_cache import RadarCache, SPLIT_KEYS, class_split, episode_plan
from networks import VARIANTS
from reporting import update_summary
from runner import cache_description, fit_cca_trial, runtime, train_one
from settings import HERE, own_path, read_json, snapshot_sources, source_hashes, validate_config, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--variants', choices=list(VARIANTS), nargs='+')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--shots', type=int, nargs='+')
    p.add_argument('--epochs', type=int)
    p.add_argument('--patience', type=int)
    p.add_argument('--train-episodes', type=int)
    p.add_argument('--val-episodes', type=int)
    p.add_argument('--test-episodes', type=int)
    p.add_argument('--test-every-epochs', type=int)
    p.add_argument('--monitor-start-epoch', type=int)
    p.add_argument('--monitor-test-episodes', type=int)
    p.add_argument('--device')
    p.add_argument('--cpu-threads', type=int)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--check-plan', '--check-split', action='store_true')
    p.add_argument('--describe', action='store_true')
    p.add_argument('--resume', type=Path, help='An ablation batch directory')
    p.add_argument('--stop-after-epoch', type=int, help='Save an epoch checkpoint and pause this batch')
    return p


def overrides(config, args, check_output_paths=True):
    cfg = copy.deepcopy(config)
    locations = {'variants': ('ablation', 'variants'), 'shots': ('experiment', 'shots'),
                 'epochs': ('training', 'epochs'), 'patience': ('training', 'patience'),
                 'train_episodes': ('training', 'episodes_per_epoch'),
                 'val_episodes': ('validation', 'episodes'), 'test_episodes': ('testing', 'episodes'),
                 'test_every_epochs': ('testing', 'every_epochs'),
                 'monitor_start_epoch': ('testing', 'start_epoch'),
                 'monitor_test_episodes': ('testing', 'periodic_episodes'),
                 'device': ('runtime', 'device'), 'cpu_threads': ('runtime', 'cpu_threads')}
    for arg, (section, key) in locations.items():
        value = getattr(args, arg)
        if value is not None:
            cfg[section][key] = value
    if args.seeds is not None:
        cfg['experiment'].update(seeds=args.seeds, runs=len(args.seeds))
    for arg, section, key in (('cache_dir', 'data', 'cache_dir'),
                               ('output_dir', 'output', 'results_dir'),
                               ('checkpoint_dir', 'output', 'checkpoints_dir')):
        if getattr(args, arg) is not None:
            cfg[section][key] = str(getattr(args, arg).resolve())
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError('stop-after-epoch must be positive')
    return validate_config(cfg, check_output_paths=check_output_paths)


def execution_order(variants):
    order = list(variants)
    dependencies = []
    if 'tpa_re_cca' in order:
        if 'tpa_re_concat' not in order:
            order.append('tpa_re_concat')
            dependencies.append('tpa_re_concat')
        order.remove('tpa_re_cca')
        order.insert(order.index('tpa_re_concat')+1, 'tpa_re_cca')
    return order, dependencies


def make_plans(data, description, config):
    original = {int(label): raw for label, raw in description['original_ids'].items()}
    plans = []
    for index, seed in enumerate(config['experiment']['seeds'], 1):
        split = class_split(data.labels, seed, counts=config['data']['class_counts'])
        raw = {key: [original[label] for label in split[key]] for key in SPLIT_KEYS}
        for key in SPLIT_KEYS:
            for shot in config['experiment']['shots']:
                episode_plan(data.labels, split[key], config['episode']['way'], shot,
                             config['episode']['query_per_class'], 1, seed)
        plan = {'seed': seed, 'directory': 'r%02d_%d' % (index, seed), 'split': split,
                'original_ids': raw,
                'sample_counts': {key: int(np.isin(data.labels, split[key]).sum()) for key in SPLIT_KEYS}}
        plans.append(plan)
        print('Run %d | shared split/training seed=%d | Sample counts: %s' % (index, seed, plan['sample_counts']), flush=True)
        for key in SPLIT_KEYS:
            print('  %s original IDs: %s' % (key, raw[key]), flush=True)
    return plans


def run_batch(data, manifest, batch, checkpoint_root, description, device,
              resume=False, stop_after=None):
    config = manifest['config']
    for plan in manifest['plans']:
        run_dir = Path(batch) / plan['directory']
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / 'split.json', plan)
        for variant in manifest['execution_variants']:
            for shot in config['experiment']['shots']:
                directory = run_dir / variant / ('k%d' % shot)
                checkpoints = checkpoint_root / plan['directory'] / variant / ('k%d' % shot)
                if variant == 'tpa_re_cca':
                    concat = checkpoint_root / plan['directory'] / 'tpa_re_concat' / ('k%d' % shot) / 'best.pt'
                    result = fit_cca_trial(data, plan['split'], config, description, plan['seed'],
                                           shot, directory, checkpoints, concat, device, resume)
                else:
                    result = train_one(variant, data, plan['split'], config, description,
                                        plan['seed'], shot, directory, checkpoints, device, resume, stop_after)
                update_summary(batch, manifest)
                if result is None:
                    print('Batch paused after an epoch checkpoint. Resume with --resume', batch, flush=True)
                    return
    summary = update_summary(batch, manifest)
    print('Completed models: %d/%d | Results: %s' % (
        summary['completed_models'], summary['expected_models'], batch), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume:
        forbidden = ('variants', 'seeds', 'shots', 'epochs', 'patience', 'train_episodes',
                     'val_episodes', 'test_episodes', 'test_every_epochs', 'monitor_start_epoch',
                     'monitor_test_episodes', 'output_dir', 'checkpoint_dir')
        if any(getattr(args, name) is not None for name in forbidden):
            raise ValueError('Resume uses the saved experiment settings')
        batch = args.resume.resolve()
        manifest = read_json(batch / 'manifest.json')
        if manifest.get('format') != 'mcg_ablation_batch_v1' or manifest['source_sha256'] != source_hashes():
            raise ValueError('Use this batch source_snapshot/train.py to resume its implementation')
        config = overrides(manifest['config'], args, check_output_paths=False)
    else:
        config = overrides(read_json(args.config), args)
        batch, manifest = None, None
    if args.describe:
        print('Variants:', config['ablation']['variants'])
        print('ResNet-12 stage widths:', config['ablation']['resnet12_channels'])
        print('Encoder activation: Mish; last-level embedding: 112; CPC: scaled cosine prototypes')
        print('Protocol: class-disjoint, no subject filtering, no support/query sample overlap')
        return
    cache_path = Path(manifest['data']['cache_dir'] if manifest is not None and args.cache_dir is None
                      else config['data']['cache_dir'])
    cache_path = cache_path if cache_path.is_absolute() else HERE / cache_path
    data = RadarCache(cache_path, config['data']['normalization'])
    description = cache_description(data)
    plans = make_plans(data, description, config)
    order, dependencies = execution_order(config['ablation']['variants'])
    if args.check_plan:
        print('Execution variants:', order, '\nDependency variants:', dependencies, flush=True)
        print('Class splits and episode sample counts verified; training not started.', flush=True)
        return
    if manifest is None:
        result_root = own_path(config['output']['results_dir'])
        result_root.mkdir(parents=True, exist_ok=True)
        batch = Path(tempfile.mkdtemp(prefix='b_', dir=result_root))
        checkpoint_root = own_path(config['output']['checkpoints_dir']) / batch.name
        import os
        manifest = {'format': 'mcg_ablation_batch_v1', 'config': config, 'plans': plans,
                    'execution_variants': order, 'dependency_variants': dependencies,
                    'data': description, 'source_sha256': source_hashes(),
                    'checkpoint_relative_root': os.path.relpath(checkpoint_root, batch)}
        write_json(batch / 'manifest.json', manifest)
        write_json(batch / 'config.json', config)
        snapshot_sources(batch / 'source_snapshot', config)
    else:
        if (manifest['data']['sha256'] != description['sha256']
                or manifest['data']['label_mapping_sha256'] != description['label_mapping_sha256']
                or manifest['plans'] != plans):
            raise ValueError('Saved batch data or class splits differ')
        checkpoint_root = (batch / manifest['checkpoint_relative_root']).resolve()
        manifest = {**manifest, 'config': config}
    run_batch(data, manifest, batch, checkpoint_root, description, runtime(config),
              resume=args.resume is not None, stop_after=args.stop_after_epoch)


if __name__ == '__main__':
    main()
