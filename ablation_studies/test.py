"""Evaluate an ablation checkpoint on an aligned DTM/ATM cache."""
import _bootstrap
import argparse
from pathlib import Path
import tempfile

from data.radar_cache import RadarCache, episode_plan, plan_seed
from data.test_cache import label_mapping
from reporting import print_statistics
from runner import checkpoint_model, evaluate, load_checkpoint, runtime, seed_all
from settings import HERE, file_hash, own_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', '--cache-dir', dest='data_dir', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--episode-seed', type=int)
    parser.add_argument('--device')
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'best.pt'
    saved = load_checkpoint(checkpoint)
    config = saved['config']
    if args.device is not None:
        config['runtime']['device'] = args.device
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error('cpu-threads must be positive')
        config['runtime']['cpu_threads'] = args.cpu_threads
    data_path = args.data_dir if args.data_dir is not None else Path(saved['cache_description']['cache_dir'])
    data = RadarCache(data_path, config['data']['normalization'])
    reverse = label_mapping(data)
    to_label = {raw: label for label, raw in reverse.items()}
    original = saved['cache_description']['original_ids']
    raw_classes = [original[str(label)] for label in saved['split']['test_classes']]
    if not set(raw_classes).issubset(to_label):
        raise ValueError('Test cache must contain the checkpoint test classes, matched by original IDs')
    classes = [to_label[raw] for raw in raw_classes]
    if args.data_dir is None:
        if file_hash(data.directory / 'label_mapping.json') != saved['cache_description']['label_mapping_sha256']:
            raise ValueError('Training label mapping changed; supply --data-dir explicitly for another cache')
        for name, expected in saved['cache_description']['sha256'].items():
            if file_hash(data.directory / (name + '.npy')) != expected:
                raise ValueError('Training cache changed; supply --data-dir explicitly for another test cache')
    count = args.episodes if args.episodes is not None else config['testing']['episodes']
    episode_seed = args.episode_seed if args.episode_seed is not None else plan_seed(saved['seed'], 0, saved['shot'], 4)
    if count < 1 or not 0 <= episode_seed < 2**32:
        parser.error('episodes must be positive and episode-seed must be uint32')
    seed_all(saved['seed'])
    device = runtime(config)
    model = checkpoint_model(saved, device)
    plan = episode_plan(data.labels, classes, config['episode']['way'], saved['shot'],
                        config['episode']['query_per_class'], count, episode_seed)
    output = own_path(str(args.output_dir.resolve()) if args.output_dir is not None else 'results/manual_test')
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='t_', dir=output))
    context = {'seed': saved['seed'], 'variant': saved['variant'], 'checkpoint_epoch': saved['epoch'],
               'checkpoint_sha256': file_hash(checkpoint), 'episode_seed': episode_seed,
               'test_cache': str(data.directory), 'source_sha256': saved['source_sha256']}
    result = evaluate(model, data, plan, saved['shot'], device, directory, context, classes)
    print_statistics(saved['variant'] + ' independent test', result['episode_statistics'])
    print('Results:', directory, flush=True)


if __name__ == '__main__':
    main()
