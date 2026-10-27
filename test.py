"""MCG-Net evaluation on a specified test cache and checkpoint."""
import argparse
from pathlib import Path
import sys

from train import positive_int, uint32_seed
from training.evaluation import evaluate_checkpoint


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', '--cache-dir', dest='data_dir', type=Path, required=True,
                        help='Folder with dt.npy, at.npy, labels.npy and label_mapping.json')
    parser.add_argument('--checkpoint', dest='checkpoint_path', type=Path, required=True,
                        help='Checkpoint file, or its directory containing best.pt')
    parser.add_argument('--output-dir', dest='output_parent', type=Path,
                        default=Path(__file__).resolve().parent / 'results/manual_test')
    parser.add_argument('--episodes', type=positive_int, help='Default: checkpoint test budget')
    parser.add_argument('--episode-seed', '--seed', dest='episode_seed', type=uint32_seed,
                        help='Direct episode sampling seed; default: checkpoint final-test seed')
    parser.add_argument('--shot', type=positive_int, help='Default: checkpoint training shot')
    parser.add_argument('--way', type=positive_int, help='Default: checkpoint evaluation way')
    parser.add_argument('--query', type=positive_int, help='Query samples per class')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=positive_int)
    parser.add_argument('--experiment-json', type=Path,
                        help='Experiment specification for checkpoints without inference metadata')
    args = parser.parse_args(argv)
    return evaluate_checkpoint(**vars(args))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit('FAILED: ' + str(error))
