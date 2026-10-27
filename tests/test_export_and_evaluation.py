"""Test-cache export and standalone checkpoint evaluation."""

import os as _seed_environment

def _test_seed(name):
    value = _seed_environment.environ.get(name)
    if value is None:
        raise unittest.SkipTest("Set explicit test seed environment variables")
    return int(value)

import contextlib
import gc
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from configuration import load_config
from data.radar_cache import read_json, write_json
from data.test_cache import load_test_cache
from training import engine
from training.evaluation import evaluate_checkpoint


class ExportAndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parent = Path(__file__).resolve().parent / '.tmp'
        cls.parent.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='evaluation_tests_', dir=str(cls.parent))
        cls.root = Path(cls.temporary.name).resolve()
        cls.cache = cls.root / 'cache'
        cls.cache.mkdir()
        rng = np.random.default_rng(_test_seed('TEST_SEED_C'))
        np.save(cls.cache / 'labels.npy', np.repeat(np.arange(50), 40))
        for name in ('dt', 'at'):
            np.save(cls.cache / (name + '.npy'), rng.normal(size=(2000, 25, 64)).astype(np.float32))
        write_json(cls.cache / 'label_mapping.json', {str(101 + 2 * label): label for label in range(50)})
        config = load_config()
        config['experiment'].update(runs=1, seeds=[_test_seed('TEST_SEED_D')], shots=[1, 5, 10, 20])
        config['data']['cache_dir'] = str(cls.cache)
        config['model'].update(channels=16, attention_channels=4)
        config['episode'].update(way=3, query_per_class=2)
        config['training'].update(epochs=2, episodes_per_epoch=1)
        config['validation']['episodes'] = 1
        config['testing'].update(episodes=3, start_epoch=50)
        config['runtime'].update(device='cpu', cpu_threads=2)
        config['output'].update(results_dir=str(cls.root / 'results'), checkpoints_dir=str(cls.root / 'weights'))
        path = cls.root / 'config.json'
        write_json(path, config)
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(['--config', str(path)])
        cls.batch = next((cls.root / 'results').glob('b_*'))
        cls.manifest = read_json(cls.batch / 'batch.json')
        cls.plan = cls.manifest['plans'][0]
        cls.round = cls.batch / cls.plan['directory']
        cls.exported = cls.round / 'test_cache'
        cls.weight_root = Path(cls.manifest['checkpoint_root']) / cls.plan['directory']
        cls.record = read_json(cls.round / 'round_result.json')
        cls.final = (cls.batch / cls.record['test_summary_path']).parent
        cls.portable = cls.root / 'portable'
        cls.portable.mkdir()
        shutil.copytree(cls.exported, cls.portable / 'data')
        for shot in (1, 5, 10, 20):
            shutil.copyfile(cls.weight_root / ('s%d_r01/best.pt' % shot), cls.portable / ('model_%d.pt' % shot))

    @classmethod
    def tearDownClass(cls):
        if cls.root.parent != cls.parent.resolve() or not cls.root.name.startswith('evaluation_tests_'):
            raise AssertionError('Unexpected temporary cleanup path')
        gc.collect()
        cls.temporary.cleanup()

    def test_export_contains_exactly_the_test_samples(self):
        labels = np.load(self.cache / 'labels.npy')
        expected = np.flatnonzero(np.isin(labels, self.plan['split']['test_classes']))
        np.testing.assert_array_equal(np.load(self.exported / 'source_indices.npy'), expected)
        np.testing.assert_array_equal(np.load(self.exported / 'labels.npy'), labels[expected])
        for name in ('dt', 'at'):
            np.testing.assert_array_equal(np.load(self.exported / (name + '.npy')),
                                          np.load(self.cache / (name + '.npy'))[expected])
        mapping = read_json(self.exported / 'label_mapping.json')
        self.assertEqual(set(mapping.values()), set(self.plan['split']['test_classes']))
        self.assertEqual(len(mapping), 10)
        self.assertEqual(self.record['test_cache_directory'], str(Path(self.plan['directory']) / 'test_cache'))

    def test_all_shots_match_the_training_final_evaluation(self):
        for shot in (1, 5, 10, 20):
            with self.subTest(shot=shot), patch.object(torch.optim, 'Adam', side_effect=AssertionError('optimizer')), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_checkpoint(self.portable / 'data', self.portable / ('model_%d.pt' % shot),
                                               self.root / 'manual_results', device='cpu', cpu_threads=2)
            row = next(row for row in self.record['rows'] if row['shot'] == shot)
            self.assertEqual(row['run_mean_accuracy'], result['run_mean_accuracy'])
            self.assertEqual(result['episodes'], 3)
            summary = read_json(self.final / 'test_summary.json')
            expected_statistics = next(row['episode_statistics'] for row in summary['runs'] if row['shot'] == shot)
            self.assertEqual(result['episode_statistics'], expected_statistics)
            self.assertTrue(result['matching_checkpoint_test_classes'])
            output = Path(result['output_directory'])
            prefix = 's%d_r01' % shot
            with np.load(output / (prefix + '_queries.npz')) as actual, \
                    np.load(self.final / (prefix + '_queries.npz')) as expected:
                for name in ('logits', 'true_original_ids', 'predicted_original_ids',
                             'episode_class_original_ids', 'predicted_probabilities'):
                    np.testing.assert_array_equal(actual[name], expected[name])
            with np.load(output / (prefix + '_source_indices.npz')) as actual, \
                    np.load(self.final / (prefix + '_episodes.npz')) as expected:
                np.testing.assert_array_equal(actual['episode_source_indices'], expected['indices'])
            with np.load(output / (prefix + '_episodes.npz')) as saved:
                self.assertEqual(engine.test_episode_statistics(saved['accuracies']), result['episode_statistics'])
            self.assertEqual(read_json(output / 'test_summary.json')['episode_statistics'], result['episode_statistics'])
            metadata = read_json(output / (prefix + '_query_metadata.json'))
            self.assertEqual(len(metadata['cache_to_original_id']), 10)

    def test_portable_cli_without_training_directories(self):
        gc.collect()
        moved = []
        try:
            for source in (self.cache, self.root / 'results', self.root / 'weights'):
                target = source.with_name('offline_' + source.name)
                self.assertEqual(source.resolve().parent, self.root)
                self.assertEqual(target.resolve().parent, self.root)
                source.rename(target)
                moved.append((source, target))
            command = [sys.executable, '-B', '-X', 'utf8', str(train.PROJECT / 'test.py'),
                       '--data-dir', str(self.portable / 'data'), '--checkpoint', str(self.portable / 'model_1.pt'),
                       '--output-dir', str(self.root / 'cli_results'), '--episodes', '2',
                       '--device', 'cpu', '--cpu-threads', '2']
            completed = subprocess.run(command, cwd=str(self.portable), capture_output=True, text=True,
                                        encoding='utf-8')
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = next((self.root / 'cli_results').glob('t_*'))
            self.assertEqual(read_json(output / 'test_summary.json')['episodes'], 2)
            self.assertIn('Test accuracy:', completed.stdout)
            self.assertIn('Sample variance:', completed.stdout)
            self.assertIn('Mean accuracy +/- 95% CI:', completed.stdout)
            saved = read_json(output / 'test_summary.json')['episode_statistics']
            self.assertIsNotNone(saved['mean_95ci_half_width_pp'])
            self.assertIsNotNone(saved['sample_variance_pp2'])
        finally:
            for source, target in reversed(moved):
                target.rename(source)

    def test_custom_folder_remapped_labels_and_explicit_episode_seed(self):
        custom = self.root / 'custom_data'
        custom.mkdir()
        mapping = read_json(self.exported / 'label_mapping.json')
        ordered = sorted(mapping, key=int)
        new_mapping = {original: index for index, original in enumerate(ordered)}
        lookup = {mapping[original]: new_mapping[original] for original in ordered}
        labels = np.load(self.exported / 'labels.npy')
        np.save(custom / 'labels.npy', np.asarray([lookup[int(label)] for label in labels]))
        write_json(custom / 'label_mapping.json', new_mapping)
        for name in ('dt', 'at'):
            shutil.copyfile(self.exported / (name + '.npy'), custom / (name + '.npy'))
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_checkpoint(custom, self.portable / 'model_1.pt', self.root / 'custom_results',
                                           device='cpu', episodes=3)
        with np.load(Path(result['output_directory']) / 's1_r01_queries.npz') as actual, \
                np.load(self.final / 's1_r01_queries.npz') as expected:
            np.testing.assert_array_equal(actual['logits'], expected['logits'])
            np.testing.assert_array_equal(actual['true_original_ids'], expected['true_original_ids'])
        indices = []
        for seed in (_test_seed('TEST_SEED_A'), _test_seed('TEST_SEED_A'), _test_seed('TEST_SEED_B')):
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_checkpoint(custom, self.portable / 'model_1.pt', self.root / 'seed_results',
                                               device='cpu', episodes=2, episode_seed=seed)
            with np.load(Path(result['output_directory']) / 's1_r01_episodes.npz') as saved:
                indices.append(saved['indices'].copy())
        np.testing.assert_array_equal(indices[0], indices[1])
        self.assertFalse(np.array_equal(indices[0], indices[2]))

    def test_test_data_and_checkpoint_validation(self):
        invalid = self.root / 'invalid_data'
        shutil.copytree(self.exported, invalid)
        with (invalid / 'at.npy').open('ab') as handle:
            handle.write(b'x')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_test_cache(invalid, 'per_map_minmax')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            load_test_cache(self.exported, 'already_normalized')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'distinct samples'):
            evaluate_checkpoint(self.exported, self.portable / 'model_1.pt', self.root / 'bad_results',
                                  device='cpu', shot=39)
        checkpoint = engine.load_torch(self.portable / 'model_1.pt')
        checkpoint['inference']['model_source_sha256']['models/cpc.py'] = 'invalid'
        engine.save_torch(self.root / 'invalid.pt', checkpoint)
        with self.assertRaisesRegex(ValueError, 'network source'):
            evaluate_checkpoint(self.exported, self.root / 'invalid.pt', self.root / 'bad_results', device='cpu')


if __name__ == '__main__':
    unittest.main()
