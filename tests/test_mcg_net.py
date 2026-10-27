"""MCG-Net module and training tests."""

import os as _seed_environment

def _test_seed(name):
    value = _seed_environment.environ.get(name)
    if value is None:
        raise unittest.SkipTest("Set explicit test seed environment variables")
    return int(value)

import contextlib
import gc
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from configuration import load_config, validate_config
from models import MCGNet, ModelConfig, episode_scores
from models.tpa_re import TemporalPhysicalAttention
from data.radar_cache import RadarCache, class_split, read_json, write_json
from training import engine


class MCGNetProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp_parent = Path(__file__).resolve().parent / '.tmp'
        cls.temp_parent.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='mcg_tests_', dir=str(cls.temp_parent))
        cls.root = Path(cls.temporary.name).resolve()
        cls.cache = cls.root / 'synthetic_cache'
        cls.cache.mkdir()
        rng = np.random.default_rng(_test_seed('TEST_SEED_C'))
        np.save(cls.cache / 'labels.npy', np.repeat(np.arange(50), 20))
        for name in ('dt', 'at'):
            np.save(cls.cache / (name + '.npy'), rng.random((1000, 25, 64), dtype=np.float32))
        write_json(cls.cache / 'label_mapping.json', {str(101 + 2 * i): i for i in range(50)})

    @classmethod
    def tearDownClass(cls):
        expected_parent = cls.temp_parent.resolve()
        if cls.root.parent != expected_parent or not cls.root.name.startswith('mcg_tests_'):
            raise AssertionError('Unexpected temporary cleanup target')
        gc.collect()  # Release memory-mapped arrays before removing temporary files.
        cls.temporary.cleanup()

    def test_seed_plan_and_class_disjointness(self):
        config = load_config()
        self.assertEqual(config['experiment']['runs'], len(config['experiment']['seeds']))
        seeds = [_test_seed('TEST_SEED_A'), _test_seed('TEST_SEED_B')]
        config['experiment'].update(runs=2, seeds=seeds)
        path = self.root / 'seed_configuration.json'
        write_json(path, config)
        self.assertEqual(load_config(path)['experiment']['seeds'], seeds)
        data = RadarCache(self.cache)
        reverse, _ = train.validate_cache(data)
        plans = train.make_run_plan(data, reverse, seeds[:2], (1, 5))
        for plan in plans:
            groups = [set(plan['split'][key]) for key in
                      ('train_classes', 'validation_classes', 'test_classes')]
            self.assertEqual([len(group) for group in groups], [30, 10, 10])
            self.assertEqual(len(set.union(*groups)), 50)
            self.assertEqual(plan['seed'], plan['split']['class_split_seed'])
        self.assertNotEqual(plans[0]['split']['test_classes'], plans[1]['split']['test_classes'])
        for values in ([7], [7, 7], [7, -1], [7, 2**32], [7, True], [7, 1.5], [7, '8'], None):
            invalid = load_config()
            invalid['experiment'].update(runs=2, seeds=values)
            with self.subTest(seeds=values), self.assertRaisesRegex(ValueError, 'experiment.seeds'):
                validate_config(invalid)
        boundary = load_config()
        boundary['experiment'].update(runs=2, seeds=[int(np.iinfo(np.uint32).min), int(np.iinfo(np.uint32).max)])
        self.assertEqual(validate_config(boundary)['experiment']['seeds'], boundary['experiment']['seeds'])

    def test_deterministic_runtime(self):
        engine.configure_determinism()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}), \
                patch.object(torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaisesRegex(ValueError, 'fresh Python process'):
                engine.configure_determinism()
        for field, value in (('deterministic_algorithms', 1), ('cublas_workspace_config', ':1:1')):
            invalid = load_config()
            invalid['runtime'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'runtime.'):
                validate_config(invalid)

    def test_episode_accuracy_statistics_and_printing(self):
        statistics = engine.test_episode_statistics([0.2, 0.4, 0.6])
        self.assertEqual(statistics['statistical_unit'], 'test_episode')
        self.assertEqual(statistics['n'], 3)
        self.assertEqual(statistics['ddof'], 1)
        self.assertAlmostEqual(statistics['mean_accuracy_percent'], 40)
        self.assertAlmostEqual(statistics['sample_variance_0_1'], 0.04)
        self.assertAlmostEqual(statistics['sample_variance_pp2'], 400)
        self.assertAlmostEqual(statistics['sample_sd_pp'], 20)
        half_width = 1.96 * 20 / np.sqrt(3)
        self.assertAlmostEqual(statistics['mean_95ci_half_width_pp'], half_width)
        self.assertAlmostEqual(statistics['mean_95ci_lower_percent'], 40 - half_width)
        self.assertAlmostEqual(statistics['mean_95ci_upper_percent'], 40 + half_width)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            engine.print_episode_statistics(statistics)
        for text in ('Mean accuracy: 40.0000%', 'Sample variance: 400.000000 pp^2',
                     'Mean accuracy +/- 95% CI:', 'Interval:'):
            self.assertIn(text, output.getvalue())
        single = engine.test_episode_statistics([0.5])
        for key in ('sample_variance_pp2', 'sample_variance_0_1', 'sample_sd_pp',
                    'mean_95ci_half_width_pp', 'mean_95ci_lower_percent', 'mean_95ci_upper_percent'):
            self.assertIsNone(single[key])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            engine.print_episode_statistics(single)
        self.assertIn('N/A', output.getvalue())
        for values in ([], [[0.5]], [float('nan')], [-0.1], [1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                engine.test_episode_statistics(values)

    def test_attention_axis_mean_output_and_gradient(self):
        attention = TemporalPhysicalAttention(4, 2, ModelConfig()).double().eval()
        left = torch.randn(2, 4, 12, 32, dtype=torch.float64, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        height, width = right.shape[-2:]
        temporal = torch.nn.functional.adaptive_avg_pool2d(right, (height, 1))
        physical = torch.nn.functional.adaptive_avg_pool2d(right, (1, width)).transpose(2, 3)
        combined = attention.shared(torch.cat((temporal, physical), dim=2))
        temporal, physical = torch.split(combined, (height, width), dim=2)
        at = attention.time_head(temporal).sigmoid()
        ap = attention.physical_head(physical.transpose(2, 3)).sigmoid()
        reference = right + at * right + ap * right
        actual = attention(left)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward()
        actual_grads = {name: parameter.grad.clone() for name, parameter in attention.named_parameters()}
        attention.zero_grad(set_to_none=True)
        reference.square().mean().backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-12)
        for name, parameter in attention.named_parameters():
            torch.testing.assert_close(actual_grads[name], parameter.grad, rtol=1e-12, atol=1e-12)

    def test_paper_network_and_scale(self):
        model = MCGNet(train.model_config()).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 703911)
        self.assertEqual(model.encoder_dt.output_dim, 112)
        self.assertEqual(model.output_dim, 224)
        self.assertEqual(model.scale.item(), 10)
        inputs = (torch.rand(80, 1, 25, 64), torch.rand(80, 1, 25, 64))
        with torch.no_grad():
            scaled = episode_scores(model, *inputs, 5, 1, 15)
            model.scale.fill_(1)
            cosine = episode_scores(model, *inputs, 5, 1, 15)
        self.assertEqual(tuple(scaled.shape), (75, 5))
        self.assertTrue(torch.equal(scaled, cosine * 10))
        self.assertTrue(torch.equal(scaled.argmax(1), cosine.argmax(1)))

    def test_repetitions_checkpoints_resume_and_reports(self):
        def arguments(output):
            return ['--seeds', str(_test_seed('TEST_SEED_A')), str(_test_seed('TEST_SEED_B')), '--runs', '2', '--shots', '1', '--epochs', '2',
                    '--train-episodes', '1', '--val-episodes', '1', '--test-every-epochs', '1',
                    '--monitor-start-epoch', '1',
                    '--monitor-test-episodes', '1', '--cache-dir', str(self.cache),
                    '--output-dir', str(output), '--checkpoint-dir', str(self.root / 'weights'),
                    '--device', 'cpu', '--cpu-threads', '2']
        original_test = engine.test_experiment
        def small_test(*args, **kwargs):
            kwargs['test_count'] = 2  # Use two episodes for the synthetic test.
            return original_test(*args, **kwargs)
        with patch.object(engine, 'test_experiment', side_effect=small_test), \
                contextlib.redirect_stdout(io.StringIO()):
            continuous = train.main(arguments(self.root / 'continuous'))
            original_train = engine.train_experiment
            second_seed = _test_seed('TEST_SEED_B')
            def stop_second(*args, **kwargs):
                if args[3].seed == second_seed:
                    original_train(*args, **kwargs, stop_after_epoch=1)
                    raise KeyboardInterrupt()
                return original_train(*args, **kwargs)
            with patch.object(engine, 'train_experiment', side_effect=stop_second):
                with self.assertRaises(KeyboardInterrupt):
                    train.main(arguments(self.root / 'interrupted'))
            a = next((self.root / 'continuous').glob('b_*'))
            b = next((self.root / 'interrupted').glob('b_*'))
            resumed = train.main(['--resume', str(b), '--device', 'cpu'])
            self.assertEqual(continuous['runs'], resumed['runs'])
            with patch.object(torch.optim, 'Adam', side_effect=AssertionError('test-only optimizer')):
                tested = train.main(['--test-only', str(b), '--device', 'cpu'])
            self.assertEqual(resumed['runs'], tested['runs'])
        manifest = read_json(a / 'batch.json')
        self.assertEqual(manifest['run_seeds'], [_test_seed('TEST_SEED_A'), _test_seed('TEST_SEED_B')])
        self.assertEqual(manifest['configuration']['experiment']['seeds'], manifest['run_seeds'])
        self.assertFalse(list(a.rglob('*.pt')), 'Weights must be outside results/')
        for plan in manifest['plans']:
            paths = []
            for root in (a, b):
                spec = read_json(root / plan['directory'] / 'experiment.json')
                weights = Path(spec['checkpoint_directory']) / 's1_r01'
                paths.append(weights)
                self.assertTrue((weights / 'best.pt').is_file())
                self.assertTrue((weights / 'last.pt').is_file())
                summary = read_json(root / plan['directory'] / 'training_summary.json')
                self.assertTrue(summary['final_test_evaluated'])
            left, right = [engine.load_torch(path / 'last.pt') for path in paths]
            self.assertTrue(all(torch.equal(left['model'][key], right['model'][key])
                                for key in left['model']))
            self.assertEqual(left['history'], right['history'])
            self.assertIn('validation_loss', left['history'][0])
            self.assertTrue(all('lambda_dt_%d' % i in left['history'][0] for i in (1, 2, 3)))
        analysis = read_json(a / 'analysis/s1_per_class.json')
        self.assertEqual(sum(row['query_count'] for row in analysis['per_class']), 300)
        for row in analysis['per_class']:
            if row['query_count'] == 0:
                self.assertIsNone(row['accuracy_percent'])
                self.assertEqual(row['status'], 'not_evaluated')
        copied = subprocess.run([sys.executable, str(a / 'sources/train.py'), '--help'],
                                cwd=str(self.root), capture_output=True, text=True)
        self.assertEqual(copied.returncode, 0, copied.stderr)

    def test_both_shots_have_separate_models_and_statistics(self):
        original_test = engine.test_experiment
        def small_test(*args, **kwargs):
            return original_test(*args, **kwargs, test_count=2)
        with patch.object(engine, 'test_experiment', side_effect=small_test), \
                contextlib.redirect_stdout(io.StringIO()):
            result = train.main(['--seed', str(_test_seed('TEST_SEED_C')), '--runs', '1', '--shots', '1', '5',
                                 '--epochs', '1', '--train-episodes', '1', '--val-episodes', '1',
                                 '--test-every-epochs', '0', '--monitor-test-episodes', '1',
                                 '--cache-dir', str(self.cache),
                                 '--output-dir', str(self.root / 'both'),
                                 '--checkpoint-dir', str(self.root / 'both_weights'), '--device', 'cpu'])
        self.assertEqual(set(result['by_shot']), {'1', '5'})
        for stats in result['by_shot'].values():
            self.assertEqual(stats['n'], 1)
            self.assertIsNone(stats['sample_variance_pp2'])

    def test_configuration_drives_model_training_and_test(self):
        config = load_config()
        config['experiment'].update(seeds=[_test_seed('TEST_SEED_H')], runs=1, shots=[1])
        config['data'].update(cache_dir=str(self.cache), class_counts=[28, 11, 11],
                              normalization='already_normalized')
        config['model'].update(channels=16, attention_channels=4, dgfm_hidden_dim=28,
                               lambda_init=[0.3, 0.5, 0.9], scale_init=7.0,
                               batch_norm_momentum=0.2)
        config['episode'].update(way=3, query_per_class=2)
        config['training'].update(epochs=2, patience=3, episodes_per_epoch=2)
        config['training']['optimizer'].update(learning_rate=0.0003, weight_decay=0.01,
                                               betas=[0.8, 0.95], eps=1e-7)
        config['training']['scheduler'].update(step_size=1, gamma=0.5)
        config['validation'].update(episodes=2, sampling='fixed')
        config['testing'].update(episodes=3, every_epochs=1, start_epoch=2, periodic_episodes=2)
        config['runtime'].update(device='cpu', cpu_threads=2)
        config['output'].update(results_dir=str(self.root / 'configured'),
                                 checkpoints_dir=str(self.root / 'configured_weights'))
        path = self.root / 'experiment.json'
        write_json(path, config)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.main(['--config', str(path)])
        batch = next((self.root / 'configured').glob('b_*'))
        manifest = read_json(batch / 'batch.json')
        self.assertEqual(manifest['run_seeds'], [_test_seed('TEST_SEED_H')])
        saved = read_json(batch / 'config.json')
        self.assertEqual(saved, read_json(batch / 'sources/configs/mcg_net.json'))
        self.assertEqual(saved, manifest['configuration'])
        self.assertEqual(manifest['settings']['test_episodes'], 3)
        plan = manifest['plans'][0]
        self.assertEqual([len(plan['split'][key]) for key in
                          ('train_classes', 'validation_classes', 'test_classes')], [28, 11, 11])
        spec = read_json(batch / plan['directory'] / 'experiment.json')
        self.assertEqual(spec['model']['scale_init'], 7.0)
        self.assertEqual(spec['reproducibility'], {
            'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8'})
        self.assertEqual(spec['data']['normalization'], 'already_normalized')
        checkpoint = engine.load_torch(Path(spec['checkpoint_directory']) / 's1_r01/last.pt')
        self.assertEqual(checkpoint['model']['encoder_dt.mapping.0.weight'].shape[0], 16)
        self.assertEqual(checkpoint['model']['fusion.guide.0.weight'].shape[0], 28)
        optimizer = checkpoint['optimizer']['param_groups'][0]
        self.assertEqual(optimizer['betas'], (0.8, 0.95))
        self.assertEqual(optimizer['weight_decay'], 0.01)
        self.assertEqual(optimizer['eps'], 1e-7)
        self.assertEqual([row['learning_rate'] for row in checkpoint['history']], [0.0003, 0.00015])
        self.assertIsNone(checkpoint['history'][0]['monitor_test_accuracy_0_1'])
        self.assertIsNotNone(checkpoint['history'][1]['monitor_test_accuracy_0_1'])
        record = read_json(batch / plan['directory'] / 'round_result.json')
        test = (batch / record['test_summary_path']).parent
        with np.load(test / 's1_r01_queries.npz') as queries:
            self.assertEqual(queries['logits'].shape, (3, 6, 3))
        self.assertEqual(sum(row['query_count'] for row in
                             read_json(batch / 'analysis/s1_per_class.json')['per_class']), 18)
        write_json(path, {'changed_for_next_experiment': True})
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = train.main(['--resume', str(batch)])
            tested = train.main(['--test-only', str(batch)])
        self.assertEqual(result['runs'], resumed['runs'])
        self.assertEqual(result['runs'], tested['runs'])
        args = train.parser().parse_args(['--runs', '2', '--seeds', str(np.iinfo(np.uint32).min), str(np.iinfo(np.uint32).max),
                                         '--shots', '5', '--test-episodes', '4',
                                         '--monitor-start-epoch', '12'])
        overrides = train.apply_overrides(load_config(), args)
        self.assertEqual(overrides['experiment']['runs'], 2)
        self.assertEqual(overrides['experiment']['seeds'], [int(np.iinfo(np.uint32).min), int(np.iinfo(np.uint32).max)])
        self.assertEqual(overrides['experiment']['shots'], [5])
        self.assertEqual(overrides['testing']['episodes'], 4)
        self.assertEqual(overrides['testing']['start_epoch'], 12)
        invalid = load_config()
        invalid['training']['optimizer']['learning_ratte'] = 0.001
        write_json(path, invalid)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            load_config(path)

    def test_multiple_shots_and_sample_capacity(self):
        shots = [1, 5, 10, 20]
        config = load_config()
        config['experiment'].update(runs=1, seeds=[_test_seed('TEST_SEED_C')], shots=shots)
        config['data']['cache_dir'] = str(self.root / 'shot_cache')
        cache = Path(config['data']['cache_dir'])
        cache.mkdir()
        for name in ('labels', 'dt', 'at'):
            np.save(cache / (name + '.npy'), np.repeat(
                np.load(self.cache / (name + '.npy')), 2, axis=0))
        write_json(cache / 'label_mapping.json', read_json(self.cache / 'label_mapping.json'))
        config['model'].update(channels=16, attention_channels=4)
        config['episode'].update(way=3, query_per_class=2)
        config['training'].update(epochs=1, episodes_per_epoch=1)
        config['validation']['episodes'] = 1
        config['testing'].update(episodes=1, start_epoch=50)
        config['runtime'].update(device='cpu', cpu_threads=2)
        config['output'].update(results_dir=str(self.root / 'multiple_shots'),
                                 checkpoints_dir=str(self.root / 'multiple_shot_weights'))
        path = self.root / 'multiple_shots.json'
        write_json(path, config)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.main(['--config', str(path)])
        self.assertEqual(set(result['by_shot']), set(map(str, shots)))
        self.assertEqual([row['shot'] for row in result['runs']], shots)
        batch = next((self.root / 'multiple_shots').glob('b_*'))
        plan = read_json(batch / 'batch.json')['plans'][0]
        spec = read_json(batch / plan['directory'] / 'experiment.json')
        for shot in shots:
            checkpoint = Path(spec['checkpoint_directory']) / ('s%d_r01' % shot) / 'best.pt'
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(result['by_shot'][str(shot)]['n'], 1)
            self.assertTrue((batch / 'analysis' / ('s%d_per_class.json' % shot)).is_file())
        args = train.parser().parse_args(['--shots', '1', '5', '10', '20'])
        self.assertEqual(train.apply_overrides(load_config(), args)['experiment']['shots'], shots)
        for values in ([], [0], [-1], [1, 1], [True], [1.5], ['5'], None):
            with self.subTest(shots=values), self.assertRaisesRegex(ValueError, 'shots'):
                engine.TrainingConfig(shots=values)
        config['experiment']['shots'] = [39]
        write_json(path, config)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'samples'):
            train.main(['--config', str(path), '--check-plan'])

    def test_best_validation_test_trigger_and_training_state(self):
        data = RadarCache(self.cache)
        split = class_split(data.labels, seed=_test_seed('TEST_SEED_C'))
        original_accuracy = engine.episode_accuracy

        def run_mode(interval, start_epoch=1, resume_at=None):
            validation_scores = iter((0.5, 0.4, 0.6, 0.6))
            seen_validation, monitor_epochs = [], []

            def observed(model, cache, plan, shot, device, loss_values=None, details=None):
                values = original_accuracy(model, cache, plan, shot, device,
                                           loss_values=loss_values, details=details)
                if loss_values is not None:
                    accuracy = next(validation_scores)
                    seen_validation.append(accuracy)
                    return np.full(len(plan), accuracy)
                if details is None and np.isin(cache.labels[plan], split['test_classes']).all():
                    monitor_epochs.append(len(seen_validation))
                return values

            name = '%d_%d_%s' % (interval, start_epoch, resume_at)
            directory = self.root / ('trigger_' + name)
            weights = self.root / ('trigger_weights_' + name)
            tc = engine.TrainingConfig(epochs=4, shots=(1,), seed=_test_seed('TEST_SEED_C'), evaluation_seed=_test_seed('TEST_SEED_C'),
                                       train_episodes=1, val_episodes=1, test_episodes=2,
                                       test_every_epochs=interval, monitor_start_epoch=start_epoch,
                                       monitor_test_episodes=2)
            with patch.object(engine, 'episode_accuracy', side_effect=observed), \
                    contextlib.redirect_stdout(io.StringIO()):
                mc = ModelConfig(channels=16, attention_channels=4)
                if resume_at is not None:
                    incomplete = engine.train_experiment(
                        data, split, mc, tc, directory, torch.device('cpu'),
                        checkpoint_directory=weights, stop_after_epoch=resume_at)
                    self.assertIsNone(incomplete)
                engine.train_experiment(data, split, mc, tc, directory, torch.device('cpu'),
                                        checkpoint_directory=weights, resume=resume_at is not None)
                tested = engine.test_experiment(data, directory, torch.device('cpu'), directory / 'final')
            checkpoint = engine.load_torch(weights / 's1_r01/last.pt')
            for row in checkpoint['history']:
                if row['monitor_test_trigger'] is not None:
                    with np.load(directory / 's1_r01/plans' / ('monitor_test_epoch_%03d.npz' % row['epoch'])) as saved:
                        statistics = engine.test_episode_statistics(saved['accuracies'])
                    for field in ('sample_variance_pp2', 'sample_sd_pp', 'mean_95ci_half_width_pp',
                                  'mean_95ci_lower_percent', 'mean_95ci_upper_percent'):
                        self.assertEqual(row['monitor_test_' + field], statistics[field])
            self.assertEqual(checkpoint['best_epoch'], 3)
            self.assertEqual(tested['runs'][0]['checkpoint_epoch'], 3)
            self.assertEqual(tested['runs'][0]['episode_statistics']['n'], 2)
            self.assertEqual([row['validation_improved'] for row in checkpoint['history']],
                             [True, False, True, False])
            self.assertEqual(checkpoint['history'][-1]['stale_epochs'], 1)
            return checkpoint, monitor_epochs

        best, best_epochs = run_mode(0)
        interval, interval_epochs = run_mode(1)
        self.assertEqual(best_epochs, [1, 3])
        self.assertEqual(interval_epochs, [1, 2, 3, 4])
        self.assertEqual([row['monitor_test_trigger'] for row in best['history']],
                         ['best_validation', None, 'best_validation', None])
        self.assertIsNone(best['history'][1]['monitor_test_sample_variance_pp2'])
        self.assertIsNotNone(best['history'][0]['monitor_test_sample_variance_pp2'])
        self.assertEqual([row['monitor_test_trigger'] for row in interval['history']], ['interval'] * 4)
        self.assertTrue(all(torch.equal(best['model'][key], interval['model'][key])
                            for key in best['model']))
        delayed, delayed_epochs = run_mode(0, start_epoch=3)
        interval_delayed, interval_delayed_epochs = run_mode(1, start_epoch=3)
        final_only, final_only_epochs = run_mode(0, start_epoch=5)
        resumed, resumed_epochs = run_mode(0, start_epoch=3, resume_at=2)
        self.assertEqual(delayed_epochs, [3])
        self.assertEqual(interval_delayed_epochs, [3, 4])
        self.assertEqual(final_only_epochs, [])
        self.assertEqual(resumed_epochs, [3])
        self.assertEqual(delayed['history'], resumed['history'])
        for result in (delayed, interval_delayed, final_only, resumed):
            self.assertTrue(all(torch.equal(best['model'][key], result['model'][key])
                                for key in best['model']))


if __name__ == '__main__':
    unittest.main()
