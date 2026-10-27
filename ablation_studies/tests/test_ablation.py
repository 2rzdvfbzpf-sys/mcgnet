"""Model, sampling, CCA and checkpoint consistency checks."""

import os as _seed_environment

def _test_seed(name):
    value = _seed_environment.environ.get(name)
    if value is None:
        raise unittest.SkipTest("Set explicit test seed environment variables")
    return int(value)

from pathlib import Path
import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap
import numpy as np
import torch

from data.radar_cache import RadarCache, class_split
from models import MCGNet, ModelConfig
from networks import DEFAULT_VARIANTS, build_model, scores, fit_cca
from reporting import statistics
from runner import (runtime, seed_all, train_one, fit_cca_trial, cache_description,
                    load_checkpoint, checkpoint_model, task_plan, evaluate)
from settings import HERE, read_json, validate_config
from train import execution_order


def configuration():
    config = read_json(HERE / 'config.json')
    config['experiment'] = {'runs': 1, 'seeds': [_test_seed('TEST_SEED_E')], 'shots': [1]}
    config['model'].update(channels=4, attention_channels=2)
    config['ablation']['resnet12_channels'] = [4]*4
    config['ablation']['cca'].update(dimensions=[2, 4], feature_batch_size=32)
    config['episode'] = {'way': 2, 'query_per_class': 1}
    config['training'].update(epochs=2, episodes_per_epoch=1, patience=30)
    config['validation']['episodes'] = 2
    config['testing'].update(episodes=3, periodic_episodes=2, start_epoch=50)
    config['runtime'].update(device='cpu', cpu_threads=1)
    return validate_config(config)


def fixture(directory):
    directory.mkdir()
    rng = np.random.default_rng(_test_seed('TEST_SEED_A'))
    labels = np.repeat(np.arange(50), 4).astype(np.int64)
    np.save(directory / 'labels.npy', labels)
    for name in ('dt', 'at'):
        np.save(directory / (name+'.npy'), rng.random((len(labels), 25, 64), dtype=np.float32))
    (directory / 'label_mapping.json').write_text(json.dumps({str(i+1): i for i in range(50)}), encoding='utf-8')
    return RadarCache(directory)


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        runtime(cls.config)

    def test_full_model_matches_existing_network(self):
        seed_all(_test_seed('TEST_SEED_B'))
        expected = MCGNet(ModelConfig(**self.config['model'])).eval()
        seed_all(_test_seed('TEST_SEED_B'))
        actual = build_model('tpa_re_dgfm', self.config['model'], self.config['ablation']).eval()
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[key]), key)
        dt, at = torch.randn(8,1,25,64), torch.randn(8,1,25,64)
        self.assertTrue(torch.equal(expected(dt, at), actual(dt, at)))

    def test_models_have_finite_cpc_gradients(self):
        for variant in [v for v in DEFAULT_VARIANTS if v != 'tpa_re_cca']:
            with self.subTest(variant=variant):
                seed_all(_test_seed('TEST_SEED_D'))
                model = build_model(variant, self.config['model'], self.config['ablation'])
                logits = scores(model, torch.randn(4,1,25,64), torch.randn(4,1,25,64), 2,1,1)
                torch.nn.functional.cross_entropy(logits, torch.arange(2)).backward()
                self.assertEqual(tuple(logits.shape), (2,2))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                if variant == 'tpa_re_weighted':
                    self.assertAlmostEqual(float(model.fusion_weights.softmax(0).sum()), 1.)

    def test_single_stream_ignores_other_representation(self):
        dt, at = torch.randn(6,1,25,64), torch.randn(6,1,25,64)
        for variant in ('dtm_only','atm_only'):
            model = build_model(variant, self.config['model'], self.config['ablation']).eval()
            first = model(dt, at)
            second = model(dt, at*100+31) if variant == 'dtm_only' else model(dt*100+31, at)
            self.assertTrue(torch.equal(first,second))
            self.assertEqual(tuple(first.shape), (6,112))

    def test_linear_cca_is_finite(self):
        rng = np.random.default_rng(_test_seed('TEST_SEED_C'))
        x = rng.normal(size=(40,6))
        y = x @ rng.normal(size=(6,6)) + rng.normal(size=(40,6))*.02
        means, projections, correlations = fit_cca(x,y,.001)
        self.assertTrue(all(np.isfinite(a).all() for a in (*means,*projections,correlations)))
        self.assertGreater(correlations[0],.9)

    def test_statistics_and_dependency_order(self):
        result = statistics([.5,.6,.7])
        self.assertAlmostEqual(result['mean_accuracy_percent'],60.)
        self.assertAlmostEqual(result['ci95_half_width_pp'],1.96*10/np.sqrt(3))
        runs = statistics([.5,.6,.7],'independent_training_run')
        self.assertGreater(runs['ci95_half_width_pp'],result['ci95_half_width_pp'])
        self.assertIsNone(statistics([.5])['ci95_half_width_pp'])
        order, dependencies = execution_order(['tpa_re_cca','dtm_only'])
        self.assertLess(order.index('tpa_re_concat'),order.index('tpa_re_cca'))
        self.assertEqual(dependencies,['tpa_re_concat'])

    def test_resume_is_exact_and_cca_uses_training_only(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('MCG_ABLATION_TEST_ROOT', str(HERE / 'tests'))) as temporary:
            root = Path(temporary)
            data = fixture(root/'cache')
            config = self.config
            split = class_split(data.labels,17)
            description = cache_description(data)
            full_dir, resume_dir = root/'full', root/'resume'
            full_ck, resume_ck = root/'full_ck', root/'resume_ck'
            full = train_one('tpa_re_concat',data,split,config,description,17,1,full_dir,full_ck,torch.device('cpu'))
            paused = train_one('tpa_re_concat',data,split,config,description,17,1,resume_dir,resume_ck,torch.device('cpu'),stop_after=1)
            self.assertIsNone(paused)
            resumed = train_one('tpa_re_concat',data,split,config,description,17,1,resume_dir,resume_ck,torch.device('cpu'),resume=True)
            first, second = load_checkpoint(full_ck/'last.pt'), load_checkpoint(resume_ck/'last.pt')
            self.assertEqual(first['history'],second['history'])
            for key,value in first['model'].items():
                self.assertTrue(torch.equal(value,second['model'][key]),key)
            with np.load(full_dir/'test/test_queries.npz') as a, np.load(resume_dir/'test/test_queries.npz') as b:
                for key in a.files:
                    self.assertTrue(np.array_equal(a[key],b[key]),key)
            cca_dir, cca_ck = root/'cca', root/'cca_ck'
            cca = fit_cca_trial(data,split,config,description,17,1,cca_dir,cca_ck,full_ck/'best.pt',torch.device('cpu'))
            saved = load_checkpoint(cca_ck/'best.pt')
            selected = np.asarray(saved['cca_fitting_sample_indices'])
            self.assertTrue(np.isin(data.labels[selected],split['train_classes']).all())
            self.assertFalse(np.isin(data.labels[selected],split['test_classes']).any())
            model = checkpoint_model(saved,torch.device('cpu'))
            plan = task_plan(data,split['test_classes'],config,17,1,4,3)
            again = evaluate(model,data,plan,1,torch.device('cpu'))
            self.assertEqual(again['accuracy'],cca['test']['accuracy'])
            self.assertEqual(again['loss'],cca['test']['loss'])
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            for array in data.arrays.values():
                array._mmap.close()

    def test_early_stopping_and_global_best_test_gate(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('MCG_ABLATION_TEST_ROOT', str(HERE / 'tests'))) as temporary:
            root = Path(temporary)
            data = fixture(root/'cache')
            config = copy.deepcopy(self.config)
            config['training'].update(epochs=5, patience=2)
            config['testing'].update(start_epoch=2, every_epochs=0)
            split = class_split(data.labels, 17)
            description = cache_description(data)
            def fixed_validation(model, cache, plan, shot, device, directory=None, context=None, classes=None):
                result = evaluate(model, cache, plan, shot, device, directory, context, classes)
                if directory is None:
                    result['accuracy'] = .5
                return result
            with patch('runner.evaluate', side_effect=fixed_validation):
                result = train_one('dtm_only', data, split, config, description, 17, 1,
                    root/'trial', root/'checkpoints', torch.device('cpu'))
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['epochs_completed'], 3)
            self.assertFalse(list((root/'trial').glob('monitor_*.npz')))
            self.assertEqual(load_checkpoint(root/'checkpoints/best.pt')['epoch'], 1)
            self.assertEqual(load_checkpoint(root/'checkpoints/last.pt')['epoch'], 3)
            for array in data.arrays.values():
                array._mmap.close()


if __name__ == '__main__':
    unittest.main()
