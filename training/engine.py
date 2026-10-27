"""MCG-Net training, validation and testing."""
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models import MCGNet, ModelConfig, episode_scores
from analysis.reports import save_query_artifacts
from data.radar_cache import SPLIT_KEYS, episode_plan, file_hash, plan_seed, write_json, read_json


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    patience: int
    runs: int
    train_episodes: int
    val_episodes: int
    test_episodes: int
    shots: tuple
    seed: int
    evaluation_seed: int
    validation_plan: str
    test_every_epochs: int
    monitor_start_epoch: int
    monitor_test_episodes: int
    way: int
    query_per_class: int
    learning_rate: float
    weight_decay: float
    adam_betas: tuple
    adam_eps: float
    scheduler_step_size: int
    scheduler_gamma: float

    def __post_init__(self):
        if (not isinstance(self.shots, (list, tuple)) or not self.shots
                or any(type(s) is not int or s < 1 for s in self.shots)
                or len(set(self.shots)) != len(self.shots)):
            raise ValueError('shots must contain distinct positive integers')
        object.__setattr__(self, 'shots', tuple(self.shots))
        object.__setattr__(self, 'adam_betas', tuple(self.adam_betas))
        for key in ('epochs', 'runs', 'train_episodes', 'val_episodes', 'test_episodes',
                    'monitor_test_episodes', 'monitor_start_epoch', 'way', 'query_per_class',
                    'scheduler_step_size'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + ' must be a positive integer')
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError('patience must be a nonnegative integer (0 disables early stopping)')
        if type(self.test_every_epochs) is not int or self.test_every_epochs < 0:
            raise ValueError('test_every_epochs must be nonnegative (0 tests on validation improvement)')
        if self.validation_plan not in ('fixed', 'per_epoch'):
            raise ValueError('validation_plan must be fixed or per_epoch')
        if min(self.seed, self.evaluation_seed) < 0:
            raise ValueError('seeds must be nonnegative')
        if self.runs != 1:
            raise ValueError('Each repetition has one training run per shot; use train.py --runs N')
        if self.seed != self.evaluation_seed:
            raise ValueError('A repetition must share one seed for splitting, training and evaluation')
        for key in ('learning_rate', 'adam_eps', 'scheduler_gamma'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not 0 < value < float('inf'):
                raise ValueError(key + ' must be finite and positive')
        if type(self.weight_decay) not in (int, float) or not 0 <= self.weight_decay < float('inf'):
            raise ValueError('weight_decay must be finite and nonnegative')
        if len(self.adam_betas) != 2 or any(type(value) not in (int, float)
                                          or not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError('adam_betas must contain two numbers in [0,1)')


LOCKED = {'encoder_dim': 112, 'fusion_dim': 224, 'optimizer': 'Adam', 'scheduler': 'StepLR',
          'cosine_scale': {'mode': 'global_learnable', 'parameterization': 'unconstrained_scalar'},
          'subject_filtering': False,
          'support_query_sample_overlap': False, 'evaluation_bn': 'training_running_statistics',
          'level_weight_constraint': 'unconstrained_learnable_scalars'}


PROJECT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    'train.py', 'test.py', 'configuration.py', 'models/__init__.py', 'models/config.py', 'models/mcg_net.py',
    'models/tpa_re.py', 'models/dgfm.py', 'models/cpc.py', 'data/__init__.py',
    'data/radar_cache.py', 'data/test_cache.py', 'training/__init__.py', 'training/engine.py',
    'training/evaluation.py',
    'analysis/__init__.py', 'analysis/reports.py',
)


def source_hashes():
    return {name: file_hash(PROJECT / name) for name in SOURCE_FILES}


def snapshot_sources(directory, configuration=None):
    directory = Path(directory)
    for name in SOURCE_FILES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT / name).read_bytes())
    target = directory / 'configs' / 'mcg_net.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, configuration if configuration is not None
               else read_json(PROJECT / 'configs' / 'mcg_net.json'))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_determinism(deterministic_algorithms, cublas_workspace_config):
    """Configure CUDA determinism before creating CUDA tensors or library handles."""
    if type(deterministic_algorithms) is not bool:
        raise ValueError('deterministic_algorithms must be a boolean')
    if cublas_workspace_config not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported CUBLAS_WORKSPACE_CONFIG')
    if (torch.cuda.is_initialized()
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != cublas_workspace_config):
        raise ValueError('CUDA is already initialized with a different CUBLAS_WORKSPACE_CONFIG; '
                         'start a fresh Python process')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, values, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
    temporary.replace(path)


def load_torch(path, device='cpu'):
    # Training checkpoints contain model, optimizer and RNG states.
    with Path(path).open('rb') as handle:
        return torch.load(handle, map_location=device, weights_only=False)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def episode_accuracy(model, data, plan, shot, device, loss_values=None, details=None):
    model.eval()
    values = []
    way, queries = plan.shape[1], plan.shape[2] - shot
    target = torch.arange(way, device=device).repeat_interleave(queries)
    with torch.no_grad():
        for indices in plan:
            dt, at = data.batch(indices.reshape(-1), device)
            scores = episode_scores(model, dt, at, way, shot, queries)
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite validation/test scores')
            predictions = scores.argmax(1)
            values.append(float((predictions == target).float().mean().item()))
            if loss_values is not None:
                loss_values.append(float(F.cross_entropy(scores, target).item()))
            if details is not None:
                details['logits'].append(scores.detach().cpu().numpy().copy())
                details['predicted_local'].append(predictions.detach().cpu().numpy().copy())
    return np.asarray(values, dtype=np.float64)


def make_spec(data_description, split, model_config, training_config):
    if split['class_split_seed'] != training_config.seed:
        raise ValueError('Class split seed must equal this repetition training/evaluation seed')
    return {'format_version': 1, 'model': model_config.to_dict(),
            'training': asdict(training_config), 'locked': LOCKED,
            'data': data_description, 'split': split, 'source_sha256': source_hashes(),
            'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
            'reproducibility': {
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
            'protocol': 'class_disjoint_no_subject_restriction'}


def _train_run(data, split, mc, tc, directory, shot, run, spec_id, device,
               resume=False, stop_after_epoch=None, checkpoint_directory=None, inference=None):
    directory.mkdir(parents=True, exist_ok=True)
    completed_path = directory / 'run_result.json'
    if completed_path.is_file():
        result = read_json(completed_path)
        if not resume or result['spec_sha256'] != spec_id:
            raise ValueError('Run already exists or specification differs')
        return result
    checkpoint_directory = Path(checkpoint_directory or directory / 'checkpoints')
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_directory / 'last.pt', checkpoint_directory / 'best.pt'
    if last_path.exists() and not resume:
        raise ValueError('Unfinished run exists; use --resume')
    seed_all(tc.seed)  # Initialize training with the run seed.
    model = MCGNet(mc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay,
                                  betas=tc.adam_betas, eps=tc.adam_eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tc.scheduler_step_size,
                                               gamma=tc.scheduler_gamma)
    start, best_val, best_epoch, stale, best_state = 1, -1.0, 0, 0, None
    rows = []
    if last_path.exists():
        last = load_torch(last_path, device)
        if last['spec_sha256'] != spec_id:
            raise ValueError('Checkpoint specification mismatch')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler'])
        start = last['epoch'] + 1
        best_val, best_epoch, stale = last['best_validation'], last['best_epoch'], last['stale']
        best_state, rows = last['best_state'], last['history']
        restore_rng(last['rng'])
        save_torch(best_path, {'model': best_state, 'epoch': best_epoch,
                              'validation_accuracy': best_val, 'spec_sha256': spec_id,
                              'inference': inference})
    target = torch.arange(tc.way, device=device).repeat_interleave(tc.query_per_class)
    plan_dir = directory / 'plans'
    plan_dir.mkdir(exist_ok=True)
    should_stop = tc.patience > 0 and stale >= tc.patience
    for epoch in range(start, tc.epochs + 1):
        if should_stop:
            break
        train_seed = plan_seed(tc.seed, run, shot, 1, epoch)
        training = episode_plan(data.labels, split['train_classes'], tc.way, shot, tc.query_per_class,
                                tc.train_episodes, train_seed)
        val_epoch = epoch if tc.validation_plan == 'per_epoch' else 0
        validation = episode_plan(data.labels, split['validation_classes'], tc.way, shot, tc.query_per_class,
                                  tc.val_episodes, plan_seed(tc.evaluation_seed, run, shot, 2, val_epoch))
        np.savez_compressed(plan_dir / ('epoch_%03d.npz' % epoch), training=training, validation=validation)
        model.train()
        losses, accuracies = [], []
        used_lr = optimizer.param_groups[0]['lr']
        for indices in training:
            dt, at = data.batch(indices.reshape(-1), device)
            scores = episode_scores(model, dt, at, tc.way, shot, tc.query_per_class)
            loss = F.cross_entropy(scores, target)
            if not torch.isfinite(loss) or not torch.isfinite(scores).all():
                raise ValueError('Non-finite training scores/loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite gradient')
            optimizer.step()
            losses.append(float(loss.item()))
            accuracies.append(float((scores.argmax(1) == target).float().mean().item()))
        scheduler.step()
        val_losses = []
        val_acc = float(episode_accuracy(model, data, validation, shot, device,
                                         loss_values=val_losses).mean())
        validation_improved = val_acc > best_val
        if validation_improved:  # Select by validation accuracy; retain the earlier epoch on ties.
            best_val, best_epoch, stale = val_acc, epoch, 0
            best_state = cpu_state(model)
            save_torch(best_path, {'model': best_state, 'epoch': epoch,
                                  'validation_accuracy': best_val, 'spec_sha256': spec_id,
                                  'inference': inference})
        else:
            stale += 1
        monitor_accuracy, monitor_seed, monitor_statistics = None, None, None
        monitor_trigger = None
        if epoch >= tc.monitor_start_epoch:
            if tc.test_every_epochs == 0 and validation_improved:
                monitor_trigger = 'best_validation'
            elif tc.test_every_epochs > 0 and epoch % tc.test_every_epochs == 0:
                monitor_trigger = 'interval'
        if monitor_trigger is not None:
            monitor_seed = plan_seed(tc.evaluation_seed, run, shot, 5, epoch)
            monitoring = episode_plan(data.labels, split['test_classes'], tc.way, shot, tc.query_per_class,
                                      tc.monitor_test_episodes, monitor_seed)
            # Preserve RNG state and model mode during periodic evaluation.
            saved_rng, saved_mode = rng_state(), model.training
            try:
                monitor_values = episode_accuracy(model, data, monitoring, shot, device)
            finally:
                restore_rng(saved_rng)
                model.train(saved_mode)
            monitor_accuracy = float(monitor_values.mean())
            monitor_statistics = test_episode_statistics(monitor_values)
            np.savez_compressed(plan_dir / ('monitor_test_epoch_%03d.npz' % epoch),
                                indices=monitoring, accuracies=monitor_values)
        rows.append({'epoch': epoch, 'loss': float(np.mean(losses)),
                     'train_accuracy': float(np.mean(accuracies)), 'validation_accuracy': val_acc,
                     'learning_rate': used_lr, 'stale_epochs': stale,
                     'training_plan_seed': train_seed,
                     'cosine_scale': float(model.scale.detach().cpu()),
                     'validation_loss': float(np.mean(val_losses)),
                     'validation_improved': validation_improved,
                     'monitor_test_accuracy_0_1': monitor_accuracy,
                     'monitor_test_plan_seed': monitor_seed,
                     'monitor_test_trigger': monitor_trigger})
        for name in ('sample_variance_pp2', 'sample_sd_pp', 'mean_95ci_half_width_pp',
                     'mean_95ci_lower_percent', 'mean_95ci_upper_percent'):
            rows[-1]['monitor_test_' + name] = (monitor_statistics[name]
                                               if monitor_statistics is not None else None)
        for modality, encoder in (('dt', model.encoder_dt), ('at', model.encoder_at)):
            weights = (model.encoder_dt if encoder is None else encoder).level_weights.detach().cpu().tolist()
            rows[-1].update({'lambda_%s_%d' % (modality, i + 1): value for i, value in enumerate(weights)})
        save_torch(last_path, {'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                              'epoch': epoch, 'best_validation': best_val, 'best_epoch': best_epoch,
                              'best_state': best_state, 'stale': stale, 'history': rows,
                              'spec_sha256': spec_id, 'inference': inference})
        with (directory / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('shot=%d run=%02d epoch=%03d loss=%.4f val_loss=%.4f train=%.2f%% val=%.2f%% best=%.2f%% scale=%.4f' %
              (shot, run + 1, epoch, rows[-1]['loss'], rows[-1]['validation_loss'], rows[-1]['train_accuracy'] * 100,
               val_acc * 100, best_val * 100, rows[-1]['cosine_scale']), flush=True)
        if monitor_accuracy is not None:
            print_episode_statistics(monitor_statistics, 'shot=%d run=%02d epoch=%03d %s test' %
                                      (shot, run + 1, epoch, monitor_trigger))
        should_stop = tc.patience > 0 and stale >= tc.patience
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return None
    if best_state is None:
        raise ValueError('No trained checkpoint')
    # Evaluate the selected checkpoint on an independent validation panel.
    confirmation = episode_plan(data.labels, split['validation_classes'], tc.way, shot, tc.query_per_class,
                                tc.val_episodes, plan_seed(tc.evaluation_seed, run, shot, 3))
    np.savez_compressed(directory / 'selection_validation.npz', indices=confirmation)
    model.load_state_dict(best_state)
    selection_accuracy = float(episode_accuracy(model, data, confirmation, shot, device).mean())
    result = {'spec_sha256': spec_id, 'shot': shot, 'run': run + 1,
              'training_seed': tc.seed, 'class_split_seed': tc.seed,
              'evaluation_seed': tc.evaluation_seed, 'best_epoch': best_epoch,
              'epochs_completed': rows[-1]['epoch'], 'early_stopped': should_stop,
              'best_validation_accuracy': best_val, 'selection_validation_accuracy': selection_accuracy,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'statistical_unit': 'one_independent_split_and_training_run',
              'test_evaluated': any(row.get('monitor_test_accuracy_0_1') is not None for row in rows),
              'test_used_for_selection': False, 'final_test_evaluated': False,
              'learned_cosine_scale': float(model.scale.detach().cpu()),
              'learned_level_weights_dt': model.encoder_dt.level_weights.detach().cpu().tolist(),
              'learned_level_weights_at': (None if model.encoder_at is None else
                                         model.encoder_at.level_weights.detach().cpu().tolist())}
    write_json(completed_path, result)
    return result


def train_experiment(data, split, mc, tc, directory, device, resume=False,
                     data_description=None, stop_after_epoch=None, checkpoint_directory=None,
                     configuration=None):
    runtime = configuration['runtime']
    configure_determinism(runtime['deterministic_algorithms'], runtime['cublas_workspace_config'])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = make_spec(data_description or data.describe(hashes=True), split, mc, tc)
    checkpoint_directory = Path(checkpoint_directory or directory / 'checkpoints').resolve()
    spec['checkpoint_directory'] = str(checkpoint_directory)
    spec_id = signature(spec)
    path = directory / 'experiment.json'
    if path.exists():
        if not resume or signature(read_json(path)) != spec_id:
            raise ValueError('Existing experiment specification differs or --resume missing')
    else:
        write_json(path, spec)
        snapshots = directory / 'sources'
        snapshots.mkdir(exist_ok=True)
        snapshot_sources(snapshots, configuration)
    # Check class pools for the requested support and query sizes.
    for shot in tc.shots:
        for key in SPLIT_KEYS:
            episode_plan(data.labels, split[key], tc.way, shot, tc.query_per_class, 1, 0)
    results = []
    from training.evaluation import inference_settings
    for shot in tc.shots:
        for run in range(tc.runs):
            result = _train_run(data, split, mc, tc, directory / ('s%d_r%02d' % (shot, run + 1)),
                                shot, run, spec_id, device, resume, stop_after_epoch,
                                checkpoint_directory / ('s%d_r%02d' % (shot, run + 1)),
                                inference_settings(data, split, mc, tc, shot, run, spec))
            if result is None:
                return None
            results.append(result)
    by_shot = {str(shot): float(np.mean([r['selection_validation_accuracy'] for r in results
                                        if r['shot'] == shot])) for shot in tc.shots}
    summary = {'spec_sha256': spec_id, 'complete': True, 'runs': results,
               'selection_metric': 'equal_weight_mean_of_shot_specific_run_mean_validation_accuracies',
               'selection_validation_by_shot': by_shot,
               'selection_score': float(np.mean(list(by_shot.values()))),
               'parameter_count': results[0]['parameter_count'],
               'test_evaluated': any(r['test_evaluated'] for r in results),
               'test_used_for_selection': False,
               'final_test_evaluated': all(r.get('final_test_evaluated', False) for r in results)}
    write_json(directory / 'training_summary.json', summary)
    return summary


def accuracy_statistics(values, unit):
    values = np.asarray(values, dtype=np.float64)
    return {'statistical_unit': unit, 'n': len(values), 'mean_accuracy_percent': float(values.mean() * 100),
            'sample_variance_pp2': float(values.var(ddof=1) * 10000) if len(values) > 1 else None,
            'sample_sd_pp': float(values.std(ddof=1) * 100) if len(values) > 1 else None,
            'ddof': 1 if len(values) > 1 else None}


def test_episode_statistics(values):
    """Accuracy dispersion and a normal-approximation CI over test episodes."""
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError('Episode accuracies must be a nonempty finite array in [0,1]')
    statistics = accuracy_statistics(values, 'test_episode')
    half_width = (1.96 * statistics['sample_sd_pp'] / np.sqrt(len(values))
                  if len(values) > 1 else None)
    mean = statistics['mean_accuracy_percent']
    statistics.update(
        mean_accuracy_0_1=float(values.mean()),
        sample_variance_0_1=float(values.var(ddof=1)) if len(values) > 1 else None,
        mean_95ci_half_width_pp=float(half_width) if half_width is not None else None,
        mean_95ci_lower_percent=float(mean - half_width) if half_width is not None else None,
        mean_95ci_upper_percent=float(mean + half_width) if half_width is not None else None,
        confidence_level=0.95, confidence_interval_method='normal_approximation_over_test_episodes')
    return statistics


def print_episode_statistics(statistics, label='Test'):
    """Print one summary for the completed set of test episodes."""
    def number(value, digits=6):
        return 'N/A' if value is None else ('%.*f' % (digits, value))
    mean = statistics['mean_accuracy_percent']
    print('%s | Episodes: %d | Mean accuracy: %.4f%%' % (label, statistics['n'], mean), flush=True)
    print('  Sample variance: %s pp^2 (ddof=%s) | Sample SD: %s pp' %
          (number(statistics['sample_variance_pp2']), statistics['ddof'],
           number(statistics['sample_sd_pp'])), flush=True)
    print('  Mean accuracy +/- 95%% CI: %.4f%% +/- %s pp | Interval: [%s%%, %s%%]' %
          (mean, number(statistics['mean_95ci_half_width_pp'], 4),
           number(statistics['mean_95ci_lower_percent'], 4),
           number(statistics['mean_95ci_upper_percent'], 4)), flush=True)


def test_experiment(data, directory, device, output_dir, test_count=None):
    directory, output_dir = Path(directory), Path(output_dir)
    spec = read_json(directory / 'experiment.json')
    configure_determinism(**spec['reproducibility'])
    summary = read_json(directory / 'training_summary.json')
    if not summary['complete'] or summary['spec_sha256'] != signature(spec):
        raise ValueError('Incomplete experiment or specification mismatch')
    if source_hashes() != spec['source_sha256']:
        raise ValueError('Implementation changed since training; use the saved source snapshot')
    if data.describe(hashes=True)['sha256'] != spec['data']['sha256']:
        raise ValueError('Cache differs from the training cache')
    if data.normalization != spec['data']['normalization']:
        raise ValueError('Normalization differs from training')
    output_dir.mkdir(parents=True, exist_ok=False)
    mc, tc = ModelConfig(**spec['model']), TrainingConfig(**spec['training'])
    count = tc.test_episodes if test_count is None else test_count
    rows = []
    for shot in tc.shots:
        for run in range(tc.runs):
            sub = directory / ('s%d_r%02d' % (shot, run + 1))
            checkpoint = load_torch(Path(spec['checkpoint_directory']) / ('s%d_r%02d' % (shot, run + 1)) / 'best.pt')
            if checkpoint['spec_sha256'] != summary['spec_sha256']:
                raise ValueError('Checkpoint specification mismatch')
            model = MCGNet(mc).to(device)
            model.load_state_dict(checkpoint['model'])
            plan = episode_plan(data.labels, spec['split']['test_classes'], tc.way, shot, tc.query_per_class,
                                count, plan_seed(tc.evaluation_seed, run, shot, 4))
            details = {'logits': [], 'predicted_local': []}
            accuracies = episode_accuracy(model, data, plan, shot, device, details=details)
            name = 's%d_r%02d' % (shot, run + 1)
            analysis = save_query_artifacts(
                data, plan, shot, np.stack(details['logits']), np.stack(details['predicted_local']),
                output_dir, name, {'seed': tc.seed, 'checkpoint_epoch': checkpoint['epoch'],
                                   'learned_cosine_scale': float(model.scale.detach().cpu()),
                                   'spec_sha256': summary['spec_sha256']}, spec['split']['test_classes'])
            np.savez_compressed(output_dir / (name + '_episodes.npz'), indices=plan, accuracies=accuracies)
            with (output_dir / (name + '.csv')).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['episode', 'accuracy_0_1'])
                writer.writerows(enumerate(accuracies.tolist(), 1))
            row = {'shot': shot, 'run': run + 1, 'checkpoint_epoch': checkpoint['epoch'],
                   'learned_cosine_scale': float(model.scale.detach().cpu()),
                   'class_analysis': analysis,
                   'episode_statistics': test_episode_statistics(accuracies),
                   'run_mean_accuracy': float(accuracies.mean())}
            rows.append(row)
            print_episode_statistics(row['episode_statistics'], name + ' final test')
            print('Query predictions/logits and per-class reports:', output_dir / (name + '_queries.npz'),
                  flush=True)
    result = {'experiment': str(directory.resolve()), 'spec_sha256': summary['spec_sha256'],
              'source_sha256_at_evaluation': source_hashes(), 'test_episodes_per_run': count,
              'evaluation_budget_override': test_count is not None, 'runs': rows,
              'by_shot': {str(shot): accuracy_statistics(
                  [r['run_mean_accuracy'] for r in rows if r['shot'] == shot],
                  'mean_test_accuracy_per_split_and_training_run') for shot in tc.shots},
              'protocol': 'class_disjoint_no_subject_restriction'}
    write_json(output_dir / 'test_summary.json', result)
    for row in rows:
        run_path = directory / ('s%d_r%02d' % (row['shot'], row['run'])) / 'run_result.json'
        saved = read_json(run_path)
        saved.update(final_test_evaluated=True, test_evaluated=True,
                     final_test_summary=str((output_dir / 'test_summary.json').resolve()))
        write_json(run_path, saved)
        for item in summary['runs']:
            if item['shot'] == row['shot'] and item['run'] == row['run']:
                item.update(final_test_evaluated=True, test_evaluated=True,
                            final_test_summary=saved['final_test_summary'])
    summary.update(final_test_evaluated=True, test_evaluated=True)
    write_json(directory / 'training_summary.json', summary)
    return result
