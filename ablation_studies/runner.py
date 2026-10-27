"""Independent episodic training, validation and testing for ablations."""
import _bootstrap
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from analysis.reports import save_query_artifacts
from data.radar_cache import episode_plan, plan_seed
from networks import build_model, fit_cca, scores, CCAClassifier
from reporting import print_statistics, statistics, write_csv
from settings import file_hash, read_json, source_hashes, write_json


def runtime(config):
    rt = config['runtime']
    workspace = rt['cublas_workspace_config']
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
        raise RuntimeError('Start a fresh process for the configured CUBLAS workspace')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    torch.use_deterministic_algorithms(rt['deterministic_algorithms'])
    torch.backends.cudnn.deterministic = rt['deterministic_algorithms']
    torch.backends.cudnn.benchmark = False
    if rt['cpu_threads'] is not None:
        torch.set_num_threads(rt['cpu_threads'])
    name = rt['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, verify_source=True):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    if saved.get('format') != 'mcg_ablation_v1':
        raise ValueError('Not an ablation checkpoint')
    if verify_source and saved['source_sha256'] != source_hashes():
        raise ValueError('Implementation differs; use the saved source_snapshot/train.py or test.py')
    return saved


def checkpoint_model(saved, device):
    model = build_model(saved['variant'], saved['config']['model'], saved['config']['ablation'],
                        saved.get('cca_dimension')).to(device)
    model.load_state_dict(saved['model'])
    return model


def cache_description(data):
    from data.test_cache import label_mapping
    reverse = label_mapping(data)
    if set(reverse) != set(range(50)):
        raise ValueError('Training cache must contain 50 classes labeled 0..49')
    return {**data.describe(hashes=True), 'label_mapping_sha256': file_hash(data.directory / 'label_mapping.json'),
            'original_ids': {str(label): raw for label, raw in reverse.items()}}


def task_plan(data, split, config, seed, shot, stage, count, epoch=0):
    return episode_plan(data.labels, split, config['episode']['way'], shot,
                        config['episode']['query_per_class'], count, plan_seed(seed, 0, shot, stage, epoch))


def evaluate(model, data, plan, shot, device, directory=None, context=None, classes=None):
    model.eval()
    way, query = plan.shape[1], plan.shape[2]-shot
    target = torch.arange(way, device=device).repeat_interleave(query)
    accuracies, losses, logits, predictions = [], [], [], []
    with torch.no_grad():
        for indices in plan:
            dt, at = data.batch(indices.reshape(-1), device)
            values = scores(model, dt, at, way, shot, query)
            loss = F.cross_entropy(values, target)
            if not torch.isfinite(values).all() or not torch.isfinite(loss):
                raise RuntimeError('Nonfinite evaluation output')
            pred = values.argmax(1)
            accuracies.append(float((pred == target).float().mean()))
            losses.append(float(loss))
            if directory is not None:
                logits.append(values.cpu().numpy())
                predictions.append(pred.cpu().numpy())
    result = {'accuracy': float(np.mean(accuracies)), 'loss': float(np.mean(losses)),
              'episode_statistics': statistics(accuracies)}
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'episode_plan.npy', plan, allow_pickle=False)
        write_csv(directory / 'episode_accuracies.csv', [{'episode': i+1, 'accuracy': value,
                  'accuracy_percent': value*100} for i, value in enumerate(accuracies)])
        result['class_analysis'] = save_query_artifacts(data, plan, shot, np.stack(logits),
            np.stack(predictions), directory, 'test', context, classes)
        write_json(directory / 'test_summary.json', {**result, 'context': context})
    return result


def parameter_values(model):
    result = {'scale': float(model.scale.detach().cpu())}
    base = model.base if isinstance(model, CCAClassifier) else model
    for name in ('dt', 'at'):
        encoder = getattr(base, 'encoder_' + name, None)
        if encoder is not None and hasattr(encoder, 'level_weights'):
            result.update({'lambda_%s_%d' % (name, i+1): float(value)
                           for i, value in enumerate(encoder.level_weights.detach().cpu())})
    if hasattr(base, 'fusion_weights'):
        for i, value in enumerate(base.fusion_weights.softmax(0).detach().cpu()):
            result['fusion_weight_' + ('dt' if i == 0 else 'at')] = float(value)
    return result


def payload(model, variant, config, split, description, seed, shot, epoch, best_epoch,
            best_val, stale, history, optimizer=None, scheduler=None):
    return {'format': 'mcg_ablation_v1', 'variant': variant, 'config': config, 'split': split,
            'cache_description': description, 'seed': seed, 'shot': shot, 'epoch': epoch,
            'best_epoch': best_epoch, 'best_validation_accuracy': best_val, 'stale_epochs': stale,
            'history': history, 'model': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'rng': rng_state(), 'source_sha256': source_hashes(),
            'cca_dimension': model.dimension if isinstance(model, CCAClassifier) else None,
            'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__),
                            'cuda': torch.version.cuda, 'device': str(next(model.parameters()).device)},
            'parameter_count': sum(p.numel() for p in model.parameters())}


def final_test(best_path, data, config, split, seed, shot, directory, device, epoch_count):
    saved = load_checkpoint(best_path)
    model = checkpoint_model(saved, device)
    plan = task_plan(data, split['test_classes'], config, seed, shot, 4, config['testing']['episodes'])
    context = {'seed': seed, 'variant': saved['variant'], 'checkpoint_epoch': saved['epoch'],
               'checkpoint_sha256': file_hash(best_path), 'source_sha256': saved['source_sha256'],
               'episode_seed': plan_seed(seed, 0, shot, 4)}
    result = evaluate(model, data, plan, shot, device, directory / 'test', context, split['test_classes'])
    print_statistics(saved['variant'] + ' final best-validation test', result['episode_statistics'])
    row = {'variant': saved['variant'], 'seed': seed, 'shot': shot, 'best_epoch': saved['best_epoch'],
           'epochs_completed': epoch_count, 'best_validation_accuracy': saved['best_validation_accuracy'],
           'checkpoint': str(best_path.resolve()), 'checkpoint_sha256': file_hash(best_path),
           'source_sha256': saved['source_sha256'], 'test': result,
           'parameters': parameter_values(model), 'parameter_count': saved['parameter_count']}
    write_json(directory / 'run_result.json', row)
    return row


def check_saved(saved, config, variant, split, description, seed, shot):
    expected, actual = copy.deepcopy(saved['config']), copy.deepcopy(config)
    for value in (expected, actual):
        value['runtime'].pop('device', None)
        value['runtime'].pop('cpu_threads', None)
        value['data'].pop('cache_dir', None)
    saved_data = saved['cache_description']
    if (expected != actual or saved['variant'] != variant or saved['seed'] != seed
            or saved['shot'] != shot or saved['split'] != split
            or saved_data['sha256'] != description['sha256']
            or saved_data['label_mapping_sha256'] != description['label_mapping_sha256']):
        raise ValueError('Checkpoint settings, split or source cache differ')


def train_one(variant, data, split, config, description, seed, shot, directory,
              checkpoint_dir, device, resume=False, stop_after=None):
    directory, checkpoint_dir = Path(directory), Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = directory / 'run_result.json'
    best_path, last_path = checkpoint_dir / 'best.pt', checkpoint_dir / 'last.pt'
    if completed.is_file():
        row = read_json(completed)
        if not resume or row['source_sha256'] != source_hashes() or row['checkpoint_sha256'] != file_hash(best_path):
            raise ValueError('Completed trial differs from the saved implementation/checkpoint')
        return row
    seed_all(seed)
    model = build_model(variant, config['model'], config['ablation']).to(device)
    options = config['training']['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'],
        weight_decay=options['weight_decay'], betas=options['betas'], eps=options['eps'])
    schedule = config['training']['scheduler']
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule['step_size'], schedule['gamma'])
    history, best_val, best_epoch, stale, start = [], -1., 0, 0, 1
    if last_path.is_file():
        if not resume:
            raise ValueError('Existing trial requires --resume')
        saved = load_checkpoint(last_path)
        check_saved(saved, config, variant, split, description, seed, shot)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best_val = saved['history'], saved['best_validation_accuracy']
        best_epoch, stale, start = saved['best_epoch'], saved['stale_epochs'], saved['epoch']+1
        restore_rng(saved['rng'])
    print('%s | seed=%d shot=%d | parameters=%d' % (
        variant, seed, shot, sum(p.numel() for p in model.parameters())), flush=True)
    way, query = config['episode']['way'], config['episode']['query_per_class']
    tr, testing = config['training'], config['testing']
    target = torch.arange(way, device=device).repeat_interleave(query)
    for epoch in range(start, tr['epochs']+1):
        if tr['patience'] and stale >= tr['patience']:
            break
        model.train()
        plan = task_plan(data, split['train_classes'], config, seed, shot, 1, tr['episodes_per_epoch'], epoch)
        losses, accuracies = [], []
        for indices in plan:
            optimizer.zero_grad(set_to_none=True)
            dt, at = data.batch(indices.reshape(-1), device)
            logits = scores(model, dt, at, way, shot, query)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss/logits')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(float((logits.detach().argmax(1) == target).float().mean()))
        val_epoch = epoch if config['validation']['sampling'] == 'per_epoch' else 0
        val_plan = task_plan(data, split['validation_classes'], config, seed, shot, 2,
                             config['validation']['episodes'], val_epoch)
        val = evaluate(model, data, val_plan, shot, device)
        improved = val['accuracy'] > best_val
        stale = 0 if improved else stale+1
        if improved:
            best_val, best_epoch = val['accuracy'], epoch
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               'train_accuracy': float(np.mean(accuracies)), 'validation_loss': val['loss'],
               'validation_accuracy': val['accuracy'], 'best_validation_accuracy': best_val,
               'validation_improved': improved, 'stale_epochs': stale,
               'learning_rate': optimizer.param_groups[0]['lr'], **parameter_values(model)}
        print('%s seed=%d shot=%d epoch=%03d loss=%.4f val_loss=%.4f train=%.2f%% val=%.2f%% best=%.2f%% scale=%.4f' % (
            variant, seed, shot, epoch, row['train_loss'], row['validation_loss'],
            row['train_accuracy']*100, val['accuracy']*100, best_val*100, row['scale']), flush=True)
        every = testing['every_epochs']
        if epoch >= testing['start_epoch'] and ((every == 0 and improved) or (every and epoch % every == 0)):
            monitor_plan = task_plan(data, split['test_classes'], config, seed, shot, 5, testing['periodic_episodes'], epoch)
            monitor = evaluate(model, data, monitor_plan, shot, device)
            print_statistics(variant + ' periodic test', monitor['episode_statistics'])
            row.update({'monitor_' + key: value for key, value in monitor['episode_statistics'].items()})
            np.savez_compressed(directory / ('monitor_%03d.npz' % epoch), plan=monitor_plan)
        history.append(row)
        scheduler.step()
        state = payload(model, variant, config, split, description, seed, shot, epoch, best_epoch,
                        best_val, stale, history, optimizer, scheduler)
        if improved:
            atomic_save(best_path, state)
        atomic_save(last_path, state)
        write_csv(directory / 'history.csv', history)
        if tr['patience'] and stale >= tr['patience']:
            print('Early stopping: %s epoch=%d best_epoch=%d' % (variant, epoch, best_epoch), flush=True)
            break
        if stop_after is not None and epoch >= stop_after and epoch < tr['epochs']:
            return None
    return final_test(best_path, data, config, split, seed, shot, directory, device, len(history))


def fit_cca_trial(data, split, config, description, seed, shot, directory, checkpoint_dir,
                  concat_checkpoint, device, resume=False):
    directory, checkpoint_dir = Path(directory), Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / 'best.pt'
    if (directory / 'run_result.json').is_file():
        row = read_json(directory / 'run_result.json')
        if not resume or row['source_sha256'] != source_hashes() or row['checkpoint_sha256'] != file_hash(best_path):
            raise ValueError('Completed CCA trial differs')
        return row
    if best_path.is_file() and resume:
        saved = load_checkpoint(best_path)
        check_saved(saved, config, 'tpa_re_cca', split, description, seed, shot)
        return final_test(best_path, data, config, split, seed, shot, directory, device, 0)
    source = load_checkpoint(concat_checkpoint)
    check_saved(source, config, 'tpa_re_concat', split, description, seed, shot)
    base = checkpoint_model(source, device).eval()
    base.requires_grad_(False)
    indices = np.flatnonzero(np.isin(data.labels, split['train_classes']))
    features_dt, features_at = [], []
    batch_size = config['ablation']['cca']['feature_batch_size']
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            dt, at = data.batch(indices[start:start+batch_size], device)
            left, right = base.paired(dt, at)
            features_dt.append(left.cpu().numpy())
            features_at.append(right.cpu().numpy())
    means, projections, correlations = fit_cca(np.concatenate(features_dt), np.concatenate(features_at),
                                               config['ablation']['cca']['regularization'])
    val_plan = task_plan(data, split['validation_classes'], config, seed, shot, 6,
                         config['validation']['episodes'])
    trials, best_model, best_val = [], None, -1.
    for dimension in sorted(config['ablation']['cca']['dimensions']):
        model = CCAClassifier(copy.deepcopy(base), dimension).to(device)
        for name, value in (('mean_dt', means[0]), ('mean_at', means[1]),
                            ('projection_dt', projections[0][:, :dimension]),
                            ('projection_at', projections[1][:, :dimension])):
            getattr(model, name).copy_(torch.as_tensor(value, dtype=torch.float32, device=device))
        val = evaluate(model, data, val_plan, shot, device)
        trials.append({'dimension': dimension, 'validation_accuracy': val['accuracy'], 'validation_loss': val['loss']})
        print('CCA seed=%d shot=%d dimension=%d validation=%.2f%%' % (seed, shot, dimension, val['accuracy']*100), flush=True)
        if val['accuracy'] > best_val:
            best_val, best_model = val['accuracy'], model
    state = payload(best_model, 'tpa_re_cca', config, split, description, seed, shot,
                    source['best_epoch'], source['best_epoch'], best_val, 0, [])
    state['cca_source_checkpoint_sha256'] = file_hash(concat_checkpoint)
    state['cca_fitting_sample_count'] = len(indices)
    state['cca_fitting_sample_indices'] = indices.tolist()
    atomic_save(best_path, state)
    write_json(directory / 'cca_selection.json', {'training_sample_count': len(indices),
        'training_class_ids': split['train_classes'], 'validation_class_ids': split['validation_classes'],
        'dimension_trials': trials, 'selected_dimension': best_model.dimension,
        'source_checkpoint_sha256': file_hash(concat_checkpoint), 'source_best_epoch': source['best_epoch'],
        'canonical_correlations': correlations.tolist(), 'projection_fitting_split': 'train_classes',
        'dimension_selection_split': 'validation_classes'})
    np.save(directory / 'cca_validation_plan.npy', val_plan, allow_pickle=False)
    return final_test(best_path, data, config, split, seed, shot, directory, device, 0)
