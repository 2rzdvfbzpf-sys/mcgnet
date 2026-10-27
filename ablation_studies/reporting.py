"""Episode-level and independent-training accuracy reports."""
import _bootstrap
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from analysis.reports import aggregate_class_reports
from settings import read_json, write_json


def statistics(values, unit='test_episode'):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError('Accuracy values must be a nonempty finite vector in [0,1]')
    n = len(values)
    mean = float(values.mean() * 100)
    sd = float(values.std(ddof=1) * 100) if n > 1 else None
    critical = float(t.ppf(0.975, n-1)) if unit == 'independent_training_run' and n > 1 else 1.96
    half = critical * sd / math.sqrt(n) if sd is not None else None
    return {'statistical_unit': unit, 'n': n, 'mean_accuracy_percent': mean,
            'sample_variance_pp2': None if sd is None else sd * sd,
            'sample_sd_pp': sd, 'ddof': 1 if n > 1 else None,
            'confidence_level': 0.95, 'ci95_half_width_pp': half,
            'ci95_lower_percent': None if half is None else mean-half,
            'ci95_upper_percent': None if half is None else mean+half,
            'ci95_method': ('student_t_over_independent_run_means' if unit == 'independent_training_run'
                            else 'normal_approximation_over_test_episodes')}


def print_statistics(label, result):
    half = result['ci95_half_width_pp']
    interval = 'N/A' if half is None else '%.4f pp | Interval: [%.4f%%, %.4f%%]' % (
        half, result['ci95_lower_percent'], result['ci95_upper_percent'])
    print('%s | %s: %d | Mean accuracy: %.4f%% +/- %s (95%% CI)' % (
        label, 'Runs' if result['statistical_unit'] == 'independent_training_run' else 'Episodes',
        result['n'], result['mean_accuracy_percent'], interval), flush=True)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_summary(batch, manifest):
    rows, groups = [], {}
    for variant in manifest['execution_variants']:
        for plan in manifest['plans']:
            for shot in manifest['config']['experiment']['shots']:
                path = Path(batch) / plan['directory'] / variant / ('k%d' % shot) / 'run_result.json'
                if not path.is_file():
                    continue
                result = read_json(path)
                stat = result['test']['episode_statistics']
                row = {'variant': variant, 'seed': plan['seed'], 'shot': shot,
                       'best_epoch': result['best_epoch'], 'epochs_completed': result['epochs_completed'],
                       'best_validation_accuracy_percent': result['best_validation_accuracy']*100,
                       'test_episodes': stat['n'], 'test_mean_accuracy_percent': stat['mean_accuracy_percent'],
                       'episode_ci95_half_width_pp': stat['ci95_half_width_pp'],
                       'checkpoint': result['checkpoint']}
                rows.append(row)
                groups.setdefault((variant, shot), []).append(result)
    summaries = []
    for (variant, shot), results in groups.items():
        run_stat = statistics([r['test']['accuracy'] for r in results], 'independent_training_run')
        episode_stat = results[0]['test']['episode_statistics'] if len(results) == 1 else None
        display = run_stat if len(results) > 1 else episode_stat
        mean, half = display['mean_accuracy_percent'], display['ci95_half_width_pp']
        summaries.append({'variant': variant, 'shot': shot, 'completed_runs': len(results),
                          'expected_runs': manifest['config']['experiment']['runs'],
                          'mean_accuracy_percent': mean, 'ci95_half_width_pp': half,
                          'ci_statistical_unit': display['statistical_unit'],
                          'accuracy_mean_pm_ci95': ('%.2f +/- %.2f' % (mean, half) if half is not None else '%.2f +/- N/A' % mean),
                          'run_statistics': run_stat, 'single_run_episode_statistics': episode_stat})
        analyses = [r['test']['class_analysis'] for r in results]
        aggregate_class_reports(analyses, Path(batch) / 'class_reports', variant + '_k%d' % shot)
    expected = len(manifest['execution_variants']) * len(manifest['plans']) * len(manifest['config']['experiment']['shots'])
    summary = {'complete': len(rows) == expected, 'completed_models': len(rows), 'expected_models': expected,
               'rows': rows, 'summaries': summaries,
               'requested_variants': manifest['config']['ablation']['variants'],
               'dependency_variants': manifest['dependency_variants']}
    write_json(Path(batch) / 'summary.json', summary)
    write_csv(Path(batch) / 'runs.csv', rows)
    write_csv(Path(batch) / 'ablation_table.csv', [{k: v for k, v in row.items()
              if k not in ('run_statistics', 'single_run_episode_statistics')} for row in summaries])
    return summary
