"""MCG-Net evaluation reports."""
import csv
from pathlib import Path

import numpy as np

from data.radar_cache import read_json, write_json


def cache_label_mapping(data):
    mapping = read_json(data.directory / 'label_mapping.json')
    reverse = {int(label): int(original) for original, label in mapping.items()}
    present = set(int(label) for label in np.unique(data.labels))
    if set(reverse) != present or len(reverse) != len(mapping):
        raise ValueError('Original-ID mapping does not cover cache labels one-to-one')
    lookup = np.full(max(present) + 1, -1, dtype=np.int64)
    for label, original in reverse.items():
        lookup[label] = original
    return lookup, np.array(sorted(reverse.values()), dtype=np.int64)


def write_class_reports(directory, prefix, class_ids, counts, episode_counts,
                        split_run_counts, evaluated_run_counts, run_accuracies, run_count):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    class_ids, counts = np.asarray(class_ids), np.asarray(counts, dtype=np.int64)
    if counts.shape != (len(class_ids), len(class_ids)) or (counts < 0).any():
        raise ValueError('Invalid confusion matrix shape/counts')
    per_class = []
    for index, original in enumerate(class_ids):
        total, correct = int(counts[index].sum()), int(counts[index, index])
        values = run_accuracies[index]
        per_class.append({
            'original_id': int(original), 'status': 'evaluated' if total else 'not_evaluated',
            'test_split_run_count': int(split_run_counts[index]),
            'evaluated_run_count': int(evaluated_run_counts[index]),
            'test_episode_count': int(episode_counts[index]),
            'query_count': total, 'correct_count': correct,
            'accuracy_percent': correct / total * 100 if total else None,
            'run_accuracy_mean_percent': float(np.mean(values)) if values else None,
            'run_accuracy_sample_sd_pp': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    files = {key: prefix + suffix for key, suffix in {
        'per_class_csv': '_per_class.csv', 'per_class_json': '_per_class.json',
        'confusion_counts_csv': '_confusion_counts.csv',
        'confusion_row_percent_csv': '_confusion_row_percent.csv'}.items()}
    with (directory / files['per_class_csv']).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    for key, normalized in (('confusion_counts_csv', False), ('confusion_row_percent_csv', True)):
        with (directory / files[key]).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['true_original_id'] + class_ids.tolist())
            for original, row in zip(class_ids, counts):
                total = int(row.sum())
                values = ([float(value) / total * 100 for value in row] if total else
                          [None] * len(row)) if normalized else row.tolist()
                writer.writerow([int(original)] + values)
    result = {'class_original_ids': class_ids.tolist(), 'confusion_counts': counts.tolist(),
              'confusion_axes': {'rows': 'true_original_id', 'columns': 'predicted_original_id'},
              'per_class': per_class, 'run_count': run_count,
              'accuracy_percent_definition': 'pooled correct query occurrences / pooled query occurrences * 100',
              'query_count_includes_repeated_samples_across_episodes': True,
              'unobserved_accuracy': None, 'artifact_files': files}
    write_json(directory / files['per_class_json'], result)
    return result


def save_query_artifacts(data, plan, shot, logits, predicted_local, output_dir, prefix, context,
                         test_classes):
    output_dir = Path(output_dir)
    lookup, class_ids = cache_label_mapping(data)
    plan = np.asarray(plan, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    predicted_local = np.asarray(predicted_local, dtype=np.int64)
    episodes, way, sample_count = plan.shape
    queries_per_class = sample_count - shot
    query_count = way * queries_per_class
    if logits.shape != (episodes, query_count, way) or predicted_local.shape != (episodes, query_count):
        raise ValueError('Query logits/predictions do not match the saved episode plan')
    if not np.isfinite(logits).all() or not np.array_equal(logits.argmax(-1), predicted_local):
        raise ValueError('Query predictions disagree with logits')
    class_labels = data.labels[plan[:, :, 0]]
    if not np.all(data.labels[plan] == class_labels[:, :, None]):
        raise ValueError('Episode samples do not follow the recorded local class order')
    query_indices = plan[:, :, shot:].reshape(episodes, query_count)
    target_local = np.broadcast_to(np.repeat(np.arange(way), queries_per_class), (episodes, query_count))
    true_cache = data.labels[query_indices]
    predicted_cache = np.take_along_axis(class_labels, predicted_local, axis=1)
    true_original, predicted_original = lookup[true_cache], lookup[predicted_cache]
    class_original = lookup[class_labels]
    shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
    exponent = np.exp(shifted)
    probability = exponent / exponent.sum(-1, keepdims=True)
    predicted_probability = np.take_along_axis(probability, predicted_local[:, :, None], axis=-1)[:, :, 0]
    files = {'query_npz': prefix + '_queries.npz', 'query_csv': prefix + '_queries.csv',
             'episode_classes_csv': prefix + '_episode_classes.csv',
             'query_metadata_json': prefix + '_query_metadata.json'}
    np.savez_compressed(output_dir / files['query_npz'],
                        episode_ids=np.arange(1, episodes + 1),
                        query_sample_indices=query_indices, target_local=target_local,
                        predicted_local=predicted_local, true_cache_labels=true_cache,
                        predicted_cache_labels=predicted_cache, true_original_ids=true_original,
                        predicted_original_ids=predicted_original,
                        episode_class_cache_labels=class_labels,
                        episode_class_original_ids=class_original, logits=logits,
                        predicted_probabilities=predicted_probability.astype(np.float32))
    with (output_dir / files['query_csv']).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'shot', 'episode', 'query_in_episode', 'cache_sample_index',
                         'true_original_id', 'predicted_original_id', 'true_cache_label',
                         'predicted_cache_label', 'true_local_class', 'predicted_local_class',
                         'correct', 'predicted_probability'])
        for episode in range(episodes):
            writer.writerows((context['seed'], shot, episode + 1, query + 1,
                              int(query_indices[episode, query]), int(true_original[episode, query]),
                              int(predicted_original[episode, query]), int(true_cache[episode, query]),
                              int(predicted_cache[episode, query]), int(target_local[episode, query]),
                              int(predicted_local[episode, query]),
                              int(true_original[episode, query] == predicted_original[episode, query]),
                              float(predicted_probability[episode, query])) for query in range(query_count))
    with (output_dir / files['episode_classes_csv']).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode', 'local_class', 'cache_label', 'original_id'])
        writer.writerows((episode + 1, local, int(class_labels[episode, local]),
                          int(class_original[episode, local]))
                         for episode in range(episodes) for local in range(way))
    metadata = {**context, 'format_version': 1, 'shot': shot, 'way': way,
                'query_per_class': queries_per_class, 'episodes': episodes,
                'logits_shape': list(logits.shape), 'logits_definition': 'learned_scale * cosine_similarity',
                'logits_column_order': 'episode_class_original_ids[episode, local_class]',
                'episode_and_query_numbers': '1-based in CSV; array positions are 0-based',
                'cache_sample_index': '0-based row of the saved cache; matches episode plan indices',
                'local_class': '0-based, valid only within that episode',
                'cache_to_original_id': {str(label): int(lookup[label]) for label in np.unique(data.labels)},
                'artifact_files': files}
    write_json(output_dir / files['query_metadata_json'], metadata)
    counts = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    np.add.at(counts, (np.searchsorted(class_ids, true_original.ravel()),
                       np.searchsorted(class_ids, predicted_original.ravel())), 1)
    totals = counts.sum(1)
    episode_counts = np.array([(class_original == original).sum() for original in class_ids])
    split_counts = np.isin(class_ids, lookup[np.asarray(test_classes, dtype=np.int64)]).astype(np.int64)
    per_run_accuracies = [[float(counts[i, i]) / total * 100] if total else []
                          for i, total in enumerate(totals)]
    result = write_class_reports(output_dir, prefix, class_ids, counts, episode_counts,
                                 split_counts, (totals > 0).astype(np.int64), per_run_accuracies, 1)
    result['artifact_files'].update(files)
    return result


def aggregate_class_reports(analyses, directory, prefix):
    if not analyses:
        raise ValueError('Cannot aggregate empty class analyses')
    class_ids = analyses[0]['class_original_ids']
    if any(analysis['class_original_ids'] != class_ids for analysis in analyses):
        raise ValueError('Original-ID axes differ across runs')
    counts = np.sum([analysis['confusion_counts'] for analysis in analyses], axis=0, dtype=np.int64)
    episodes, split_runs, evaluated_runs, accuracies = [], [], [], []
    for index in range(len(class_ids)):
        rows = [analysis['per_class'][index] for analysis in analyses]
        episodes.append(sum(row['test_episode_count'] for row in rows))
        split_runs.append(sum(row['test_split_run_count'] for row in rows))
        evaluated_runs.append(sum(row['evaluated_run_count'] for row in rows))
        accuracies.append([row['accuracy_percent'] for row in rows if row['query_count']])
    return write_class_reports(directory, prefix, class_ids, counts, episodes, split_runs,
                                evaluated_runs, accuracies, len(analyses))
