import json
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

num_clients = 20
raw_dir_path = "dataset/pu/"
dir_path = "dataset/pu/"
train_ratio = 0.8
dirichlet_alpha = 0.05
size_jitter_ratio = 0.12
condition_profile = "balanced"
test_split_mode = "proportional"
split_strategy = "by_source_file"
window_size = 2048
window_stride = 512

label_map = {
    "K001": 0,
    "KI01": 1,
    "KA01": 2,
    "KB23": 3,
}

label_names = {
    0: "normal",
    1: "inner_race_fault",
    2: "outer_race_fault",
    3: "ball_fault",
}


def parse_file_info(file_name):
    match = re.match(r"^(N\d+_M\d+_F\d+)_(K001|KA01|KB23|KI01)_(\d+)\.mat$", file_name)
    if match is None:
        raise ValueError(f"Unsupported PU file name: {file_name}")

    condition_name, fault_code, repeat_id = match.groups()
    return label_map[fault_code], condition_name, int(repeat_id)


def _unwrap_scalar(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.reshape(-1)[0]
    return x


def load_signal(mat_path):
    data = sio.loadmat(mat_path)
    top_key = next((k for k in data.keys() if not k.startswith("__")), None)
    if top_key is None:
        raise KeyError(f"No valid top-level key found in {mat_path}")

    sample = data[top_key][0, 0]
    y_block = sample["Y"]

    for idx in range(y_block.shape[1]):
        entry = y_block[0, idx]
        name = _unwrap_scalar(entry["Name"])
        if str(name) == "vibration_1":
            signal = np.asarray(_unwrap_scalar(entry["Data"]), dtype=np.float32).reshape(-1)
            return signal

    raise KeyError(f"'vibration_1' not found in {mat_path}")


def segment_signal(signal, segment_length, stride):
    if len(signal) < segment_length:
        raise ValueError(
            f"Signal is shorter than the requested segment length: {len(signal)} < {segment_length}"
        )

    segments = []
    for start in range(0, len(signal) - segment_length + 1, stride):
        window = signal[start:start + segment_length]
        segments.append(window[:, None])

    return np.asarray(segments, dtype=np.float32)


def standardize_segments(segments):
    flat = segments.reshape(segments.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)
    normalized = ((flat - mean) / std).reshape(segments.shape)
    return normalized.astype(np.float32)


def load_all_conditions(raw_dir_path):
    file_infos = []
    for fault_dir in sorted(os.listdir(raw_dir_path)):
        fault_path = os.path.join(raw_dir_path, fault_dir)
        if not os.path.isdir(fault_path) or fault_dir not in label_map:
            continue
        for file_name in sorted(os.listdir(fault_path)):
            if not file_name.lower().endswith(".mat"):
                continue
            label_id, condition_name, repeat_id = parse_file_info(file_name)
            file_infos.append((fault_dir, file_name, label_id, condition_name, repeat_id))

    if not file_infos:
        raise FileNotFoundError(f"No MAT files were found under {raw_dir_path}")

    min_length = None
    signals = {}
    for fault_dir, file_name, _, _, _ in file_infos:
        mat_path = os.path.join(raw_dir_path, fault_dir, file_name)
        signal = load_signal(mat_path)
        signals[(fault_dir, file_name)] = signal
        min_length = len(signal) if min_length is None else min(min_length, len(signal))

    source_records = []
    condition_names = sorted({condition_name for _, _, _, condition_name, _ in file_infos})
    condition_to_id = {name: idx for idx, name in enumerate(condition_names)}

    for source_id, (fault_dir, file_name, label_id, condition_name, _) in enumerate(file_infos):
        signal = signals[(fault_dir, file_name)][:min_length]
        source_records.append(
            {
                "source_id": int(source_id),
                "fault_dir": fault_dir,
                "file_name": file_name,
                "label": int(label_id),
                "condition_id": int(condition_to_id[condition_name]),
                "condition_name": condition_name,
                "signal": signal.astype(np.float32),
            }
        )

        print(
            f"Loaded {file_name:<24} -> condition={condition_name}, "
            f"label={label_names[label_id]}, samples={len(signal)}"
        )

    return source_records, condition_names


def plot_single_client_distribution(client_id, client_labels, output_prefix):
    unique_labels, counts = np.unique(client_labels, return_counts=True)
    full_counts = np.zeros(len(label_map), dtype=np.int64)
    for label, count in zip(unique_labels, counts):
        full_counts[int(label)] = int(count)

    tick_labels = [label_names[idx] for idx in range(len(full_counts))]
    x = np.arange(len(full_counts))
    colors = plt.cm.Set3(np.linspace(0, 1, len(full_counts)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Label Distribution for Client {client_id}")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    total = max(int(full_counts.sum()), 1)
    ymax = max(full_counts.max() * 1.18, 1)
    ax.set_ylim(0, ymax)

    for bar, count in zip(bars, full_counts):
        if count == 0:
            continue
        ratio = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{count}\n{ratio:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_distribution_figure(dir_path, client_labels, client_id=0):
    fig_dir = os.path.join(dir_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    client_id = max(0, min(client_id, len(client_labels) - 1))
    plot_single_client_distribution(
        client_id,
        client_labels[client_id],
        os.path.join(fig_dir, f"pu_client_{client_id}_label_distribution"),
    )


def prepare_output_dirs(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for split_path in [train_path, test_path]:
        for file_name in os.listdir(split_path):
            if file_name.endswith(".npz"):
                os.remove(os.path.join(split_path, file_name))

    return train_path + os.sep, test_path + os.sep


def get_clients_per_condition(num_clients, num_conditions, profile):
    if profile == "balanced":
        base = num_clients // num_conditions
        counts = [base] * num_conditions
        for i in range(num_clients % num_conditions):
            counts[i] += 1
        return counts

    if profile == "moderate":
        if num_clients != 20 or num_conditions != 4:
            raise ValueError("The 'moderate' condition profile currently expects 4 conditions and 20 clients.")
        return [7, 5, 4, 4]

    if profile == "severe":
        if num_clients != 20 or num_conditions != 4:
            raise ValueError("The 'severe' condition profile currently expects 4 conditions and 20 clients.")
        return [10, 5, 3, 2]

    raise ValueError(f"Unsupported condition profile: {profile}")


def count_windows(signal_length):
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // window_stride


def summarize_clients(client_records, condition_names):
    statistic = []
    sizes = []
    for client_id, records in enumerate(client_records):
        client_labels = np.array([record["label"] for record in records], dtype=np.int64)
        client_conditions = np.array([record["condition_id"] for record in records], dtype=np.int64)
        estimated_windows = sum(count_windows(len(record["signal"])) for record in records)
        client_stat = []
        for label in np.unique(client_labels):
            label_window_count = sum(
                count_windows(len(record["signal"])) for record in records if record["label"] == label
            )
            client_stat.append((int(label), int(label_window_count)))
        statistic.append(client_stat)
        sizes.append(estimated_windows)
        condition_id = int(np.unique(client_conditions)[0])
        readable_labels = [label_names[int(label)] for label in np.unique(client_labels)]
        print(
            f"Client {client_id}\t Size of data: {estimated_windows}\t "
            f"Condition: {condition_names[condition_id]}\t Labels: {readable_labels}"
        )
        print(f"\t\t Samples of labels: {client_stat}")
        print("-" * 50)

    print(f"Client size range: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
    return statistic


def build_jittered_quotas(total_size, num_parts, jitter_ratio):
    base = total_size // num_parts
    quotas = np.full(num_parts, base, dtype=np.int64)
    quotas[: total_size % num_parts] += 1

    max_jitter = max(1, int(base * jitter_ratio))
    deltas = np.random.randint(-max_jitter, max_jitter + 1, size=num_parts)
    deltas -= int(np.round(np.mean(deltas)))
    quotas = np.maximum(quotas + deltas, max(1, base - max_jitter))

    diff = int(total_size - quotas.sum())
    order = np.random.permutation(num_parts)
    ptr = 0
    while diff != 0:
        idx = order[ptr % num_parts]
        if diff > 0:
            quotas[idx] += 1
            diff -= 1
        else:
            min_quota = max(1, base - max_jitter)
            if quotas[idx] > min_quota:
                quotas[idx] -= 1
                diff += 1
        ptr += 1
    return quotas


def build_source_groups(source_records):
    return [dict(record) for record in source_records]


def allocate_condition_random(condition_indices, quotas):
    shuffled = np.array(condition_indices, dtype=np.int64).copy()
    np.random.shuffle(shuffled)
    client_indices = []
    cursor = 0
    for quota in quotas:
        client_indices.append(np.array(sorted(shuffled[cursor:cursor + int(quota)].tolist()), dtype=np.int64))
        cursor += int(quota)
    return client_indices


def allocate_condition_label_skew(condition_indices, dataset_y, quotas, num_classes, alpha):
    condition_indices = np.array(condition_indices, dtype=np.int64)
    per_class_indices = []
    for class_id in range(num_classes):
        idxs = condition_indices[dataset_y[condition_indices] == class_id]
        np.random.shuffle(idxs)
        per_class_indices.append(idxs.tolist())

    num_parts = len(quotas)
    client_buckets = [[] for _ in range(num_parts)]
    remaining = quotas.astype(np.int64).copy()

    while np.any(remaining > 0):
        active_clients = np.where(remaining > 0)[0].tolist()
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes), size=len(active_clients))
        desired = np.zeros((len(active_clients), num_classes), dtype=np.int64)

        for row_id, client_id in enumerate(active_clients):
            quota = int(remaining[client_id])
            raw = proportions[row_id] * quota
            cnts = np.floor(raw).astype(np.int64)
            deficit = quota - cnts.sum()
            if deficit > 0:
                order = np.argsort(raw - cnts)[::-1]
                cnts[order[:deficit]] += 1
            desired[row_id] = cnts

        progress = False
        for row_id, client_id in enumerate(active_clients):
            taken = 0
            class_order = np.argsort(desired[row_id])[::-1]
            for class_id in class_order:
                want = int(desired[row_id, class_id])
                if want <= 0:
                    continue
                available = len(per_class_indices[class_id])
                if available <= 0:
                    continue
                take = min(want, available, int(remaining[client_id]) - taken)
                if take <= 0:
                    continue
                chosen = per_class_indices[class_id][:take]
                del per_class_indices[class_id][:take]
                client_buckets[client_id].extend(chosen)
                taken += take
                progress = True
                if taken == int(remaining[client_id]):
                    break
            remaining[client_id] -= taken

        if not progress:
            break

        if np.any(remaining > 0):
            leftover_pool = []
            for class_id in range(num_classes):
                leftover_pool.extend(per_class_indices[class_id])
                per_class_indices[class_id] = []
            np.random.shuffle(leftover_pool)
            cursor = 0
            for client_id in np.where(remaining > 0)[0]:
                need = int(remaining[client_id])
                client_buckets[client_id].extend(leftover_pool[cursor:cursor + need])
                cursor += need
                remaining[client_id] = 0
            break

    return [np.array(sorted(bucket), dtype=np.int64) for bucket in client_buckets]


def allocate_condition_random_sources(condition_source_groups, quotas):
    shuffled = condition_source_groups.copy()
    np.random.shuffle(shuffled)
    client_records = []
    cursor = 0
    for quota in quotas:
        assigned_groups = shuffled[cursor:cursor + int(quota)]
        cursor += int(quota)
        client_records.append([dict(group) for group in assigned_groups])
    return client_records


def allocate_condition_label_skew_sources(condition_source_groups, quotas, num_classes, alpha):
    per_class_groups = []
    for class_id in range(num_classes):
        groups = [group for group in condition_source_groups if group["label"] == class_id]
        np.random.shuffle(groups)
        per_class_groups.append(groups)

    num_parts = len(quotas)
    client_group_buckets = [[] for _ in range(num_parts)]
    remaining = quotas.astype(np.int64).copy()

    while np.any(remaining > 0):
        active_clients = np.where(remaining > 0)[0].tolist()
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes), size=len(active_clients))
        desired = np.zeros((len(active_clients), num_classes), dtype=np.int64)

        for row_id, client_id in enumerate(active_clients):
            quota = int(remaining[client_id])
            raw = proportions[row_id] * quota
            cnts = np.floor(raw).astype(np.int64)
            deficit = quota - cnts.sum()
            if deficit > 0:
                order = np.argsort(raw - cnts)[::-1]
                cnts[order[:deficit]] += 1
            desired[row_id] = cnts

        progress = False
        for row_id, client_id in enumerate(active_clients):
            taken = 0
            class_order = np.argsort(desired[row_id])[::-1]
            for class_id in class_order:
                want = int(desired[row_id, class_id])
                if want <= 0:
                    continue
                available = len(per_class_groups[class_id])
                if available <= 0:
                    continue
                take = min(want, available, int(remaining[client_id]) - taken)
                if take <= 0:
                    continue
                chosen = per_class_groups[class_id][:take]
                del per_class_groups[class_id][:take]
                client_group_buckets[client_id].extend(chosen)
                taken += take
                progress = True
                if taken == int(remaining[client_id]):
                    break
            remaining[client_id] -= taken

        if not progress:
            break

        if np.any(remaining > 0):
            leftover_pool = []
            for class_id in range(num_classes):
                leftover_pool.extend(per_class_groups[class_id])
                per_class_groups[class_id] = []
            np.random.shuffle(leftover_pool)
            cursor = 0
            for client_id in np.where(remaining > 0)[0]:
                need = int(remaining[client_id])
                client_group_buckets[client_id].extend(leftover_pool[cursor:cursor + need])
                cursor += need
                remaining[client_id] = 0
            break

    client_records = []
    for bucket in client_group_buckets:
        client_records.append([dict(group) for group in bucket])
    return client_records


def allocate_clients_by_condition(source_records, condition_names, num_clients, num_classes, niid, profile):
    num_conditions = len(condition_names)
    clients_per_condition = get_clients_per_condition(num_clients, num_conditions, profile)
    all_client_records = []
    source_groups = build_source_groups(source_records)

    for condition_id in range(num_conditions):
        condition_source_groups = [group for group in source_groups if group["condition_id"] == condition_id]
        quotas = build_jittered_quotas(
            len(condition_source_groups), clients_per_condition[condition_id], size_jitter_ratio
        )

        if niid:
            local_client_indices = allocate_condition_label_skew_sources(
                condition_source_groups, quotas, num_classes, dirichlet_alpha
            )
        else:
            local_client_indices = allocate_condition_random_sources(condition_source_groups, quotas)

        for client_record_list in local_client_indices:
            all_client_records.append(client_record_list)

    return all_client_records


def train_test_split_np(X, y, train_size, seed, stratify=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))

    if stratify is None:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        train_count = int(round(len(shuffled) * train_size))
        train_count = min(max(train_count, 1), len(shuffled) - 1)
        train_idx = shuffled[:train_count]
        test_idx = shuffled[train_count:]
    else:
        train_idx_parts = []
        test_idx_parts = []
        for label in np.unique(stratify):
            label_indices = indices[stratify == label].copy()
            rng.shuffle(label_indices)
            label_train_count = int(round(len(label_indices) * train_size))
            label_train_count = min(max(label_train_count, 1), len(label_indices) - 1)
            train_idx_parts.append(label_indices[:label_train_count])
            test_idx_parts.append(label_indices[label_train_count:])

        train_idx = np.concatenate(train_idx_parts)
        test_idx = np.concatenate(test_idx_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_split_fixed_test_count(X, y, test_count, seed, stratify=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))

    if len(indices) <= 1:
        raise ValueError("Each client must contain at least 2 samples to perform train/test split.")

    test_count = int(test_count)
    test_count = min(max(test_count, 1), len(indices) - 1)

    if stratify is None:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        test_idx = shuffled[:test_count]
        train_idx = shuffled[test_count:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    unique_labels, label_counts = np.unique(stratify, return_counts=True)
    if np.any(label_counts < 2):
        return train_test_split_fixed_test_count(X, y, test_count, seed, stratify=None)

    raw_test_counts = label_counts / label_counts.sum() * test_count
    per_label_test = np.floor(raw_test_counts).astype(np.int64)
    per_label_test = np.minimum(per_label_test, label_counts - 1)

    deficit = test_count - int(per_label_test.sum())
    if deficit > 0:
        remainders = raw_test_counts - per_label_test
        order = np.argsort(remainders)[::-1]
        for idx in order:
            if deficit == 0:
                break
            if per_label_test[idx] < label_counts[idx] - 1:
                per_label_test[idx] += 1
                deficit -= 1

    while int(per_label_test.sum()) > test_count:
        reducible = np.where(per_label_test > 0)[0]
        if len(reducible) == 0:
            break
        idx = reducible[np.argmax(per_label_test[reducible])]
        per_label_test[idx] -= 1

    if int(per_label_test.sum()) != test_count:
        return train_test_split_fixed_test_count(X, y, test_count, seed, stratify=None)

    train_idx_parts = []
    test_idx_parts = []
    for label, label_test_count in zip(unique_labels, per_label_test):
        label_indices = indices[stratify == label].copy()
        rng.shuffle(label_indices)
        label_test_count = int(label_test_count)
        test_idx_parts.append(label_indices[:label_test_count])
        train_idx_parts.append(label_indices[label_test_count:])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def segment_standardized_signal(signal):
    segments = segment_signal(signal, window_size, window_stride)
    return standardize_segments(segments)


def split_raw_signal_by_time(signal, train_size, test_count=None):
    total_windows = count_windows(len(signal))
    if total_windows < 2:
        raise ValueError("Signal is too short to create non-overlapping train/test window sets.")

    if test_count is None:
        test_ratio = 1.0 - train_size
    else:
        test_ratio = min(max(test_count / total_windows, 1.0 / total_windows), (total_windows - 1) / total_windows)

    split_point = int(round(len(signal) * (1.0 - test_ratio)))
    min_train_len = window_size
    min_test_len = window_size
    split_point = min(max(split_point, min_train_len), len(signal) - min_test_len)

    train_signal = signal[:split_point]
    test_signal = signal[split_point:]
    return train_signal, test_signal


def choose_test_source_indices(client_records, train_size, seed, target_test_count=None):
    num_sources = len(client_records)
    if num_sources <= 1:
        return set()

    rng = np.random.default_rng(seed)
    source_window_counts = np.array([count_windows(len(record["signal"])) for record in client_records], dtype=np.int64)
    avg_source_windows = max(1.0, float(np.mean(source_window_counts)))

    if target_test_count is None:
        desired_test_sources = int(round(num_sources * (1 - train_size)))
    else:
        desired_test_sources = int(round(target_test_count / avg_source_windows))
    desired_test_sources = min(max(desired_test_sources, 1), num_sources - 1)

    label_to_indices = {}
    for idx, record in enumerate(client_records):
        label_to_indices.setdefault(record["label"], []).append(idx)
    for idxs in label_to_indices.values():
        rng.shuffle(idxs)

    labels = sorted(label_to_indices.keys())
    counts = np.array([len(label_to_indices[label]) for label in labels], dtype=np.int64)
    max_test_per_label = np.maximum(counts - 1, 0)
    raw = counts / counts.sum() * desired_test_sources
    per_label_test = np.floor(raw).astype(np.int64)
    per_label_test = np.minimum(per_label_test, max_test_per_label)

    deficit = desired_test_sources - int(per_label_test.sum())
    if deficit > 0:
        remainders = raw - per_label_test
        order = np.argsort(remainders)[::-1]
        for idx in order:
            if deficit == 0:
                break
            if per_label_test[idx] < max_test_per_label[idx]:
                per_label_test[idx] += 1
                deficit -= 1

    test_indices = []
    for idx, label in enumerate(labels):
        take = int(per_label_test[idx])
        test_indices.extend(label_to_indices[label][:take])

    if len(test_indices) == 0:
        candidate_labels = [label for label in labels if len(label_to_indices[label]) > 1]
        if candidate_labels:
            test_indices.append(label_to_indices[candidate_labels[0]][0])
        else:
            test_indices.append(0)

    test_indices = set(test_indices)
    if len(test_indices) >= num_sources:
        test_indices.remove(next(iter(test_indices)))
    return test_indices


def records_to_window_dataset(records):
    if len(records) == 0:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    xs = []
    ys = []
    for record in records:
        segments = segment_standardized_signal(record["signal"])
        xs.append(segments)
        ys.append(np.full(len(segments), record["label"], dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def split_data_custom(client_records, seed):
    train_data, test_data = [], []
    train_counts, test_counts = [], []
    client_sizes = [sum(count_windows(len(record["signal"])) for record in records) for records in client_records]
    all_client_labels = []

    if test_split_mode == "balanced":
        total_test_samples = int(round(sum(client_sizes) * (1 - train_ratio)))
        target_test_count = max(1, total_test_samples // len(client_records))
        print("Test split mode: balanced")
        print(f"Target test samples per client: {target_test_count}")
    else:
        target_test_count = None
        print("Test split mode: proportional")

    for client_id, records in enumerate(client_records):
        if split_strategy == "window_random":
            X_full, y_full = records_to_window_dataset(records)
            _, counts = np.unique(y_full, return_counts=True)
            stratify_labels = y_full if np.min(counts) >= 2 else None
            if test_split_mode == "balanced":
                fixed_test_count = min(target_test_count, len(y_full) - 1)
                X_train, X_test, y_train, y_test = train_test_split_fixed_test_count(
                    X_full,
                    y_full,
                    fixed_test_count,
                    stratify=stratify_labels,
                    seed=seed + client_id,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split_np(
                    X_full,
                    y_full,
                    train_ratio,
                    stratify=stratify_labels,
                    seed=seed + client_id,
                )
        else:
            fixed_test_count = target_test_count
            if len(records) == 1:
                train_signal, test_signal = split_raw_signal_by_time(
                    records[0]["signal"],
                    train_ratio,
                    test_count=fixed_test_count,
                )
                train_records = [dict(records[0], signal=train_signal)]
                test_records = [dict(records[0], signal=test_signal)]
            else:
                test_source_indices = choose_test_source_indices(
                    records,
                    train_ratio,
                    seed + client_id,
                    target_test_count=fixed_test_count,
                )
                train_records = [dict(record) for idx, record in enumerate(records) if idx not in test_source_indices]
                test_records = [dict(record) for idx, record in enumerate(records) if idx in test_source_indices]

            X_train, y_train = records_to_window_dataset(train_records)
            X_test, y_test = records_to_window_dataset(test_records)

        train_data.append({"x": X_train, "y": y_train})
        test_data.append({"x": X_test, "y": y_test})
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))
        all_client_labels.append(np.concatenate([y_train, y_test], axis=0))

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Train/Test ratio: {train_ratio:.1%}/{1-train_ratio:.1%}")
    print()

    return train_data, test_data, all_client_labels


def save_file_custom(
    config_path,
    train_path,
    test_path,
    train_data,
    test_data,
    num_clients,
    num_classes,
    statistic,
    niid,
    balance,
    partition,
    seed,
    condition_names,
):
    config = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "non_iid": niid,
        "balance": balance,
        "partition": partition,
        "seed": seed,
        "train_ratio": train_ratio,
        "dirichlet_alpha": dirichlet_alpha if niid else None,
        "size_jitter_ratio": size_jitter_ratio,
        "condition_per_client": "single",
        "condition_profile": condition_profile,
        "test_split_mode": test_split_mode,
        "split_strategy": split_strategy,
        "window_size": window_size,
        "window_stride": window_stride,
        "signal_channel": "vibration_1",
        "condition_names": condition_names,
        "label_names": [label_names[idx] for idx in range(num_classes)],
        "Size of samples for labels in clients": statistic,
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Finish generating dataset.\n")


def generate_dataset(dir_path, raw_dir_path, num_clients, niid, balance, partition, seed):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(dir_path)
    config_path = dir_path + "config.json"

    source_records, condition_names = load_all_conditions(raw_dir_path)
    num_classes = len({record["label"] for record in source_records})
    estimated_total_windows = sum(count_windows(len(record["signal"])) for record in source_records)

    print(f"Estimated number of windows before train/test split: {estimated_total_windows}")
    print(f"Number of classes: {num_classes}")
    print(f"Random seed: {seed}")
    print("Client-condition mapping: one client belongs to exactly one condition")
    print(f"Condition profile: {condition_profile}")
    print(f"Window size / stride: {window_size} / {window_stride}")
    print(f"Size jitter ratio: {size_jitter_ratio}")
    print(f"Test split mode: {test_split_mode}")
    print(f"Split strategy: {split_strategy}")

    client_records = allocate_clients_by_condition(
        source_records, condition_names, num_clients, num_classes, niid, condition_profile
    )

    statistic = summarize_clients(client_records, condition_names)
    train_data, test_data, client_labels = split_data_custom(client_records, seed)
    save_file_custom(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
        seed,
        condition_names,
    )
    save_distribution_figure(dir_path, client_labels, client_id=0)
    print(f"Saved figures to {os.path.join(dir_path, 'figures')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python generate_pu.py <iid|noniid> [balance|-] [pat|dir|exdir|-] [seed] "
            "[condition_profile] [size_jitter_ratio] [proportional|balanced] [by_source_file|window_random]\n"
            "Examples:\n"
            "  python generate_pu.py iid - - 42\n"
            "  python generate_pu.py noniid - - 42 severe 0.25\n"
            "  python generate_pu.py noniid - - 42 severe 0.50 balanced by_source_file\n"
        )

    mode = sys.argv[1]
    if mode not in {"iid", "noniid"}:
        raise SystemExit("The first argument must be 'iid' or 'noniid'.")

    balance_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    partition_arg = sys.argv[3] if len(sys.argv) > 3 else "-"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    condition_profile = sys.argv[5] if len(sys.argv) > 5 else "balanced"
    size_jitter_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else 0.12
    test_split_mode = sys.argv[7] if len(sys.argv) > 7 else "proportional"
    split_strategy = sys.argv[8] if len(sys.argv) > 8 else "by_source_file"

    niid = mode == "noniid"
    balance = balance_arg == "balance"
    partition = partition_arg if partition_arg != "-" else None

    if test_split_mode not in {"proportional", "balanced"}:
        raise SystemExit("The seventh argument must be 'proportional' or 'balanced'.")
    if split_strategy not in {"by_source_file", "window_random"}:
        raise SystemExit("The eighth argument must be 'by_source_file' or 'window_random'.")

    generate_dataset(dir_path, raw_dir_path, num_clients, niid, balance, partition, seed)
