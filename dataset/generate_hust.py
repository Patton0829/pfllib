import json
import os
import random
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


num_clients = 20
raw_dir_path = "dataset/hust/Raw_data/"
dir_path = "dataset/hust/"

# HUST has long continuous signals. A stricter and harder protocol is used:
# raw signal -> non-overlapping source chunks -> client allocation -> train/test split -> windowing.
train_ratio = 0.7
dirichlet_alpha = 0.05
size_jitter_ratio = 0.25
condition_profile = "balanced"
test_split_mode = "balanced"
split_strategy = "by_source_file"
window_size = 2048
window_stride = 128
source_chunk_length = 32768
signal_channel = "Y"

target_conditions = ["65Hz", "70Hz", "75Hz", "80Hz"]

label_map = {
    "H": 0,
    "I": 1,
    "O": 2,
    "B": 3,
}

label_names = {
    0: "healthy",
    1: "inner_fault",
    2: "outer_fault",
    3: "ball_fault",
}

channel_columns = {
    "speed": 1,
    "x": 2,
    "y": 3,
    "z": 4,
}


def parse_file_info(file_name):
    base = os.path.splitext(file_name)[0]

    healthy_match = re.match(r"^(H)_(65|70|75|80)Hz$", base, flags=re.IGNORECASE)
    if healthy_match:
        fault_code, freq = healthy_match.groups()
        return label_map[fault_code.upper()], f"{freq}Hz", "healthy"

    fault_match = re.match(r"^(0\.5X)_([IOB])_(65|70|75|80)Hz$", base, flags=re.IGNORECASE)
    if fault_match:
        severity, fault_code, freq = fault_match.groups()
        return label_map[fault_code.upper()], f"{freq}Hz", severity

    raise ValueError(f"Unsupported or intentionally skipped HUST file: {file_name}")


def read_text_lines(path):
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read().splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def load_hust_signal(path, channel):
    lines = read_text_lines(path)
    data_start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "data":
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError(f"No 'Data' marker found in {path}")

    numeric_rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) < 5:
            continue
        try:
            numeric_rows.append([float(value) for value in parts[:5]])
        except ValueError:
            continue

    if not numeric_rows:
        raise ValueError(f"No numeric rows found in {path}")

    data = np.asarray(numeric_rows, dtype=np.float32)
    column = channel_columns[channel.lower()]
    return data[:, column].astype(np.float32)


def count_windows(signal_length):
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // window_stride


def split_signal_into_source_chunks(signal, chunk_length):
    usable_length = (len(signal) // chunk_length) * chunk_length
    if usable_length < chunk_length:
        raise ValueError(f"Signal is too short for source chunks: {len(signal)} < {chunk_length}")

    chunks = []
    for start in range(0, usable_length, chunk_length):
        chunk = signal[start:start + chunk_length]
        if count_windows(len(chunk)) >= 2:
            chunks.append(chunk.astype(np.float32))
    return chunks


def load_all_conditions(raw_dir_path):
    file_infos = []
    for file_name in sorted(os.listdir(raw_dir_path)):
        if not file_name.lower().endswith(".xls"):
            continue
        try:
            label_id, condition_name, severity = parse_file_info(file_name)
        except ValueError:
            continue
        file_infos.append((file_name, label_id, condition_name, severity))

    if not file_infos:
        raise FileNotFoundError(
            f"No target HUST files were found under {raw_dir_path}. "
            "Expected H_65Hz.xls and 0.5X_{I,O,B}_65Hz.xls style files."
        )

    condition_to_id = {name: idx for idx, name in enumerate(target_conditions)}
    source_records = []
    source_id = 0

    for file_name, label_id, condition_name, severity in file_infos:
        path = os.path.join(raw_dir_path, file_name)
        signal = load_hust_signal(path, signal_channel)
        chunks = split_signal_into_source_chunks(signal, source_chunk_length)

        for chunk_id, chunk in enumerate(chunks):
            source_records.append(
                {
                    "source_id": source_id,
                    "file_name": file_name,
                    "chunk_id": chunk_id,
                    "label": int(label_id),
                    "condition_id": int(condition_to_id[condition_name]),
                    "condition_name": condition_name,
                    "severity": severity,
                    "signal": chunk,
                }
            )
            source_id += 1

        print(
            f"Loaded {file_name:<20} -> condition={condition_name}, "
            f"label={label_names[label_id]}, chunks={len(chunks)}, samples={len(signal)}"
        )

    expected_pairs = {(condition, label) for condition in target_conditions for label in range(len(label_map))}
    actual_pairs = {(record["condition_name"], record["label"]) for record in source_records}
    missing = expected_pairs - actual_pairs
    if missing:
        raise ValueError(f"Missing condition-label pairs in HUST data: {sorted(missing)}")

    return source_records, target_conditions


def segment_signal(signal, segment_length, stride):
    if len(signal) < segment_length:
        raise ValueError(f"Signal is shorter than segment length: {len(signal)} < {segment_length}")

    segments = []
    for start in range(0, len(signal) - segment_length + 1, stride):
        segments.append(signal[start:start + segment_length, None])
    return np.asarray(segments, dtype=np.float32)


def standardize_segments(segments):
    flat = segments.reshape(segments.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    return ((flat - mean) / std).reshape(segments.shape).astype(np.float32)


def segment_standardized_signal(signal):
    return standardize_segments(segment_signal(signal, window_size, window_stride))


def records_to_window_dataset(records):
    if not records:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    xs = []
    ys = []
    for record in records:
        segments = segment_standardized_signal(record["signal"])
        xs.append(segments)
        ys.append(np.full(len(segments), record["label"], dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def prepare_output_dirs(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for split_path in (train_path, test_path):
        for file_name in os.listdir(split_path):
            if file_name.endswith(".npz"):
                os.remove(os.path.join(split_path, file_name))

    return train_path + os.sep, test_path + os.sep


def get_clients_per_condition(num_clients, num_conditions, profile):
    if profile == "balanced":
        base = num_clients // num_conditions
        counts = [base] * num_conditions
        for idx in range(num_clients % num_conditions):
            counts[idx] += 1
        return counts

    if profile == "moderate":
        if num_clients != 20 or num_conditions != 4:
            raise ValueError("'moderate' expects 20 clients and 4 conditions.")
        return [7, 5, 4, 4]

    if profile == "severe":
        if num_clients != 20 or num_conditions != 4:
            raise ValueError("'severe' expects 20 clients and 4 conditions.")
        return [10, 5, 3, 2]

    raise ValueError(f"Unsupported condition profile: {profile}")


def build_jittered_quotas(total_size, num_parts, jitter_ratio):
    if num_parts <= 0:
        raise ValueError("num_parts must be positive.")
    if total_size < num_parts:
        raise ValueError(f"Cannot split {total_size} source chunks across {num_parts} clients.")

    base = total_size // num_parts
    quotas = np.full(num_parts, base, dtype=np.int64)
    quotas[: total_size % num_parts] += 1

    max_jitter = max(1, int(round(base * jitter_ratio)))
    deltas = np.random.randint(-max_jitter, max_jitter + 1, size=num_parts)
    deltas -= int(np.round(np.mean(deltas)))
    quotas = np.maximum(quotas + deltas, 1)

    diff = int(total_size - quotas.sum())
    order = np.random.permutation(num_parts)
    ptr = 0
    while diff != 0:
        idx = order[ptr % num_parts]
        if diff > 0:
            quotas[idx] += 1
            diff -= 1
        elif quotas[idx] > 1:
            quotas[idx] -= 1
            diff += 1
        ptr += 1
    return quotas


def allocate_condition_random_sources(condition_groups, quotas):
    shuffled = condition_groups.copy()
    np.random.shuffle(shuffled)
    client_records = []
    cursor = 0
    for quota in quotas:
        assigned = shuffled[cursor:cursor + int(quota)]
        cursor += int(quota)
        client_records.append([dict(record) for record in assigned])
    return client_records


def allocate_condition_label_skew_sources(condition_groups, quotas, num_classes, alpha):
    per_class = []
    for class_id in range(num_classes):
        groups = [group for group in condition_groups if group["label"] == class_id]
        np.random.shuffle(groups)
        per_class.append(groups)

    buckets = [[] for _ in range(len(quotas))]
    remaining = quotas.astype(np.int64).copy()

    while np.any(remaining > 0):
        active = np.where(remaining > 0)[0].tolist()
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes), size=len(active))
        progress = False

        for row_id, client_id in enumerate(active):
            quota = int(remaining[client_id])
            raw = proportions[row_id] * quota
            desired = np.floor(raw).astype(np.int64)
            deficit = quota - int(desired.sum())
            if deficit > 0:
                order = np.argsort(raw - desired)[::-1]
                desired[order[:deficit]] += 1

            taken = 0
            for class_id in np.argsort(desired)[::-1]:
                want = int(desired[class_id])
                if want <= 0 or not per_class[class_id]:
                    continue
                take = min(want, len(per_class[class_id]), quota - taken)
                if take <= 0:
                    continue
                buckets[client_id].extend(per_class[class_id][:take])
                del per_class[class_id][:take]
                taken += take
                progress = True
                if taken == quota:
                    break
            remaining[client_id] -= taken

        if not progress:
            leftover = []
            for groups in per_class:
                leftover.extend(groups)
                groups.clear()
            np.random.shuffle(leftover)
            cursor = 0
            for client_id in np.where(remaining > 0)[0]:
                need = int(remaining[client_id])
                buckets[client_id].extend(leftover[cursor:cursor + need])
                cursor += need
                remaining[client_id] = 0

    return [[dict(record) for record in bucket] for bucket in buckets]


def allocate_clients_by_condition(source_records, condition_names, num_clients, num_classes, niid, profile):
    clients_per_condition = get_clients_per_condition(num_clients, len(condition_names), profile)
    all_client_records = []

    for condition_id, local_client_count in enumerate(clients_per_condition):
        groups = [dict(record) for record in source_records if record["condition_id"] == condition_id]
        quotas = build_jittered_quotas(len(groups), local_client_count, size_jitter_ratio)

        if niid:
            local_clients = allocate_condition_label_skew_sources(groups, quotas, num_classes, dirichlet_alpha)
        else:
            local_clients = allocate_condition_random_sources(groups, quotas)
        all_client_records.extend(local_clients)

    return all_client_records


def split_raw_signal_by_time(signal, train_size, test_count=None):
    total_windows = count_windows(len(signal))
    if total_windows < 2:
        raise ValueError("Signal is too short to split by time.")

    if test_count is None:
        test_ratio = 1.0 - train_size
    else:
        test_ratio = min(max(test_count / total_windows, 1.0 / total_windows), (total_windows - 1) / total_windows)

    split_point = int(round(len(signal) * (1.0 - test_ratio)))
    split_point = min(max(split_point, window_size), len(signal) - window_size)
    return signal[:split_point], signal[split_point:]


def choose_test_source_indices(client_records, train_size, seed, target_test_count=None):
    if len(client_records) <= 1:
        return set()

    rng = np.random.default_rng(seed)
    window_counts = np.array([count_windows(len(record["signal"])) for record in client_records], dtype=np.int64)
    avg_windows = max(1.0, float(np.mean(window_counts)))
    desired_sources = int(round(len(client_records) * (1.0 - train_size)))
    if target_test_count is not None:
        desired_sources = int(round(target_test_count / avg_windows))
    desired_sources = min(max(desired_sources, 1), len(client_records) - 1)

    label_to_indices = {}
    for idx, record in enumerate(client_records):
        label_to_indices.setdefault(record["label"], []).append(idx)
    for indices in label_to_indices.values():
        rng.shuffle(indices)

    labels = sorted(label_to_indices.keys())
    counts = np.array([len(label_to_indices[label]) for label in labels], dtype=np.int64)
    max_per_label = np.maximum(counts - 1, 0)
    raw = counts / counts.sum() * desired_sources
    per_label_test = np.minimum(np.floor(raw).astype(np.int64), max_per_label)

    deficit = desired_sources - int(per_label_test.sum())
    if deficit > 0:
        order = np.argsort(raw - per_label_test)[::-1]
        for idx in order:
            if deficit == 0:
                break
            if per_label_test[idx] < max_per_label[idx]:
                per_label_test[idx] += 1
                deficit -= 1

    test_indices = []
    for idx, label in enumerate(labels):
        test_indices.extend(label_to_indices[label][: int(per_label_test[idx])])

    if not test_indices:
        candidates = [label for label in labels if len(label_to_indices[label]) > 1]
        test_indices.append(label_to_indices[candidates[0]][0] if candidates else 0)

    test_indices = set(test_indices)
    if len(test_indices) >= len(client_records):
        test_indices.remove(next(iter(test_indices)))
    return test_indices


def train_test_split_np(X, y, train_size, seed, stratify=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))

    if stratify is None:
        rng.shuffle(indices)
        train_count = min(max(int(round(len(indices) * train_size)), 1), len(indices) - 1)
        train_idx = indices[:train_count]
        test_idx = indices[train_count:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    train_parts = []
    test_parts = []
    for label in np.unique(stratify):
        label_indices = indices[stratify == label].copy()
        rng.shuffle(label_indices)
        train_count = min(max(int(round(len(label_indices) * train_size)), 1), len(label_indices) - 1)
        train_parts.append(label_indices[:train_count])
        test_parts.append(label_indices[train_count:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_split_fixed_test_count(X, y, test_count, seed, stratify=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    test_count = min(max(int(test_count), 1), len(indices) - 1)

    if stratify is None:
        rng.shuffle(indices)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    unique_labels, label_counts = np.unique(stratify, return_counts=True)
    if np.any(label_counts < 2):
        return train_test_split_fixed_test_count(X, y, test_count, seed)

    raw = label_counts / label_counts.sum() * test_count
    per_label_test = np.minimum(np.floor(raw).astype(np.int64), label_counts - 1)
    deficit = test_count - int(per_label_test.sum())
    if deficit > 0:
        order = np.argsort(raw - per_label_test)[::-1]
        for idx in order:
            if deficit == 0:
                break
            if per_label_test[idx] < label_counts[idx] - 1:
                per_label_test[idx] += 1
                deficit -= 1

    if int(per_label_test.sum()) != test_count:
        return train_test_split_fixed_test_count(X, y, test_count, seed)

    train_parts = []
    test_parts = []
    for label, count in zip(unique_labels, per_label_test):
        label_indices = indices[stratify == label].copy()
        rng.shuffle(label_indices)
        test_parts.append(label_indices[: int(count)])
        train_parts.append(label_indices[int(count):])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split_data_custom(client_records, seed):
    train_data = []
    test_data = []
    client_labels_for_plot = []
    train_counts = []
    test_counts = []
    client_sizes = [sum(count_windows(len(record["signal"])) for record in records) for records in client_records]

    if test_split_mode == "balanced":
        total_test = int(round(sum(client_sizes) * (1.0 - train_ratio)))
        target_test_count = max(1, total_test // len(client_records))
        print("Test split mode: balanced")
        print(f"Target test samples per client: {target_test_count}")
    else:
        target_test_count = None
        print("Test split mode: proportional")

    for client_id, records in enumerate(client_records):
        if split_strategy == "window_random":
            X_full, y_full = records_to_window_dataset(records)
            _, label_counts = np.unique(y_full, return_counts=True)
            stratify = y_full if len(label_counts) > 0 and np.min(label_counts) >= 2 else None
            if test_split_mode == "balanced":
                X_train, X_test, y_train, y_test = train_test_split_fixed_test_count(
                    X_full, y_full, min(target_test_count, len(y_full) - 1), seed + client_id, stratify
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split_np(
                    X_full, y_full, train_ratio, seed + client_id, stratify
                )
        else:
            if len(records) == 1:
                train_signal, test_signal = split_raw_signal_by_time(
                    records[0]["signal"], train_ratio, test_count=target_test_count
                )
                train_records = [dict(records[0], signal=train_signal)]
                test_records = [dict(records[0], signal=test_signal)]
            else:
                test_indices = choose_test_source_indices(records, train_ratio, seed + client_id, target_test_count)
                train_records = [dict(record) for idx, record in enumerate(records) if idx not in test_indices]
                test_records = [dict(record) for idx, record in enumerate(records) if idx in test_indices]
            X_train, y_train = records_to_window_dataset(train_records)
            X_test, y_test = records_to_window_dataset(test_records)

        train_data.append({"x": X_train, "y": y_train})
        test_data.append({"x": X_test, "y": y_test})
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))
        client_labels_for_plot.append(np.concatenate([y_train, y_test], axis=0))

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Train/Test ratio: {train_ratio:.1%}/{1.0 - train_ratio:.1%}")
    return train_data, test_data, client_labels_for_plot


def summarize_clients(client_records, condition_names):
    statistic = []
    sizes = []
    for client_id, records in enumerate(client_records):
        labels = np.array([record["label"] for record in records], dtype=np.int64)
        conditions = np.array([record["condition_id"] for record in records], dtype=np.int64)
        estimated_windows = sum(count_windows(len(record["signal"])) for record in records)
        client_stat = []
        for label in np.unique(labels):
            label_count = sum(count_windows(len(record["signal"])) for record in records if record["label"] == label)
            client_stat.append((int(label), int(label_count)))
        statistic.append(client_stat)
        sizes.append(estimated_windows)
        condition_id = int(np.unique(conditions)[0])
        readable_labels = [label_names[int(label)] for label in np.unique(labels)]
        print(
            f"Client {client_id}\t Size of data: {estimated_windows}\t "
            f"Condition: {condition_names[condition_id]}\t Labels: {readable_labels}"
        )
        print(f"\t\t Samples of labels: {client_stat}")
        print("-" * 50)

    print(f"Client size range: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
    return statistic


def plot_single_client_distribution(client_id, client_labels, output_prefix):
    unique_labels, counts = np.unique(client_labels, return_counts=True)
    full_counts = np.zeros(len(label_map), dtype=np.int64)
    for label, count in zip(unique_labels, counts):
        full_counts[int(label)] = int(count)

    x = np.arange(len(full_counts))
    tick_labels = [label_names[idx] for idx in range(len(full_counts))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(full_counts)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Label Distribution for Client {client_id}")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    total = max(int(full_counts.sum()), 1)
    ymax = max(full_counts.max() * 1.18, 1)
    ax.set_ylim(0, ymax)
    for bar, count in zip(bars, full_counts):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{count}\n{100.0 * count / total:.1f}%",
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
        os.path.join(fig_dir, f"hust_client_{client_id}_label_distribution"),
    )


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
        "source_chunk_length": source_chunk_length,
        "signal_channel": signal_channel,
        "fault_severity": "0.5X for faulty classes; H has no severity prefix",
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
    print("Finish generating HUST dataset.\n")


def generate_dataset(dir_path, raw_dir_path, num_clients, niid, balance, partition, seed):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(dir_path)
    config_path = os.path.join(dir_path, "config.json")

    source_records, condition_names = load_all_conditions(raw_dir_path)
    num_classes = len(label_map)
    total_windows = sum(count_windows(len(record["signal"])) for record in source_records)

    print(f"Estimated number of windows before train/test split: {total_windows}")
    print(f"Number of source chunks: {len(source_records)}")
    print(f"Number of classes: {num_classes}")
    print(f"Random seed: {seed}")
    print("Client-condition mapping: one client belongs to exactly one condition")
    print(f"Condition profile: {condition_profile}")
    print(f"Window size / stride: {window_size} / {window_stride}")
    print(f"Source chunk length: {source_chunk_length}")
    print(f"Signal channel: {signal_channel}")
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
            "Usage: python dataset/generate_hust.py <iid|noniid> [balance|-] [pat|dir|exdir|-] [seed] "
            "[condition_profile] [size_jitter_ratio] [proportional|balanced] "
            "[by_source_file|window_random] [signal_channel]\n"
            "Examples:\n"
            "  python dataset/generate_hust.py iid - - 42\n"
            "  python dataset/generate_hust.py noniid - - 42 balanced 0.25 balanced by_source_file\n"
            "  python dataset/generate_hust.py noniid - - 42 severe 0.35 balanced by_source_file Y\n"
        )

    mode = sys.argv[1]
    if mode not in {"iid", "noniid"}:
        raise SystemExit("The first argument must be 'iid' or 'noniid'.")

    balance_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    partition_arg = sys.argv[3] if len(sys.argv) > 3 else "-"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    condition_profile = sys.argv[5] if len(sys.argv) > 5 else "balanced"
    size_jitter_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
    test_split_mode = sys.argv[7] if len(sys.argv) > 7 else "balanced"
    split_strategy = sys.argv[8] if len(sys.argv) > 8 else "by_source_file"
    signal_channel = sys.argv[9] if len(sys.argv) > 9 else "Y"

    if test_split_mode not in {"proportional", "balanced"}:
        raise SystemExit("The seventh argument must be 'proportional' or 'balanced'.")
    if split_strategy not in {"by_source_file", "window_random"}:
        raise SystemExit("The eighth argument must be 'by_source_file' or 'window_random'.")
    if signal_channel.lower() not in channel_columns:
        raise SystemExit(f"The ninth argument must be one of: {sorted(channel_columns)}.")

    niid = mode == "noniid"
    balance = balance_arg == "balance"
    partition = partition_arg if partition_arg != "-" else None

    generate_dataset(dir_path, raw_dir_path, num_clients, niid, balance, partition, seed)
