import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import scipy.io as sio

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


num_clients = 20
raw_root_path = "dataset/uo"
dir_path = "dataset/uo_medium/"

train_ratio = 0.7
medium_alpha = 0.25
condition_affinity = 0.25
stage_affinity = 0.35
min_labels_per_client = 2

window_size = 2048
window_stride = 4096
source_chunk_length = 65536
selected_channels = ["Channel_1", "Channel_2"]
dataset_variant = "uo_medium"
dataset_display_name = "UO-Medium"

label_names = {
    0: "healthy",
    1: "inner_fault",
    2: "outer_fault",
    3: "ball_fault",
    4: "combination_fault",
}

label_map = {
    "H": 0,
    "I": 1,
    "O": 2,
    "B": 3,
    "C": 4,
}

condition_names = ["A", "B", "C", "D"]
stage_names = ["1", "2", "3"]


def prepare_output_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for split_dir in (train_dir, test_dir):
        for name in os.listdir(split_dir):
            if name.endswith(".npz"):
                os.remove(os.path.join(split_dir, name))
    return train_dir + os.sep, test_dir + os.sep


def discover_raw_dir(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Raw root directory does not exist: {root_dir}")

    candidates = []
    for current_root, dirs, _ in os.walk(root_dir):
        if os.path.basename(current_root).lower() == "origninal_data":
            candidates.append(current_root)
        dirs[:] = [name for name in dirs if name.lower() not in {"train", "test", "figures"}]

    if not candidates:
        raise FileNotFoundError(f"No Origninal_data directory was found under {root_dir}")
    candidates.sort(key=len)
    return candidates[0]


def parse_file_info(file_name):
    base = os.path.splitext(file_name)[0]
    match = re.match(r"^([HIOBC])-([ABCD])-([123])$", base, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported UO file name: {file_name}")
    label_code, condition, stage = match.groups()
    label_code = label_code.upper()
    condition = condition.upper()
    stage = stage.upper()
    return label_map[label_code], condition, stage


def load_uo_signal(path):
    data = sio.loadmat(path)
    missing = [name for name in selected_channels if name not in data]
    if missing:
        raise ValueError(f"Missing channels {missing} in {path}")

    channels = []
    for name in selected_channels:
        channel = np.asarray(data[name]).reshape(-1).astype(np.float32)
        channels.append(channel)
    min_len = min(len(channel) for channel in channels)
    if min_len < window_size:
        raise ValueError(f"Signal is shorter than window_size in {path}: {min_len}")
    return np.stack([channel[:min_len] for channel in channels], axis=1)


def count_windows(signal_length):
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // window_stride


def standardize_window(window):
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    return ((window - mean) / (std + 1e-8)).astype(np.float32)


def chunk_to_windows(signal, start, length):
    windows = []
    end = start + length
    for window_start in range(start, end - window_size + 1, window_stride):
        window = signal[window_start:window_start + window_size]
        windows.append(standardize_window(window))
    return windows


def load_all_source_records(raw_dir):
    records = []
    mat_files = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(raw_dir)
        for name in files
        if name.lower().endswith(".mat")
    )
    if not mat_files:
        raise FileNotFoundError(f"No MAT files were found under {raw_dir}")

    for path in mat_files:
        file_name = os.path.basename(path)
        label, condition, stage = parse_file_info(file_name)
        signal = load_uo_signal(path)
        usable_length = (len(signal) // source_chunk_length) * source_chunk_length
        chunk_count = 0
        window_count = 0
        for chunk_start in range(0, usable_length, source_chunk_length):
            chunk_windows = count_windows(source_chunk_length)
            if chunk_windows <= 0:
                continue
            records.append(
                {
                    "path": path,
                    "source_file": file_name,
                    "chunk_start": int(chunk_start),
                    "chunk_length": int(source_chunk_length),
                    "label": int(label),
                    "condition": condition,
                    "condition_id": condition_names.index(condition),
                    "stage": stage,
                    "stage_id": stage_names.index(stage),
                    "window_count": int(chunk_windows),
                }
            )
            chunk_count += 1
            window_count += chunk_windows
        print(
            f"Loaded {file_name:<10} label={label_names[label]:<18} "
            f"condition={condition} stage={stage} chunks={chunk_count} windows={window_count}"
        )
    return records


def split_source_records_by_file(records, seed):
    rng = random.Random(seed)
    train_records = []
    test_records = []
    groups = defaultdict(list)
    for record in records:
        groups[record["source_file"]].append(record)

    for source_file, items in sorted(groups.items()):
        items = [dict(record) for record in items]
        items.sort(key=lambda record: record["chunk_start"])
        split_idx = min(max(int(round(len(items) * train_ratio)), 1), len(items) - 1)
        train_part = items[:split_idx]
        test_part = items[split_idx:]
        rng.shuffle(train_part)
        rng.shuffle(test_part)
        train_records.extend(train_part)
        test_records.extend(test_part)

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def enforce_min_labels_per_client(buckets, min_labels):
    if min_labels <= 1:
        return

    def labels(bucket):
        return {record["label"] for record in bucket}

    for client_id, bucket in enumerate(buckets):
        attempts = 0
        while len(labels(bucket)) < min_labels and attempts < 200:
            attempts += 1
            current_labels = labels(bucket)
            donor_choice = None
            moved_record_idx = None
            donor_order = sorted(
                range(len(buckets)),
                key=lambda idx: (len(labels(buckets[idx])), len(buckets[idx])),
                reverse=True,
            )
            for donor_id in donor_order:
                if donor_id == client_id:
                    continue
                donor_bucket = buckets[donor_id]
                if len(labels(donor_bucket)) <= min_labels:
                    continue
                for idx, record in enumerate(donor_bucket):
                    if record["label"] not in current_labels:
                        remaining_labels = labels(donor_bucket[:idx] + donor_bucket[idx + 1:])
                        if len(remaining_labels) >= min_labels:
                            donor_choice = donor_id
                            moved_record_idx = idx
                            break
                if donor_choice is not None:
                    break
            if donor_choice is None:
                break
            bucket.append(buckets[donor_choice].pop(moved_record_idx))


def allocate_medium_train_clients(records, seed):
    rng = np.random.default_rng(seed)
    num_classes = len(label_names)
    num_conditions = len(condition_names)
    num_stages = len(stage_names)
    primary_conditions = np.arange(num_clients) % num_conditions
    primary_stages = np.arange(num_clients) % num_stages
    label_preferences = rng.dirichlet(np.repeat(medium_alpha, num_classes), size=num_clients)
    buckets = [[] for _ in range(num_clients)]

    groups = defaultdict(list)
    for record in records:
        key = (record["condition_id"], record["stage_id"], record["label"])
        groups[key].append(record)

    for (condition_id, stage_id, label_id), group_records in sorted(groups.items()):
        shuffled = [dict(record) for record in group_records]
        rng.shuffle(shuffled)

        condition_weights = np.where(primary_conditions == condition_id, 1.0, condition_affinity)
        stage_weights = np.where(primary_stages == stage_id, 1.0, stage_affinity)
        weights = label_preferences[:, label_id] * condition_weights * stage_weights
        weights = weights / weights.sum()
        counts = rng.multinomial(len(shuffled), weights)

        cursor = 0
        for client_id, count in enumerate(counts):
            if count <= 0:
                continue
            buckets[client_id].extend(shuffled[cursor:cursor + count])
            cursor += count

    for client_id, bucket in enumerate(buckets):
        if bucket:
            continue
        donor_id = int(np.argmax([len(items) for items in buckets]))
        bucket.append(buckets[donor_id].pop())

    enforce_min_labels_per_client(buckets, min_labels_per_client)
    for bucket in buckets:
        rng.shuffle(bucket)
    return buckets, primary_conditions, primary_stages, label_preferences


def records_to_window_dataset(records):
    X = []
    y = []
    by_path = defaultdict(list)
    for record in records:
        by_path[record["path"]].append(record)

    for path, path_records in by_path.items():
        signal = load_uo_signal(path)
        for record in path_records:
            windows = chunk_to_windows(signal, record["chunk_start"], record["chunk_length"])
            X.extend(windows)
            y.extend([record["label"]] * len(windows))

    if not X:
        return (
            np.empty((0, window_size, len(selected_channels)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def allocate_balanced_test_data(records, seed):
    rng = np.random.default_rng(seed)
    X_test_all, y_test_all = records_to_window_dataset(records)
    buckets = [[] for _ in range(num_clients)]

    for label_id in sorted(np.unique(y_test_all)):
        label_indices = np.where(y_test_all == label_id)[0]
        rng.shuffle(label_indices)
        for idx, sample_idx in enumerate(label_indices):
            buckets[idx % num_clients].append(int(sample_idx))

    test_data = []
    for bucket in buckets:
        indices = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(indices)
        test_data.append({"x": X_test_all[indices], "y": y_test_all[indices]})
    return test_data


def label_stats(y):
    unique, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(unique, counts)]


def summarize_records(client_records):
    statistic = []
    sizes = []
    for client_id, records in enumerate(client_records):
        estimated_windows = sum(record["window_count"] for record in records)
        sizes.append(estimated_windows)
        labels = sorted({record["label"] for record in records})
        conditions = sorted({record["condition_id"] for record in records})
        stages = sorted({record["stage_id"] for record in records})
        client_stat = []
        for label in labels:
            label_count = sum(record["window_count"] for record in records if record["label"] == label)
            client_stat.append((int(label), int(label_count)))
        statistic.append(client_stat)
        readable_labels = [label_names[label] for label in labels]
        readable_conditions = [condition_names[condition] for condition in conditions]
        readable_stages = [stage_names[stage] for stage in stages]
        print(
            f"Client {client_id}\t Size: {estimated_windows}\t "
            f"Conditions: {readable_conditions}\t Stages: {readable_stages}\t Labels: {readable_labels}"
        )
        print(f"\t\t Samples of labels: {client_stat}")
        print("-" * 50)
    print(f"Client size range: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
    return statistic


def plot_single_client_distribution(client_id, labels, output_prefix):
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed; skip label distribution figure generation.")
        return

    full_counts = np.zeros(len(label_names), dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        full_counts[int(label)] = int(count)

    x = np.arange(len(full_counts))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=plt.cm.Set3(np.linspace(0, 1, len(full_counts))), edgecolor="black")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"{dataset_display_name} Label Distribution for Client {client_id}")
    ax.set_xticks(x)
    ax.set_xticklabels([label_names[i] for i in x], rotation=15)
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


def generate_dataset(seed):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(dir_path)

    raw_dir_path = discover_raw_dir(raw_root_path)
    print(f"Using raw directory: {raw_dir_path}")

    source_records = load_all_source_records(raw_dir_path)
    train_records, test_records = split_source_records_by_file(source_records, seed)
    train_clients, primary_conditions, primary_stages, label_preferences = allocate_medium_train_clients(train_records, seed)
    test_data = allocate_balanced_test_data(test_records, seed + 1)

    print(f"Number of train source chunks: {len(train_records)}")
    print(f"Number of test source chunks: {len(test_records)}")
    print(f"Medium alpha: {medium_alpha}")
    print(f"Condition affinity: {condition_affinity}")
    print(f"Stage affinity: {stage_affinity}")
    print("Client mapping: mixed labels with soft primary condition/stage preferences")

    statistic = summarize_records(train_clients)
    train_counts = []
    test_counts = []
    client_labels_for_plot = []

    for client_id in range(num_clients):
        X_train, y_train = records_to_window_dataset(train_clients[client_id])
        X_test = test_data[client_id]["x"]
        y_test = test_data[client_id]["y"]

        with open(train_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_train, "y": y_train})
        with open(test_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_test, "y": y_test})

        train_counts.append(len(y_train))
        test_counts.append(len(y_test))
        client_labels_for_plot.append(np.concatenate([y_train, y_test], axis=0))
        print(
            f"Client {client_id}\t Train: {len(y_train):<6} Test: {len(y_test):<6} "
            f"Train labels: {label_stats(y_train)} Test labels: {label_stats(y_test)}"
        )

    config = {
        "num_clients": num_clients,
        "num_classes": len(label_names),
        "non_iid": True,
        "seed": seed,
        "dataset_variant": dataset_variant,
        "train_ratio": train_ratio,
        "dirichlet_alpha": medium_alpha,
        "condition_affinity": condition_affinity,
        "stage_affinity": stage_affinity,
        "min_labels_per_client": min_labels_per_client,
        "condition_per_client": "mixed_soft_primary",
        "stage_per_client": "mixed_soft_primary",
        "test_split_mode": "label_balanced_round_robin",
        "split_strategy": "source_chunk_train_test_then_client_allocation",
        "window_size": window_size,
        "window_stride": window_stride,
        "source_chunk_length": source_chunk_length,
        "input_shape": [window_size, len(selected_channels)],
        "selected_channels": selected_channels,
        "condition_names": condition_names,
        "stage_names": stage_names,
        "label_names": [label_names[idx] for idx in range(len(label_names))],
        "primary_conditions": [condition_names[int(idx)] for idx in primary_conditions],
        "primary_stages": [stage_names[int(idx)] for idx in primary_stages],
        "label_preferences": label_preferences.tolist(),
        "Size of samples for labels in clients": statistic,
    }
    with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if HAS_MATPLOTLIB:
        fig_dir = os.path.join(dir_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_single_client_distribution(
            0,
            client_labels_for_plot[0],
            os.path.join(fig_dir, f"{dataset_variant}_client_0_label_distribution"),
        )
    else:
        print("matplotlib is not installed; generated train/test data without figures.")

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Finish generating {dataset_display_name} dataset.")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
