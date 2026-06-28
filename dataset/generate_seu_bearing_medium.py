import json
import os
import random
import re
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


num_clients = 20
raw_dir_path = "dataset/seu/SEU-datasets/gearbox/bearingset/"
dir_path = "dataset/seu_bearing_medium/"

train_ratio = 0.7
medium_alpha = 0.3
condition_affinity = 0.25
min_labels_per_client = 2

window_size = 2048
window_stride = 2048
source_chunk_length = 32768
channel_indices = [1, 2, 3]  # README: rows/channels 2, 3, 4 are effective vibration signals.
dataset_variant = "seu_bearing_medium"
dataset_display_name = "SEU-Bearing-Medium"

label_names = {
    0: "health",
    1: "ball_fault",
    2: "inner_fault",
    3: "outer_fault",
    4: "comb_fault",
}

label_map = {
    "health": 0,
    "ball": 1,
    "inner": 2,
    "outer": 3,
    "comb": 4,
}

condition_names = ["20_0", "30_2"]


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


def parse_file_info(file_name):
    base = os.path.splitext(file_name)[0].lower()
    match = re.match(r"^(health|ball|inner|outer|comb)_(20_0|30_2)$", base)
    if not match:
        raise ValueError(f"Unsupported SEU bearing file: {file_name}")
    label_name, condition = match.groups()
    return label_map[label_name], condition


def find_data_start(path):
    with open(path, "r", encoding="latin1") as f:
        for idx, line in enumerate(f):
            stripped = line.strip().lower()
            first_cell = re.split(r"[,\s]+", stripped)[0] if stripped else ""
            if first_cell == "data":
                return idx + 1
    raise ValueError(f"No 'Data' marker found in {path}")


def detect_numeric_delimiter(path, skiprows):
    with open(path, "r", encoding="latin1") as f:
        for idx, line in enumerate(f):
            if idx < skiprows:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            return "," if "," in stripped else None
    return None


def load_seu_signal(path):
    skiprows = find_data_start(path)
    delimiter = detect_numeric_delimiter(path, skiprows)
    data = np.loadtxt(
        path,
        dtype=np.float32,
        delimiter=delimiter,
        skiprows=skiprows,
        usecols=channel_indices,
    )
    if data.ndim != 2 or data.shape[1] != len(channel_indices):
        raise ValueError(f"Unexpected signal shape {data.shape} in {path}")
    return data.astype(np.float32)


def split_signal_into_source_chunks(signal, chunk_length):
    usable_length = (len(signal) // chunk_length) * chunk_length
    if usable_length <= 0:
        return []
    signal = signal[:usable_length]
    return [
        signal[start:start + chunk_length].copy()
        for start in range(0, usable_length, chunk_length)
    ]


def count_windows(signal_length):
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // window_stride


def standardize_window(window):
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    return ((window - mean) / (std + 1e-8)).astype(np.float32)


def chunk_to_windows(signal):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_stride):
        window = signal[start:start + window_size]
        windows.append(standardize_window(window))
    return windows


def records_to_window_dataset(records):
    X = []
    y = []
    for record in records:
        windows = chunk_to_windows(record["signal"])
        X.extend(windows)
        y.extend([record["label"]] * len(windows))
    if not X:
        return (
            np.empty((0, window_size, len(channel_indices)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def load_all_source_records(raw_dir):
    records = []
    csv_files = sorted(name for name in os.listdir(raw_dir) if name.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {raw_dir}")

    for file_name in csv_files:
        label, condition = parse_file_info(file_name)
        path = os.path.join(raw_dir, file_name)
        signal = load_seu_signal(path)
        chunks = split_signal_into_source_chunks(signal, source_chunk_length)
        condition_id = condition_names.index(condition)
        for chunk_id, chunk in enumerate(chunks):
            records.append(
                {
                    "signal": chunk,
                    "label": label,
                    "condition": condition,
                    "condition_id": condition_id,
                    "source_file": file_name,
                    "source_chunk_id": chunk_id,
                }
            )
        print(
            f"Loaded {file_name:<18} label={label_names[label]:<12} "
            f"condition={condition:<4} chunks={len(chunks)} windows={sum(count_windows(len(c)) for c in chunks)}"
        )
    return records


def split_source_records_by_file(records, seed):
    rng = random.Random(seed)
    train_records = []
    test_records = []
    groups = {}
    for record in records:
        groups.setdefault(record["source_file"], []).append(record)

    for source_file, items in sorted(groups.items()):
        items = [dict(record) for record in items]
        items.sort(key=lambda record: record["source_chunk_id"])
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


def allocate_medium_train_clients(records, seed):
    rng = np.random.default_rng(seed)
    num_classes = len(label_names)
    num_conditions = len(condition_names)
    primary_conditions = np.arange(num_clients) % num_conditions
    label_preferences = rng.dirichlet(np.repeat(medium_alpha, num_classes), size=num_clients)
    buckets = [[] for _ in range(num_clients)]

    groups = {}
    for record in records:
        key = (record["condition_id"], record["label"])
        groups.setdefault(key, []).append(record)

    for (condition_id, label_id), group_records in sorted(groups.items()):
        shuffled = [dict(record) for record in group_records]
        rng.shuffle(shuffled)

        condition_weights = np.where(primary_conditions == condition_id, 1.0, condition_affinity)
        weights = label_preferences[:, label_id] * condition_weights
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
    return buckets, primary_conditions, label_preferences


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
        estimated_windows = sum(count_windows(len(record["signal"])) for record in records)
        sizes.append(estimated_windows)
        labels = sorted({record["label"] for record in records})
        conditions = sorted({record["condition_id"] for record in records})
        client_stat = []
        for label in labels:
            label_count = sum(count_windows(len(record["signal"])) for record in records if record["label"] == label)
            client_stat.append((int(label), int(label_count)))
        statistic.append(client_stat)
        readable_labels = [label_names[label] for label in labels]
        readable_conditions = [condition_names[condition] for condition in conditions]
        print(
            f"Client {client_id}\t Size: {estimated_windows}\t "
            f"Conditions: {readable_conditions}\t Labels: {readable_labels}"
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

    source_records = load_all_source_records(raw_dir_path)
    train_records, test_records = split_source_records_by_file(source_records, seed)
    train_clients, primary_conditions, label_preferences = allocate_medium_train_clients(train_records, seed)
    test_data = allocate_balanced_test_data(test_records, seed + 1)

    print(f"Number of train source chunks: {len(train_records)}")
    print(f"Number of test source chunks: {len(test_records)}")
    print(f"Medium alpha: {medium_alpha}")
    print(f"Condition affinity: {condition_affinity}")
    print("Client-condition mapping: mixed conditions with a soft primary-condition preference")

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
        "min_labels_per_client": min_labels_per_client,
        "condition_per_client": "mixed_soft_primary",
        "test_split_mode": "label_balanced_round_robin",
        "split_strategy": "source_chunk_train_test_then_client_allocation",
        "window_size": window_size,
        "window_stride": window_stride,
        "source_chunk_length": source_chunk_length,
        "input_shape": [window_size, len(channel_indices)],
        "selected_channels": ["Channel2", "Channel3", "Channel4"],
        "condition_names": condition_names,
        "label_names": [label_names[idx] for idx in range(len(label_names))],
        "primary_conditions": [condition_names[int(idx)] for idx in primary_conditions],
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
