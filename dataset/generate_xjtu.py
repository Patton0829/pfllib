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
raw_dir_path = "dataset/xjtu/XJTU-SY_Bearing_Datasets/"
dir_path = "dataset/xjtu/"

window_size = 2048
window_stride = 1024
train_ratio_description = "bearing-level fixed split"
test_split_mode = "bearing_level"
split_strategy = "by_bearing"

label_names = {
    0: "Normal",
    1: "Outer",
    2: "Inner",
    3: "Cage",
    4: "Compound",
}

condition_dirs = {
    "Bearing1": "35Hz12kN",
    "Bearing2": "37.5Hz11kN",
    "Bearing3": "40Hz10kN",
}

bearing_file_counts = {
    "Bearing1_1": 123,
    "Bearing1_2": 161,
    "Bearing1_3": 158,
    "Bearing1_4": 122,
    "Bearing1_5": 52,
    "Bearing2_1": 491,
    "Bearing2_2": 161,
    "Bearing2_3": 533,
    "Bearing2_4": 42,
    "Bearing2_5": 339,
    "Bearing3_1": 2538,
    "Bearing3_2": 2496,
    "Bearing3_3": 371,
    "Bearing3_4": 1515,
    "Bearing3_5": 114,
}

fault_bearings = {
    1: ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing2_2", "Bearing2_4", "Bearing2_5", "Bearing3_1", "Bearing3_5"],
    2: ["Bearing2_1", "Bearing3_3", "Bearing3_4"],
    3: ["Bearing1_4", "Bearing2_3"],
    4: ["Bearing1_5", "Bearing3_2"],
}

train_bearings = {
    "Bearing1_1",
    "Bearing1_2",
    "Bearing2_2",
    "Bearing2_5",
    "Bearing3_1",
    "Bearing2_1",
    "Bearing3_3",
    "Bearing2_3",
    "Bearing3_2",
}

test_bearings = {
    "Bearing1_3",
    "Bearing2_4",
    "Bearing3_5",
    "Bearing3_4",
    "Bearing1_4",
    "Bearing1_5",
}


def bearing_condition_name(bearing_name):
    prefix = bearing_name.split("_")[0]
    return condition_dirs[prefix]


def bearing_path(raw_root, bearing_name):
    return os.path.join(raw_root, bearing_condition_name(bearing_name), bearing_name)


def normal_indices(total_count):
    end = int(np.floor(0.2 * total_count))
    return range(1, end + 1)


def fault_indices(total_count):
    start = int(np.floor(0.7 * total_count)) + 1
    return range(start, total_count + 1)


def fault_label_for_bearing(bearing_name):
    for label_id, bearings in fault_bearings.items():
        if bearing_name in bearings:
            return label_id
    raise ValueError(f"No fault label configured for {bearing_name}")


def load_csv_signal(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 vibration columns in {csv_path}, got shape {data.shape}")
    return data.astype(np.float32)


def segment_signal(signal):
    if signal.shape[0] < window_size:
        raise ValueError(f"Signal too short for windowing: {signal.shape[0]} < {window_size}")
    segments = []
    for start in range(0, signal.shape[0] - window_size + 1, window_stride):
        segments.append(signal[start:start + window_size, :])
    return np.asarray(segments, dtype=np.float32)


def standardize_segments(segments):
    # Normalize each channel in each window independently.
    mean = segments.mean(axis=1, keepdims=True)
    std = np.maximum(segments.std(axis=1, keepdims=True), 1e-6)
    return ((segments - mean) / std).astype(np.float32)


def load_windows_for_csv(csv_path):
    return standardize_segments(segment_signal(load_csv_signal(csv_path)))


def build_records(raw_root):
    records = []
    for bearing_name, total_count in sorted(bearing_file_counts.items()):
        b_path = bearing_path(raw_root, bearing_name)
        if not os.path.isdir(b_path):
            raise FileNotFoundError(f"Missing bearing directory: {b_path}")

        split = "train" if bearing_name in train_bearings else "test"
        if bearing_name not in train_bearings and bearing_name not in test_bearings:
            raise ValueError(f"Bearing {bearing_name} is not assigned to train or test.")

        for file_idx in normal_indices(total_count):
            records.append(make_record(bearing_name, b_path, file_idx, 0, split, stage="early_normal"))

        fault_label = fault_label_for_bearing(bearing_name)
        for file_idx in fault_indices(total_count):
            records.append(make_record(bearing_name, b_path, file_idx, fault_label, split, stage="late_fault"))

    return records


def make_record(bearing_name, bearing_dir, file_idx, label_id, split, stage):
    csv_path = os.path.join(bearing_dir, f"{file_idx}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing csv file: {csv_path}")
    return {
        "bearing": bearing_name,
        "condition": bearing_condition_name(bearing_name),
        "file_idx": int(file_idx),
        "csv_path": csv_path,
        "label": int(label_id),
        "split": split,
        "stage": stage,
    }


def records_to_dataset(records):
    xs = []
    ys = []
    for record in records:
        segments = load_windows_for_csv(record["csv_path"])
        xs.append(segments)
        ys.append(np.full(len(segments), record["label"], dtype=np.int64))
    if not xs:
        return np.empty((0, window_size, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def split_sequence(items, parts):
    if parts <= 0:
        return []
    chunks = []
    for idx in range(parts):
        start = idx * len(items) // parts
        end = (idx + 1) * len(items) // parts
        chunks.append(items[start:end])
    return chunks


def proportional_client_counts(group_sizes, total_clients):
    groups = list(group_sizes.keys())
    counts = {group: 1 for group in groups}
    remaining = total_clients - len(groups)
    if remaining < 0:
        raise ValueError(f"Need at least {len(groups)} clients, got {total_clients}.")

    total_size = sum(group_sizes.values())
    fractional = []
    for group in groups:
        exact_extra = remaining * group_sizes[group] / max(total_size, 1)
        extra = int(np.floor(exact_extra))
        counts[group] += extra
        fractional.append((exact_extra - extra, group))

    assigned = sum(counts.values())
    for _, group in sorted(fractional, reverse=True):
        if assigned >= total_clients:
            break
        counts[group] += 1
        assigned += 1

    return counts


def allocate_train_records_by_bearing(records, total_clients):
    train_records = [record for record in records if record["split"] == "train"]
    by_bearing = {}
    for record in train_records:
        by_bearing.setdefault(record["bearing"], []).append(record)

    for bearing_records in by_bearing.values():
        bearing_records.sort(key=lambda item: (item["label"], item["file_idx"]))

    client_counts = proportional_client_counts(
        {bearing: len(bearing_records) for bearing, bearing_records in by_bearing.items()},
        total_clients,
    )

    clients = []
    for bearing in sorted(by_bearing):
        for chunk in split_sequence(by_bearing[bearing], client_counts[bearing]):
            clients.append(chunk)

    if len(clients) != total_clients:
        raise RuntimeError(f"Expected {total_clients} train clients, got {len(clients)}")
    return clients


def allocate_test_records_balanced(records, total_clients, seed):
    rng = random.Random(seed)
    test_records = [record for record in records if record["split"] == "test"]
    by_label = {}
    for record in test_records:
        by_label.setdefault(record["label"], []).append(record)

    clients = [[] for _ in range(total_clients)]
    for label in sorted(by_label):
        label_records = by_label[label]
        rng.shuffle(label_records)
        for idx, record in enumerate(label_records):
            clients[idx % total_clients].append(record)

    for client_records in clients:
        client_records.sort(key=lambda item: (item["label"], item["bearing"], item["file_idx"]))
    return clients


def build_clients(records, total_clients, seed):
    train_clients = allocate_train_records_by_bearing(records, total_clients)
    test_clients = allocate_test_records_balanced(records, total_clients, seed)
    return [
        {"train": train_clients[idx], "test": test_clients[idx]}
        for idx in range(total_clients)
    ]


def label_stats(y):
    unique, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(unique, counts)]


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


def plot_single_client_distribution(client_id, labels, output_prefix):
    full_counts = np.zeros(len(label_names), dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        full_counts[int(label)] = int(count)

    x = np.arange(len(full_counts))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=plt.cm.Set3(np.linspace(0, 1, len(full_counts))), edgecolor="black")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"XJTU Label Distribution for Client {client_id}")
    ax.set_xticks(x)
    ax.set_xticklabels([label_names[i] for i in x])
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
            f"{count}\n{100 * count / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_distribution_figure(base_dir, client_labels, client_id=0):
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    client_id = max(0, min(client_id, len(client_labels) - 1))
    plot_single_client_distribution(
        client_id,
        client_labels[client_id],
        os.path.join(fig_dir, f"xjtu_client_{client_id}_label_distribution"),
    )


def generate_dataset(base_dir, raw_root, seed):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(base_dir)

    records = build_records(raw_root)
    clients = build_clients(records, num_clients, seed)

    train_data = []
    test_data = []
    client_labels = []
    statistic = []
    train_counts = []
    test_counts = []

    for client_id, client in enumerate(clients):
        X_train, y_train = records_to_dataset(client["train"])
        X_test, y_test = records_to_dataset(client["test"])
        train_data.append({"x": X_train, "y": y_train})
        test_data.append({"x": X_test, "y": y_test})
        combined_y = np.concatenate([y_train, y_test], axis=0)
        client_labels.append(combined_y)
        statistic.append(label_stats(combined_y))
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))

        train_bearings_for_client = sorted({record["bearing"] for record in client["train"]})
        test_bearings_for_client = sorted({record["bearing"] for record in client["test"]})
        print(
            f"Client {client_id}\t Train bearings: {','.join(train_bearings_for_client):<18} "
            f"Test bearings: {','.join(test_bearings_for_client):<42} "
            f"Train: {len(y_train):<6} Test: {len(y_test):<6} Labels: {statistic[-1]}"
        )

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=test_dict)

    config = {
        "num_clients": len(clients),
        "num_classes": len(label_names),
        "non_iid": True,
        "seed": seed,
        "split_strategy": split_strategy,
        "test_split_mode": test_split_mode,
        "train_ratio": train_ratio_description,
        "window_size": window_size,
        "window_stride": window_stride,
        "input_shape": [window_size, 2],
        "label_names": [label_names[i] for i in range(len(label_names))],
        "normal_rule": "first 20 percent of each bearing lifetime",
        "fault_rule": "last 30 percent of each bearing lifetime",
        "discard_rule": "middle 50 percent discarded",
        "train_bearings": sorted(train_bearings),
        "test_bearings": sorted(test_bearings),
        "Size of samples for labels in clients": statistic,
    }
    with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    save_distribution_figure(base_dir, client_labels, client_id=0)
    print("Finish generating XJTU dataset.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        raise SystemExit(
            "Usage: python dataset/generate_xjtu.py [seed]\n"
            "Generates a five-class XJTU-SY bearing dataset using fixed bearing-level train/test split."
        )

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(dir_path, raw_dir_path, seed)
