import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_hust as base


num_clients = 20
raw_dir_path = "dataset/hust/Raw_data/"
dir_path = "dataset/hust_mild/"

train_ratio = 0.7
mild_alpha = 0.5
condition_affinity = 0.35
window_size = base.window_size
window_stride = base.window_stride
source_chunk_length = base.source_chunk_length
signal_channel = base.signal_channel
target_conditions = base.target_conditions
label_names = base.label_names


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


def split_source_records(records, seed):
    rng = np.random.default_rng(seed)
    train_records = []
    test_records = []

    groups = {}
    for record in records:
        key = (record["condition_id"], record["label"])
        groups.setdefault(key, []).append(record)

    for key, group_records in sorted(groups.items()):
        shuffled = [dict(record) for record in group_records]
        rng.shuffle(shuffled)
        train_count = min(max(int(round(len(shuffled) * train_ratio)), 1), len(shuffled) - 1)
        train_records.extend(shuffled[:train_count])
        test_records.extend(shuffled[train_count:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def allocate_mild_train_clients(train_records, seed):
    rng = np.random.default_rng(seed)
    num_classes = len(label_names)
    primary_conditions = np.arange(num_clients) % len(target_conditions)
    label_preferences = rng.dirichlet(np.repeat(mild_alpha, num_classes), size=num_clients)
    buckets = [[] for _ in range(num_clients)]

    groups = {}
    for record in train_records:
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

    # Avoid empty clients in rare random draws by moving one record from the largest bucket.
    for client_id, bucket in enumerate(buckets):
        if bucket:
            continue
        donor_id = int(np.argmax([len(items) for items in buckets]))
        bucket.append(buckets[donor_id].pop())

    for bucket in buckets:
        rng.shuffle(bucket)
    return buckets, primary_conditions, label_preferences


def allocate_balanced_test_data(test_records, seed):
    rng = np.random.default_rng(seed)
    X_test_all, y_test_all = records_to_window_dataset(test_records)
    buckets = [[] for _ in range(num_clients)]

    for label_id in sorted(np.unique(y_test_all)):
        label_indices = np.where(y_test_all == label_id)[0]
        rng.shuffle(label_indices)
        for idx, sample_idx in enumerate(label_indices):
            buckets[idx % num_clients].append(int(sample_idx))

    test_data = []
    for bucket in buckets:
        indices = np.array(bucket, dtype=np.int64)
        rng.shuffle(indices)
        test_data.append({"x": X_test_all[indices], "y": y_test_all[indices]})
    return test_data


def records_to_window_dataset(records):
    return base.records_to_window_dataset(records)


def label_stats(y):
    unique, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(unique, counts)]


def summarize_records(client_records, condition_names):
    statistic = []
    sizes = []
    for client_id, records in enumerate(client_records):
        estimated_windows = sum(base.count_windows(len(record["signal"])) for record in records)
        sizes.append(estimated_windows)
        labels = sorted({record["label"] for record in records})
        conditions = sorted({record["condition_id"] for record in records})
        client_stat = []
        for label in labels:
            label_count = sum(base.count_windows(len(record["signal"])) for record in records if record["label"] == label)
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
    full_counts = np.zeros(len(label_names), dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        full_counts[int(label)] = int(count)

    x = np.arange(len(full_counts))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=plt.cm.Set3(np.linspace(0, 1, len(full_counts))), edgecolor="black")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"HUST-Mild Label Distribution for Client {client_id}")
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
    config_path = os.path.join(dir_path, "config.json")

    source_records, condition_names = base.load_all_conditions(raw_dir_path)
    train_records, test_records = split_source_records(source_records, seed)
    train_clients, primary_conditions, label_preferences = allocate_mild_train_clients(train_records, seed)
    test_data = allocate_balanced_test_data(test_records, seed)

    print(f"Number of train source chunks: {len(train_records)}")
    print(f"Number of test source chunks: {len(test_records)}")
    print(f"Mild alpha: {mild_alpha}")
    print(f"Condition affinity: {condition_affinity}")
    print("Client-condition mapping: mixed conditions with a soft primary-condition preference")

    statistic = summarize_records(train_clients, condition_names)

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
        "dataset_variant": "hust_mild",
        "train_ratio": train_ratio,
        "dirichlet_alpha": mild_alpha,
        "condition_affinity": condition_affinity,
        "condition_per_client": "mixed_soft_primary",
        "test_split_mode": "label_balanced_round_robin",
        "split_strategy": "source_chunk_train_test_then_client_allocation",
        "window_size": window_size,
        "window_stride": window_stride,
        "source_chunk_length": source_chunk_length,
        "signal_channel": signal_channel,
        "fault_severity": "0.5X for faulty classes; H has no severity prefix",
        "condition_names": condition_names,
        "label_names": [label_names[idx] for idx in range(len(label_names))],
        "primary_conditions": [condition_names[int(idx)] for idx in primary_conditions],
        "label_preferences": label_preferences.tolist(),
        "Size of samples for labels in clients": statistic,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    fig_dir = os.path.join(dir_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_single_client_distribution(0, client_labels_for_plot[0], os.path.join(fig_dir, "hust_mild_client_0_label_distribution"))

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print("Finish generating HUST-Mild dataset.")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
