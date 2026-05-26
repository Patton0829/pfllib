import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_xjtu as base


num_clients = 20
raw_dir_path = base.raw_dir_path
dir_path = "dataset/xjtu_balanced/"

train_ratio = 0.7
window_size = base.window_size
window_stride = base.window_stride
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


def balance_records_by_label(records, seed):
    rng = random.Random(seed)
    by_label = {}
    for record in records:
        item = dict(record)
        item["split"] = "balanced_pool"
        by_label.setdefault(item["label"], []).append(item)

    min_count = min(len(items) for items in by_label.values())
    balanced = []
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        balanced.extend(items[:min_count])

    rng.shuffle(balanced)
    return balanced, min_count


def split_train_test_by_label(records, seed):
    rng = random.Random(seed)
    train_records = []
    test_records = []
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    for label in sorted(by_label):
        items = [dict(record) for record in by_label[label]]
        rng.shuffle(items)
        train_count = min(max(int(round(len(items) * train_ratio)), 1), len(items) - 1)
        train_records.extend(items[:train_count])
        test_records.extend(items[train_count:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def allocate_balanced_clients(records, seed):
    rng = random.Random(seed)
    buckets = [[] for _ in range(num_clients)]
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)

    for label in sorted(by_label):
        items = [dict(record) for record in by_label[label]]
        rng.shuffle(items)
        for idx, record in enumerate(items):
            buckets[idx % num_clients].append(record)

    for bucket in buckets:
        rng.shuffle(bucket)
    return buckets


def records_to_dataset(records):
    return base.records_to_dataset(records)


def label_stats(y):
    unique, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(unique, counts)]


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
    ax.set_title(f"XJTU-Balanced Label Distribution for Client {client_id}")
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


def generate_dataset(seed):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(dir_path)

    all_records = base.build_records(raw_dir_path)
    balanced_records, per_label_source_count = balance_records_by_label(all_records, seed)
    train_records, test_records = split_train_test_by_label(balanced_records, seed)
    train_clients = allocate_balanced_clients(train_records, seed)
    test_clients = allocate_balanced_clients(test_records, seed + 1)

    train_counts = []
    test_counts = []
    client_labels_for_plot = []
    statistic = []

    for client_id in range(num_clients):
        X_train, y_train = records_to_dataset(train_clients[client_id])
        X_test, y_test = records_to_dataset(test_clients[client_id])

        with open(train_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_train, "y": y_train})
        with open(test_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_test, "y": y_test})

        train_counts.append(len(y_train))
        test_counts.append(len(y_test))
        combined_y = np.concatenate([y_train, y_test], axis=0)
        client_labels_for_plot.append(combined_y)
        statistic.append(label_stats(combined_y))
        print(
            f"Client {client_id}\t Train: {len(y_train):<6} Test: {len(y_test):<6} "
            f"Train labels: {label_stats(y_train)} Test labels: {label_stats(y_test)}"
        )

    fig_dir = os.path.join(dir_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_single_client_distribution(
        0,
        client_labels_for_plot[0],
        os.path.join(fig_dir, "xjtu_balanced_client_0_label_distribution"),
    )

    config = {
        "num_clients": num_clients,
        "num_classes": len(label_names),
        "non_iid": False,
        "seed": seed,
        "dataset_variant": "xjtu_balanced",
        "split_strategy": "source_csv_label_balanced",
        "train_ratio": train_ratio,
        "window_size": window_size,
        "window_stride": window_stride,
        "input_shape": [window_size, 2],
        "label_names": [label_names[i] for i in range(len(label_names))],
        "per_label_source_count": per_label_source_count,
        "raw_label_rule": "normal: first 20 percent; fault: last 30 percent",
        "client_allocation": "round-robin balanced by label for train and test",
        "Size of samples for labels in clients": statistic,
    }
    with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print("Finish generating XJTU-Balanced dataset.")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
