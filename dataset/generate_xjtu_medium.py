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
dir_path = "dataset/xjtu_medium/"

train_ratio = 0.7
medium_alpha = 0.3
condition_affinity = 0.25
min_labels_per_client = 2
dataset_variant = "xjtu_medium"
dataset_display_name = "XJTU-Medium"
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
        item["split"] = "medium_pool"
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


def record_condition_id(record):
    condition_names = sorted(set(base.condition_dirs.values()))
    return condition_names.index(record["condition"])


def allocate_medium_train_clients(records, seed):
    rng = np.random.default_rng(seed)
    num_classes = len(label_names)
    num_conditions = len(set(base.condition_dirs.values()))
    primary_conditions = np.arange(num_clients) % num_conditions
    label_preferences = rng.dirichlet(np.repeat(medium_alpha, num_classes), size=num_clients)
    buckets = [[] for _ in range(num_clients)]

    groups = {}
    for record in records:
        key = (record_condition_id(record), record["label"])
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


def records_to_dataset(records):
    return base.records_to_dataset(records)


def allocate_balanced_test_data(records, seed):
    rng = np.random.default_rng(seed)
    X_test_all, y_test_all = records_to_dataset(records)
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
    ax.set_title(f"{dataset_display_name} Label Distribution for Client {client_id}")
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
    train_clients, primary_conditions, label_preferences = allocate_medium_train_clients(train_records, seed)
    test_data = allocate_balanced_test_data(test_records, seed + 1)

    train_counts = []
    test_counts = []
    client_labels_for_plot = []
    statistic = []
    condition_names = sorted(set(base.condition_dirs.values()))

    for client_id in range(num_clients):
        X_train, y_train = records_to_dataset(train_clients[client_id])
        X_test = test_data[client_id]["x"]
        y_test = test_data[client_id]["y"]

        with open(train_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_train, "y": y_train})
        with open(test_path + str(client_id) + ".npz", "wb") as f:
            np.savez_compressed(f, data={"x": X_test, "y": y_test})

        train_counts.append(len(y_train))
        test_counts.append(len(y_test))
        combined_y = np.concatenate([y_train, y_test], axis=0)
        client_labels_for_plot.append(combined_y)
        statistic.append(label_stats(combined_y))
        client_conditions = sorted({record_condition_id(record) for record in train_clients[client_id]})
        print(
            f"Client {client_id}\t Conditions: {[condition_names[i] for i in client_conditions]}\t "
            f"Train: {len(y_train):<6} Test: {len(y_test):<6} "
            f"Train labels: {label_stats(y_train)} Test labels: {label_stats(y_test)}"
        )

    fig_dir = os.path.join(dir_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_single_client_distribution(0, client_labels_for_plot[0], os.path.join(fig_dir, f"{dataset_variant}_client_0_label_distribution"))

    config = {
        "num_clients": num_clients,
        "num_classes": len(label_names),
        "non_iid": True,
        "seed": seed,
        "dataset_variant": dataset_variant,
        "split_strategy": "source_csv_label_balanced_then_medium_client_allocation",
        "train_ratio": train_ratio,
        "dirichlet_alpha": medium_alpha,
        "condition_affinity": condition_affinity,
        "min_labels_per_client": min_labels_per_client,
        "window_size": window_size,
        "window_stride": window_stride,
        "input_shape": [window_size, 2],
        "label_names": [label_names[i] for i in range(len(label_names))],
        "condition_names": condition_names,
        "primary_conditions": [condition_names[int(idx)] for idx in primary_conditions],
        "label_preferences": label_preferences.tolist(),
        "per_label_source_count": per_label_source_count,
        "raw_label_rule": "normal: first 20 percent; fault: last 30 percent",
        "test_split_mode": "label_balanced_window_round_robin",
        "Size of samples for labels in clients": statistic,
    }
    with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Finish generating {dataset_display_name} dataset.")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
