import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_cwru import load_all_conditions as load_cwru_conditions
from generate_jnu import (
    allocate_condition_label_skew,
    allocate_condition_random,
    build_jittered_quotas,
    train_test_split_fixed_test_count,
    train_test_split_np,
)
from generate_jnu import load_all_conditions as load_jnu_conditions


num_clients = 20
num_classes = 4
jnu_clients = 10
cwru_clients = 10
jnu_raw_dir_path = "dataset/jnu/JNU-Bearing-Dataset/"
cwru_raw_dir_path = "dataset/cwru/"
dir_path = "dataset/jnu_cwru_mix/"
train_ratio = 0.8
dirichlet_alpha = 0.05
size_jitter_ratio = 0.25
condition_profile = "severe"
test_split_mode = "balanced"

label_names = {
    0: "normal",
    1: "inner_race_fault",
    2: "outer_race_fault",
    3: "ball_fault",
}

# Inference based on the common CWRU 10-class ordering used in bearing diagnosis:
# [normal, ball_007, ball_014, ball_021, inner_007, inner_014, inner_021, outer_007, outer_014, outer_021]
CWRU_SUPERCLASS_MAP = {
    0: 0,
    1: 3,
    2: 3,
    3: 3,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
}


def prepare_output_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    train_path = os.path.join(base_dir, "train")
    test_path = os.path.join(base_dir, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for split_path in [train_path, test_path]:
        for file_name in os.listdir(split_path):
            if file_name.endswith(".npz"):
                os.remove(os.path.join(split_path, file_name))

    return train_path + os.sep, test_path + os.sep


def plot_single_client_distribution(client_id, client_labels, output_prefix):
    unique_labels, counts = np.unique(client_labels, return_counts=True)
    full_counts = np.zeros(num_classes, dtype=np.int64)
    for label, count in zip(unique_labels, counts):
        full_counts[int(label)] = int(count)

    tick_labels = [label_names[idx] for idx in range(num_classes)]
    x = np.arange(num_classes)
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))

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


def save_distribution_figure(base_dir, client_labels, client_id=0):
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    client_id = max(0, min(client_id, len(client_labels) - 1))
    plot_single_client_distribution(
        client_id,
        client_labels[client_id],
        os.path.join(fig_dir, f"jnu_cwru_mix_client_{client_id}_label_distribution"),
    )


def get_clients_per_condition_for_domain(domain_num_clients, num_conditions, profile):
    if profile == "balanced":
        base = domain_num_clients // num_conditions
        counts = [base] * num_conditions
        for i in range(domain_num_clients % num_conditions):
            counts[i] += 1
        return counts

    if profile == "moderate":
        if domain_num_clients == 10 and num_conditions == 3:
            return [4, 3, 3]
        if domain_num_clients == 10 and num_conditions == 4:
            return [3, 3, 2, 2]

    if profile == "severe":
        if domain_num_clients == 10 and num_conditions == 3:
            return [5, 3, 2]
        if domain_num_clients == 10 and num_conditions == 4:
            return [4, 3, 2, 1]

    raise ValueError(
        f"Unsupported domain client allocation: domain_num_clients={domain_num_clients}, "
        f"num_conditions={num_conditions}, profile={profile}"
    )


def map_cwru_labels_to_superclasses(cwru_y):
    mapped = np.array([CWRU_SUPERCLASS_MAP[int(label)] for label in cwru_y], dtype=np.int64)
    return mapped


def build_domain_condition_blocks():
    jnu_x, jnu_y, jnu_cond, jnu_condition_names = load_jnu_conditions(jnu_raw_dir_path)
    cwru_x, cwru_y, cwru_cond, cwru_condition_names = load_cwru_conditions(cwru_raw_dir_path)
    cwru_y = map_cwru_labels_to_superclasses(cwru_y)

    blocks = []

    for condition_id, condition_name in enumerate(jnu_condition_names):
        idxs = np.where(jnu_cond == condition_id)[0]
        blocks.append(
            {
                "domain": "jnu",
                "condition_name": f"jnu_{condition_name}rpm",
                "x": jnu_x[idxs],
                "y": jnu_y[idxs],
            }
        )

    for condition_id, condition_name in enumerate(cwru_condition_names):
        idxs = np.where(cwru_cond == condition_id)[0]
        blocks.append(
            {
                "domain": "cwru",
                "condition_name": f"cwru_{condition_name}",
                "x": cwru_x[idxs],
                "y": cwru_y[idxs],
            }
        )

    return blocks


def allocate_clients_for_block(block_x, block_y, num_block_clients, niid):
    condition_indices = np.arange(len(block_y), dtype=np.int64)
    quotas = build_jittered_quotas(len(condition_indices), num_block_clients, size_jitter_ratio)
    if niid:
        local_client_indices = allocate_condition_label_skew(
            condition_indices, block_y, quotas, num_classes, dirichlet_alpha
        )
    else:
        local_client_indices = allocate_condition_random(condition_indices, quotas)

    clients_x = [block_x[idxs] for idxs in local_client_indices]
    clients_y = [block_y[idxs] for idxs in local_client_indices]
    return clients_x, clients_y


def split_data_custom(X, y, seed):
    train_data, test_data = [], []
    train_counts, test_counts = [], []
    client_sizes = [len(labels) for labels in y]

    if test_split_mode == "balanced":
        total_test_samples = int(round(sum(client_sizes) * (1 - train_ratio)))
        target_test_count = max(1, total_test_samples // len(y))
        print(f"Test split mode: balanced")
        print(f"Target test samples per client: {target_test_count}")
    else:
        target_test_count = None
        print(f"Test split mode: proportional")

    for client_id in range(len(y)):
        _, counts = np.unique(y[client_id], return_counts=True)
        stratify_labels = y[client_id] if np.min(counts) >= 2 else None

        if test_split_mode == "balanced":
            fixed_test_count = min(target_test_count, len(y[client_id]) - 1)
            x_train, x_test, y_train, y_test = train_test_split_fixed_test_count(
                X[client_id],
                y[client_id],
                fixed_test_count,
                stratify=stratify_labels,
                seed=seed + client_id,
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split_np(
                X[client_id],
                y[client_id],
                train_ratio,
                stratify=stratify_labels,
                seed=seed + client_id,
            )

        train_data.append({"x": x_train, "y": y_train})
        test_data.append({"x": x_test, "y": y_test})
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Train/Test ratio: {train_ratio:.1%}/{1-train_ratio:.1%}")
    print()

    return train_data, test_data


def summarize_clients(client_labels, client_domains, client_conditions):
    statistic = []
    sizes = []
    for client_id in range(len(client_labels)):
        labels = client_labels[client_id]
        client_stat = []
        for label in np.unique(labels):
            client_stat.append((int(label), int(np.sum(labels == label))))
        statistic.append(client_stat)
        sizes.append(len(labels))
        readable_labels = [label_names[int(label)] for label in np.unique(labels)]
        print(
            f"Client {client_id}\t Size of data: {len(labels)}\t "
            f"Domain: {client_domains[client_id]}\t Condition: {client_conditions[client_id]}\t "
            f"Labels: {readable_labels}"
        )
        print(f"\t\t Samples of labels: {client_stat}")
        print("-" * 50)

    print(f"Client size range: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
    return statistic


def save_file_custom(config_path, train_path, test_path, train_data, test_data, statistic, seed, client_domains, client_conditions):
    config = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "seed": seed,
        "train_ratio": train_ratio,
        "dirichlet_alpha": dirichlet_alpha,
        "size_jitter_ratio": size_jitter_ratio,
        "condition_profile": condition_profile,
        "test_split_mode": test_split_mode,
        "dataset_type": "mixed_domain_signal",
        "client_domain_mode": "single_domain_single_condition",
        "domain_split": {"jnu": jnu_clients, "cwru": cwru_clients},
        "client_domains": client_domains,
        "client_conditions": client_conditions,
        "label_names": [label_names[idx] for idx in range(num_classes)],
        "cwru_superclass_map": CWRU_SUPERCLASS_MAP,
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

    print("Finish generating mixed-domain dataset.\n")


def generate_dataset(seed, niid):
    random.seed(seed)
    np.random.seed(seed)
    train_path, test_path = prepare_output_dirs(dir_path)
    config_path = dir_path + "config.json"

    blocks = build_domain_condition_blocks()
    jnu_blocks = [block for block in blocks if block["domain"] == "jnu"]
    cwru_blocks = [block for block in blocks if block["domain"] == "cwru"]

    jnu_clients_per_condition = get_clients_per_condition_for_domain(jnu_clients, len(jnu_blocks), condition_profile)
    cwru_clients_per_condition = get_clients_per_condition_for_domain(cwru_clients, len(cwru_blocks), condition_profile)

    client_x = []
    client_y = []
    client_domains = []
    client_conditions = []

    for block, block_clients in zip(jnu_blocks, jnu_clients_per_condition):
        local_x, local_y = allocate_clients_for_block(block["x"], block["y"], block_clients, niid)
        client_x.extend(local_x)
        client_y.extend(local_y)
        client_domains.extend([block["domain"]] * len(local_x))
        client_conditions.extend([block["condition_name"]] * len(local_x))

    for block, block_clients in zip(cwru_blocks, cwru_clients_per_condition):
        local_x, local_y = allocate_clients_for_block(block["x"], block["y"], block_clients, niid)
        client_x.extend(local_x)
        client_y.extend(local_y)
        client_domains.extend([block["domain"]] * len(local_x))
        client_conditions.extend([block["condition_name"]] * len(local_x))

    if len(client_x) != num_clients:
        raise ValueError(f"Expected {num_clients} clients, but got {len(client_x)}")

    print(f"Number of clients: {num_clients}")
    print(f"Domain split: JNU={jnu_clients}, CWRU={cwru_clients}")
    print(f"Number of classes after alignment: {num_classes}")
    print(f"Condition profile: {condition_profile}")
    print(f"Size jitter ratio: {size_jitter_ratio}")
    print(f"Test split mode: {test_split_mode}")
    print("Client setup: one client belongs to exactly one domain and one condition")

    statistic = summarize_clients(client_y, client_domains, client_conditions)
    train_data, test_data = split_data_custom(client_x, client_y, seed)
    save_file_custom(config_path, train_path, test_path, train_data, test_data, statistic, seed, client_domains, client_conditions)
    save_distribution_figure(dir_path, client_y, client_id=0)
    print(f"Saved figures to {os.path.join(dir_path, 'figures')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python generate_jnu_cwru_mix.py <iid|noniid> [seed] [condition_profile] "
            "[size_jitter_ratio] [proportional|balanced]\n"
            "Examples:\n"
            "  python generate_jnu_cwru_mix.py noniid 42 severe 0.25 balanced\n"
            "  python generate_jnu_cwru_mix.py noniid 42 severe 0.50 balanced\n"
        )

    mode = sys.argv[1]
    if mode not in {"iid", "noniid"}:
        raise SystemExit("The first argument must be 'iid' or 'noniid'.")

    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    condition_profile = sys.argv[3] if len(sys.argv) > 3 else "severe"
    size_jitter_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    test_split_mode = sys.argv[5] if len(sys.argv) > 5 else "balanced"

    if test_split_mode not in {"proportional", "balanced"}:
        raise SystemExit("The fifth argument must be 'proportional' or 'balanced'.")

    generate_dataset(seed=seed, niid=(mode == "noniid"))
