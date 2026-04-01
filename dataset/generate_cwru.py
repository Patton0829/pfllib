import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_clients = 20
raw_dir_path = "cwru/"
dir_path = "cwru/"
train_ratio = 0.8
dirichlet_alpha = 0.05
size_jitter_ratio = 0.25
condition_profile = "balanced"


def load_split(split_path):
    data = np.load(split_path)
    return data["x"], data["y"]


def load_all_conditions(raw_dir_path):
    dataset_x = []
    dataset_y = []
    dataset_conditions = []

    condition_names = sorted(
        name for name in os.listdir(raw_dir_path)
        if os.path.isdir(os.path.join(raw_dir_path, name)) and name.startswith("condition")
    )

    for condition_id, condition_name in enumerate(condition_names):
        condition_path = os.path.join(raw_dir_path, condition_name)
        for split_name in ["cnn_train.npz", "cnn_valid.npz", "cnn_test.npz"]:
            x, y = load_split(os.path.join(condition_path, split_name))
            dataset_x.append(x)
            dataset_y.append(y)
            dataset_conditions.append(np.full(len(y), condition_id, dtype=np.int64))

    dataset_x = np.concatenate(dataset_x, axis=0)
    dataset_y = np.concatenate(dataset_y, axis=0)
    dataset_conditions = np.concatenate(dataset_conditions, axis=0)
    return dataset_x, dataset_y, dataset_conditions, condition_names


def plot_single_client_distribution(client_id, client_labels, output_prefix):
    unique_labels, counts = np.unique(client_labels, return_counts=True)
    full_counts = np.zeros(10, dtype=np.int64)
    for label, count in zip(unique_labels, counts):
        full_counts[int(label)] = int(count)

    label_names = [f"Class {i}" for i in range(len(full_counts))]
    x = np.arange(len(full_counts))
    colors = plt.cm.Set3(np.linspace(0, 1, len(full_counts)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bars = ax.bar(x, full_counts, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Label Distribution for Client {client_id}")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=0)
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
        os.path.join(fig_dir, f"cwru_client_{client_id}_label_distribution"),
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
        if num_conditions != 4 or num_clients != 20:
            raise ValueError("The 'moderate' condition profile currently expects 4 conditions and 20 clients.")
        return [7, 5, 4, 4]

    if profile == "severe":
        if num_conditions != 4 or num_clients != 20:
            raise ValueError("The 'severe' condition profile currently expects 4 conditions and 20 clients.")
        return [10, 5, 3, 2]

    raise ValueError(f"Unsupported condition profile: {profile}")


def summarize_clients(client_indices, client_labels, client_conditions, condition_names):
    statistic = []
    sizes = []
    for client_id in range(len(client_indices)):
        client_stat = []
        for label in np.unique(client_labels[client_id]):
            client_stat.append((int(label), int(np.sum(client_labels[client_id] == label))))
        statistic.append(client_stat)
        sizes.append(len(client_indices[client_id]))
        condition_id = int(np.unique(client_conditions[client_id])[0])
        print(
            f"Client {client_id}\t Size of data: {len(client_indices[client_id])}\t "
            f"Condition: {condition_names[condition_id]}\t Labels: ",
            np.unique(client_labels[client_id]),
        )
        print(f"\t\t Samples of labels: ", client_stat)
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


def allocate_clients_by_condition(dataset_y, dataset_conditions, condition_names, num_clients, num_classes, niid, profile):
    num_conditions = len(condition_names)
    clients_per_condition = get_clients_per_condition(num_clients, num_conditions, profile)
    all_client_indices = []
    all_client_labels = []
    all_client_conditions = []

    for condition_id in range(num_conditions):
        condition_indices = np.where(dataset_conditions == condition_id)[0]
        quotas = build_jittered_quotas(len(condition_indices), clients_per_condition[condition_id], size_jitter_ratio)

        if niid:
            local_client_indices = allocate_condition_label_skew(
                condition_indices, dataset_y, quotas, num_classes, dirichlet_alpha
            )
        else:
            local_client_indices = allocate_condition_random(condition_indices, quotas)

        for idxs in local_client_indices:
            all_client_indices.append(idxs)
            all_client_labels.append(dataset_y[idxs])
            all_client_conditions.append(dataset_conditions[idxs])

    return all_client_indices, all_client_labels, all_client_conditions


def split_data_custom(X, y, seed):
    train_data, test_data = [], []
    train_counts, test_counts = [], []

    for client_id in range(len(y)):
        _, counts = np.unique(y[client_id], return_counts=True)
        stratify_labels = y[client_id] if np.min(counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X[client_id],
            y[client_id],
            train_size=train_ratio,
            shuffle=True,
            random_state=seed + client_id,
            stratify=stratify_labels,
        )

        train_data.append({"x": X_train, "y": y_train})
        test_data.append({"x": X_test, "y": y_test})
        train_counts.append(len(y_train))
        test_counts.append(len(y_test))

    print("Total number of samples:", sum(train_counts) + sum(test_counts))
    print("The number of train samples:", train_counts)
    print("The number of test samples:", test_counts)
    print(f"Train/Test ratio: {train_ratio:.1%}/{1-train_ratio:.1%}")
    print()

    return train_data, test_data


def save_file_custom(config_path, train_path, test_path, train_data, test_data,
                     num_clients, num_classes, statistic, niid, balance, partition, seed):
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

    dataset_x, dataset_y, dataset_conditions, condition_names = load_all_conditions(raw_dir_path)
    num_classes = len(set(dataset_y))

    print(f"Number of samples: {len(dataset_y)}")
    print(f"Number of classes: {num_classes}")
    print(f"Random seed: {seed}")
    print("Client-condition mapping: one client belongs to exactly one condition")
    print(f"Condition profile: {condition_profile}")
    print(f"Size jitter ratio: {size_jitter_ratio}")

    client_indices, y, client_conditions = allocate_clients_by_condition(
        dataset_y, dataset_conditions, condition_names, num_clients, num_classes, niid, condition_profile
    )

    statistic = summarize_clients(client_indices, y, client_conditions, condition_names)

    X = [dataset_x[idxs] for idxs in client_indices]
    train_data, test_data = split_data_custom(X, y, seed)
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
    )
    save_distribution_figure(dir_path, y, client_id=0)
    print(f"Saved figures to {os.path.join(dir_path, 'figures')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python generate_cwru.py <iid|noniid> [balance|-] [pat|dir|exdir|-] [seed] [condition_profile] [size_jitter_ratio]\n"
            "Examples:\n"
            "  python generate_cwru.py iid - - 42\n"
            "  python generate_cwru.py noniid - - 42 moderate 0.20\n"
            "  python generate_cwru.py noniid - - 42 severe 0.25\n"
            "  python generate_cwru.py noniid balance dir 42 balanced 0.12"
        )

    mode = sys.argv[1]
    if mode not in {"iid", "noniid"}:
        raise SystemExit("The first argument must be 'iid' or 'noniid'.")

    balance_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    partition_arg = sys.argv[3] if len(sys.argv) > 3 else "-"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    condition_profile = sys.argv[5] if len(sys.argv) > 5 else "balanced"
    size_jitter_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else 0.12

    niid = mode == "noniid"
    balance = balance_arg == "balance"
    partition = partition_arg if partition_arg != "-" else None

    generate_dataset(dir_path, raw_dir_path, num_clients, niid, balance, partition, seed)
