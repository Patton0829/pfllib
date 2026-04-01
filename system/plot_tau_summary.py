import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    numeric_fields = [
        "final_test_acc",
        "best_test_acc",
        "last10_avg_test_acc",
        "final_train_loss",
        "best_train_loss",
    ]
    for row in rows:
        for field in numeric_fields:
            value = row.get(field, "")
            row[field] = float(value) if value not in ("", None) else np.nan
    return rows


def build_labels(rows):
    labels = []
    for row in rows:
        algo = row["algorithm"]
        goal = row["goal"]
        if algo == "FedAvg":
            labels.append("FedAvg\nbase")
        elif algo == "FedAvgSim":
            labels.append(goal.replace("sim_stau_", "Sim\ns="))
        elif algo == "FedAvgAcc":
            labels.append(goal.replace("acc_atau_", "Acc\na="))
        elif algo == "FedAvgSimAcc":
            label = goal.replace("simacc_stau_", "SimAcc\ns=")
            label = label.replace("_atau_", ", a=")
            labels.append(label)
        else:
            labels.append(goal)
    return labels


def color_for_algorithm(algo):
    palette = {
        "FedAvg": "#4C6A92",
        "FedAvgSim": "#2A9D8F",
        "FedAvgAcc": "#E9C46A",
        "FedAvgSimAcc": "#E76F51",
    }
    return palette.get(algo, "#7A7A7A")


def plot_summary(rows, save_path):
    rows = sorted(rows, key=lambda x: x["best_test_acc"], reverse=True)
    labels = build_labels(rows)
    colors = [color_for_algorithm(row["algorithm"]) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    metrics = [
        ("best_test_acc", "Best Test Accuracy"),
        ("final_test_acc", "Final Test Accuracy"),
        ("last10_avg_test_acc", "Last-10 Avg Test Accuracy"),
        ("final_train_loss", "Final Train Loss"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        values = [row[metric] for row in rows]
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if "acc" in metric:
            ax.set_ylim(max(0.85, min(values) - 0.01), max(values) + 0.01)
        if "loss" in metric:
            ax.set_ylim(0, max(values) * 1.15)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    legend_handles = []
    seen = set()
    for row in rows:
        algo = row["algorithm"]
        if algo in seen:
            continue
        seen.add(algo)
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_for_algorithm(algo)))
    fig.legend(legend_handles, list(seen), loc="upper center", ncol=len(seen), frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "tau_experiment_summary.csv"),
        help="Path to the summary CSV file",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "tau_experiment_summary.png"),
        help="Path to save the comparison figure",
    )
    args = parser.parse_args()

    rows = load_rows(os.path.abspath(args.csv))
    if not rows:
        raise ValueError("No rows found in CSV.")

    plot_summary(rows, os.path.abspath(args.save))
    print(f"Figure saved to: {os.path.abspath(args.save)}")


if __name__ == "__main__":
    main()
