import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_results(result_path):
    with h5py.File(result_path, "r") as f:
        keys = list(f.keys())
        data = {key: f[key][:] for key in keys}
    return data


def print_summary(data):
    print("Available keys:", ", ".join(data.keys()))

    if "rs_test_acc" in data and len(data["rs_test_acc"]) > 0:
        arr = data["rs_test_acc"]
        print(f"Final Test Accuracy: {arr[-1]:.4f}")
        print(f"Last 10 Avg Test Accuracy: {np.mean(arr[-10:]):.4f}")
        print(f"Best Test Accuracy: {np.max(arr):.4f}")

    if "rs_test_auc" in data and len(data["rs_test_auc"]) > 0:
        arr = data["rs_test_auc"]
        print(f"Final Test AUC: {arr[-1]:.4f}")
        print(f"Last 10 Avg Test AUC: {np.mean(arr[-10:]):.4f}")
        print(f"Best Test AUC: {np.max(arr):.4f}")

    if "rs_train_loss" in data and len(data["rs_train_loss"]) > 0:
        arr = data["rs_train_loss"]
        print(f"Final Train Loss: {arr[-1]:.4f}")
        print(f"Last 10 Avg Train Loss: {np.mean(arr[-10:]):.4f}")
        print(f"Best Train Loss: {np.min(arr):.4f}")


def plot_results(data, save_path=None):
    candidate_items = [
        ("rs_test_acc", "Test Accuracy"),
        ("rs_test_auc", "Test AUC"),
        ("rs_train_loss", "Train Loss"),
    ]

    valid_items = []
    for key, title in candidate_items:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size == 0:
            continue
        if not np.any(np.isfinite(arr)):
            continue
        valid_items.append((key, title, arr))

    if not valid_items:
        print("No valid data available for plotting.")
        return

    fig, axes = plt.subplots(1, len(valid_items), figsize=(5 * len(valid_items), 4.5), constrained_layout=True)
    if len(valid_items) == 1:
        axes = [axes]

    for ax, (key, title, arr) in zip(axes, valid_items):
        ax.plot(arr, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.35)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=r"E:\fl\PFLlib\results\cwru_FedAvg_baseline_fedavg_0.h5",
        help="Path to the .h5 result file",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="",
        help="Optional path to save the plotted figure",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Result file not found: {args.path}")

    data = load_results(args.path)
    print_summary(data)
    plot_results(data, save_path=args.save or None)


if __name__ == "__main__":
    main()
