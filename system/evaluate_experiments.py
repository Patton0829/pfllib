import argparse
import csv
import os
from glob import glob

import h5py
import numpy as np


def load_metrics(result_path):
    with h5py.File(result_path, "r") as f:
        data = {}
        for key in ("rs_test_acc", "rs_test_auc", "rs_train_loss"):
            if key in f:
                data[key] = np.asarray(f[key][:], dtype=float)
    return data


def summarize_result(result_path):
    data = load_metrics(result_path)
    row = {
        "file": os.path.basename(result_path),
        "algorithm": "",
        "goal": "",
        "final_test_acc": "",
        "best_test_acc": "",
        "last10_avg_test_acc": "",
        "final_test_auc": "",
        "best_test_auc": "",
        "final_train_loss": "",
        "best_train_loss": "",
    }

    stem = os.path.splitext(os.path.basename(result_path))[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        row["algorithm"] = parts[1]
        row["goal"] = "_".join(parts[2:-1]) if parts[-1].isdigit() else "_".join(parts[2:])

    if "rs_test_acc" in data and data["rs_test_acc"].size > 0:
        arr = data["rs_test_acc"]
        row["final_test_acc"] = float(arr[-1])
        row["best_test_acc"] = float(np.max(arr))
        row["last10_avg_test_acc"] = float(np.mean(arr[-10:]))

    if "rs_test_auc" in data and data["rs_test_auc"].size > 0:
        arr = data["rs_test_auc"]
        row["final_test_auc"] = float(arr[-1])
        row["best_test_auc"] = float(np.max(arr))

    if "rs_train_loss" in data and data["rs_train_loss"].size > 0:
        arr = data["rs_train_loss"]
        row["final_train_loss"] = float(arr[-1])
        row["best_train_loss"] = float(np.min(arr))

    return row


def filter_result_files(result_dir, dataset, algorithms, keywords, match_all_keywords=True):
    pattern = os.path.join(result_dir, f"{dataset}_*.h5")
    files = sorted(glob(pattern))

    filtered = []
    for path in files:
        name = os.path.basename(path)
        if algorithms and not any(f"_{algo}_" in name for algo in algorithms):
            continue
        if keywords:
            if match_all_keywords and not all(keyword in name for keyword in keywords):
                continue
            if not match_all_keywords and not any(keyword in name for keyword in keywords):
                continue
        filtered.append(path)
    return filtered


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows):
    if not rows:
        print("No matching result files found.")
        return

    columns = [
        "algorithm",
        "goal",
        "best_test_acc",
        "final_test_acc",
        "last10_avg_test_acc",
        "best_test_auc",
        "final_train_loss",
        "file",
    ]

    widths = {}
    for col in columns:
        widths[col] = max(len(col), *(len(format_value(row.get(col, ""))) for row in rows))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(format_value(row.get(col, "")).ljust(widths[col]) for col in columns))


def save_csv(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwru", help="Dataset prefix in result filenames")
    parser.add_argument(
        "--result-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Directory containing .h5 result files",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="*",
        default=["FedAvg", "FedAvgSim", "FedAvgAcc", "FedAvgSimAcc"],
        help="Algorithms to include",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        default=[],
        help="Only keep files whose names contain all given keywords",
    )
    parser.add_argument(
        "--match-any-keywords",
        action="store_true",
        help="Keep files whose names contain any given keyword instead of all keywords",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="best_test_acc",
        choices=[
            "best_test_acc",
            "final_test_acc",
            "last10_avg_test_acc",
            "best_test_auc",
            "final_train_loss",
            "file",
        ],
        help="Metric used for sorting",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to save the summary as CSV",
    )
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    files = filter_result_files(
        result_dir,
        args.dataset,
        args.algorithms,
        args.keywords,
        match_all_keywords=not args.match_any_keywords,
    )
    rows = [summarize_result(path) for path in files]

    def sort_key(row):
        value = row.get(args.sort_by, "")
        if value == "":
            return float("inf") if args.ascending else float("-inf")
        return value

    rows.sort(key=sort_key, reverse=not args.ascending)

    print(f"Matched {len(rows)} result files in: {result_dir}")
    print_table(rows)

    if args.csv:
        output_path = os.path.abspath(args.csv)
        save_csv(rows, output_path)
        print(f"\nCSV saved to: {output_path}")


if __name__ == "__main__":
    main()
