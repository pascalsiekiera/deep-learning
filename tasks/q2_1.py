"""Q2.1: data statistics and a linear baseline (multinomial logistic regression, plain SGD).

    python -m tasks.q2_1      # (from the project root) trains 3 seeds, prints the statistics and accuracies
                              # writes results/q2_1/data_stats.json, results/q2_1/linear/,
                              # checkpoints/q2_1/linear/, figures/q2_1_curves.pdf
    python -m tasks.q2_1 --stats-only   # only the data statistics (no training)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from common import RESULTS_DIR, ROOT, fmt, get_data, load_results, run_seeds
from data import channel_stats
from load_data import CLASSES, SPLITS

NAME = "q2_1/linear"
CONFIG = {
    "hidden": (),          # no hidden layer
    "optimizer": "sgd",
    "lr": 0.0003,          # selected on validation by tasks/q2_1_lr_sweep.py
    "batch_size": 64,
    "epochs": 50,
    "patience": None,      # show the full curve; the best validation epoch is restored anyway
}
FIGURES_DIR = os.path.join(ROOT, "figures")
STATS_PATH = os.path.join(RESULTS_DIR, "q2_1", "data_stats.json")


def data_statistics(path=STATS_PATH):
    """Print the split sizes, the class distribution of each split, and the per-channel
    normalization statistics (training set only), and save them as JSON to `path`."""
    data = get_data()
    mean, std = channel_stats(data["train"]["images"])
    stats = {
        "examples_per_split": {split: len(data[split]["labels"]) for split in SPLITS},
        "class_counts": {split: dict(zip(CLASSES, torch.bincount(data[split]["labels"],
                                                                 minlength=len(CLASSES)).tolist()))
                         for split in SPLITS},
        "normalization": {
            "computed_on": "train",
            "pixel_scale": "uint8 / 255, i.e. values in [0, 1]",
            "channels": ["R", "G", "B"],
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }
    print("examples per split:", stats["examples_per_split"])
    for split, counts in stats["class_counts"].items():
        print(f"class counts ({split}):", counts)
    print("training-set channel mean (R, G, B):", [round(v, 4) for v in stats["normalization"]["mean"]])
    print("training-set channel std  (R, G, B):", [round(v, 4) for v in stats["normalization"]["std"]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    print("saved", path)
    return stats


def plot_curves(runs, path):
    """Mean ± std over seeds of the training loss and the validation accuracy per epoch."""
    loss = torch.tensor([r["history"]["train_loss"] for r in runs])
    acc = torch.tensor([r["history"]["val_score"] for r in runs]) * 100
    epochs = torch.arange(1, loss.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, values, label in ((axes[0], loss, "training loss"), (axes[1], acc, "validation accuracy (%)")):
        mean, std = values.mean(0), values.std(0)
        ax.plot(epochs, mean)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true",
                        help="only compute and save the data statistics (no training)")
    args = parser.parse_args()
    data_statistics()
    if args.stats_only:
        raise SystemExit
    summary = run_seeds(NAME, CONFIG, save_model=True)
    print(f"linear baseline | parameters {summary['runs'][0]['num_parameters']} | "
          f"train {fmt(summary['train_acc'])} | val {fmt(summary['val_acc'])} | "
          f"test {fmt(summary['test_acc'])}")
    plot_curves(load_results(NAME), os.path.join(FIGURES_DIR, "q2_1_curves.pdf"))
