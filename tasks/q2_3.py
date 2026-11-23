"""Q2.3: a one-factor-at-a-time study around the selected MLP of Q2.2.

Protocol
  * Base = the configuration selected in Q2.2 (results/q2_2_summary.json).
  * Each factor is changed alone, everything else stays fixed.
  * Same budget for every run: 50 epochs, no early stopping (patience=None). As everywhere,
    the epoch with the best validation accuracy is restored at the end, so "val" below is the
    best val accuracy within the 50-epoch budget; "final val" is the accuracy after epoch 50.
  * 3 seeds per setting. A difference to the base is flagged as meaningful if it is larger than
    twice its standard error, sqrt(s_a^2 / 3 + s_b^2 / 3) (a rough rule with n = 3).
  * Only train and validation are used for the study; test is read only for the best model.
  * For the best setting (by mean val accuracy): per-class test accuracy and the confusion
    matrix, summed over the 3 seeds (3000 test predictions).

    python -m tasks.q2_3      # about 1.5 hours on a laptop CPU; finished runs are cached

Writes results/q23/<setting>/, checkpoints/q23/<best setting>/, results/q2_3_summary.json,
figures/q2_3_curves.pdf and figures/q2_3_confusion.pdf.
"""

import json
import math
import os

import torch

from common import (RESULTS_DIR, ROOT, SEEDS, get_loaders, load_model, mean_std, predict,
                    full_config)
from load_data import CLASSES
from metrics import confusion_matrix, per_class_accuracy
from tasks.q2_2 import cached_run

BUDGET = {"epochs": 50, "patience": None}

# factor -> (config key, values). The base value is included in each list and shared.
FACTORS = {
    "depth": ("hidden", [(1024,), (1024, 1024), (1024, 1024, 1024)]),
    "width": ("hidden", [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]),
    "dropout": ("dropout", [0.0, 0.2, 0.5]),
    "weight_decay": ("weight_decay", [0.0, 1e-4, 1e-3, 1e-2]),
    "learning_rate": ("lr", [0.004, 0.013, 0.04]),
    "normalization": ("normalize", [False, True]),
    "augmentation": ("flip", [False, True]),
}


def base_config():
    with open(os.path.join(RESULTS_DIR, "q2_2_summary.json")) as f:
        config = json.load(f)["selected_config"]
    config["hidden"] = tuple(config["hidden"])
    return {**config, **BUDGET}


def value_name(key, value):
    if key == "hidden":
        return "x".join(map(str, value))
    return f"{value:g}" if isinstance(value, float) else str(value)


def setting_name(base, key, value):
    return "q23/base" if value == base[key] else f"q23/{key}_{value_name(key, value)}"


def summarize(runs):
    s = {k: mean_std([r[k] for r in runs]) for k in ("train_acc", "val_acc", "test_acc")}
    s["final_val"] = mean_std([r["history"]["val_score"][-1] for r in runs])
    s["final_train_loss"] = mean_std([r["history"]["train_loss"][-1] for r in runs])
    s["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
    s["num_parameters"] = runs[0]["num_parameters"]
    return s


def diff_vs(a, b):
    """Difference of mean val accuracies a - b (in points) and its standard error."""
    d = 100 * (a["mean"] - b["mean"])
    se = 100 * math.sqrt(a["std"] ** 2 / len(a["values"]) + b["std"] ** 2 / len(b["values"]))
    return d, se


def pct(stats):
    return f"{100 * stats['mean']:.2f} ± {100 * stats['std']:.2f}"


def plot_curves(all_runs, base, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(FACTORS), figsize=(3.1 * len(FACTORS), 5.6), sharex=True)
    for col, (factor, (key, values)) in enumerate(FACTORS.items()):
        for value in values:
            runs = all_runs[setting_name(base, key, value)]
            loss = torch.tensor([r["history"]["train_loss"] for r in runs])
            val = 100 * torch.tensor([r["history"]["val_score"] for r in runs])
            epochs = torch.arange(1, loss.shape[1] + 1)
            label = value_name(key, value) + (" (base)" if value == base[key] else "")
            for ax, y in ((axes[0, col], loss), (axes[1, col], val)):
                line, = ax.plot(epochs, y.mean(0), label=label, lw=1.2)
                ax.fill_between(epochs, y.mean(0) - y.std(0), y.mean(0) + y.std(0),
                                color=line.get_color(), alpha=0.15)
        axes[0, col].set_title(factor)
        axes[0, col].set_yscale("log")
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_xlabel("epoch")
        for ax in axes[:, col]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("training loss (log)")
    axes[1, 0].set_ylabel("validation accuracy (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def plot_confusion(matrix, path):
    import matplotlib.pyplot as plt

    rates = matrix.float() / matrix.sum(1, keepdim=True)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(100 * rates, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{100 * rates[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax, label="% of the true class (test, 3 seeds)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def main():
    base = base_config()
    all_runs, settings = {}, {}
    for factor, (key, values) in FACTORS.items():
        for value in values:
            name = setting_name(base, key, value)
            if name not in all_runs:
                config = {**base, key: value}
                runs = []
                for seed in SEEDS:
                    r = cached_run(name, config, seed)
                    print(f"[{name} seed {seed}] val {100 * r['val_acc']:.2f} | train "
                          f"{100 * r['train_acc']:.2f} | best epoch {r['history']['best_epoch']}",
                          flush=True)
                    runs.append(r)
                all_runs[name] = runs
                settings[name] = {"config": full_config(config), **summarize(runs)}

    b = settings["q23/base"]
    tables = {}
    for factor, (key, values) in FACTORS.items():
        print(f"\n### {factor}\n")
        print("| value | params | train | val | Δval vs base | final val | best epochs | final train loss |")
        print("|---|---|---|---|---|---|---|---|")
        rows = []
        for value in values:
            name = setting_name(base, key, value)
            s = settings[name]
            d, se = diff_vs(s["val_acc"], b["val_acc"])
            meaningful = name != "q23/base" and abs(d) > 2 * se
            rows.append({"value": value_name(key, value), "name": name, "delta_val": d,
                         "delta_se": se, "meaningful": meaningful})
            delta = "–" if name == "q23/base" else f"{d:+.2f} ± {se:.2f}{' *' if meaningful else ''}"
            print(f"| {value_name(key, value)}{' (base)' if name == 'q23/base' else ''} | "
                  f"{s['num_parameters']:,} | {pct(s['train_acc'])} | {pct(s['val_acc'])} | {delta} | "
                  f"{pct(s['final_val'])} | {', '.join(map(str, s['best_epochs']))} | "
                  f"{s['final_train_loss']['mean']:.3f} |")
        tables[factor] = rows

    best = max(settings, key=lambda n: settings[n]["val_acc"]["mean"])
    print(f"\nbest setting by mean val accuracy: {best} (val {pct(settings[best]['val_acc'])})")
    # Retrains with save_model=True if the checkpoints are missing (cached_run checks them).
    for seed in SEEDS:
        cached_run(best, settings[best]["config"], seed, save_model=True)

    loaders = get_loaders(full_config(settings[best]["config"]))
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    per_class = []
    for seed in SEEDS:
        preds, labels = predict(load_model(best, seed), loaders["test"])
        matrix += confusion_matrix(preds, labels)
        per_class.append(per_class_accuracy(preds, labels))
    per_class = torch.stack(per_class)   # (seeds, classes)
    print(f"\nper-class test accuracy of {best} (mean ± std over seeds):")
    for c, name in enumerate(CLASSES):
        print(f"  {name:11s} {100 * per_class[:, c].mean():5.1f} ± {100 * per_class[:, c].std():4.1f}")
    off = matrix.clone()
    off.fill_diagonal_(0)
    pairs = sorted(((off[i, j].item(), CLASSES[i], CLASSES[j]) for i in range(len(CLASSES))
                    for j in range(len(CLASSES)) if i != j), reverse=True)[:10]
    print(f"most frequent confusions (true -> predicted, count out of {100 * len(SEEDS)}):")
    for count, t, p in pairs:
        print(f"  {t:11s} -> {p:11s} {count}")
    print("confusion matrix (rows = true, summed over seeds):")
    print(matrix)

    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    plot_curves(all_runs, base, os.path.join(ROOT, "figures", "q2_3_curves.pdf"))
    plot_confusion(matrix, os.path.join(ROOT, "figures", "q2_3_confusion.pdf"))
    with open(os.path.join(RESULTS_DIR, "q2_3_summary.json"), "w") as f:
        json.dump({"base_config": base, "factors": {k: [k2, [value_name(k2, v) for v in vs]]
                                                    for k, (k2, vs) in FACTORS.items()},
                   "settings": settings, "tables": tables, "best": best,
                   "test_acc_best": settings[best]["test_acc"],
                   "per_class_test_acc": {c: mean_std(per_class[:, i].tolist())
                                          for i, c in enumerate(CLASSES)},
                   "confusion_matrix": matrix.tolist(),
                   "top_confusions": [{"true": t, "pred": p, "count": n} for n, t, p in pairs]},
                  f, indent=1)


if __name__ == "__main__":
    main()
