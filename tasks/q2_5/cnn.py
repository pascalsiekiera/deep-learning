"""Q2.5: a small CNN, trained and selected with the same protocol as the MLP of Q2.2, then the
shift and pixel-permutation experiments of Q2.4 repeated with it.

Protocol (as in Q2.2; all decisions on validation, test only for the final report):
  1. Random search: --trials configurations from SEARCH_SPACE (sampling seed 0), each trained
     once with seed 0; at most 50 epochs, early stopping on val accuracy with patience 8.
  2. Re-evaluation: the --top best re-run with seeds 0, 1, 2; select by MEAN val accuracy.
  3. Report: the selected CNN with 3 seeds (learning curves, train/val/test), next to the linear
     baseline (Q2.1) and the best MLP (Q2.2).
  4. Q2.4 experiments with the CNN: shifted test images (+ the border-only control), and a
     retrain on pixels permuted with the same fixed permutation (seed 0).

No data augmentation, as for the MLP (flips would also mix augmentation into the shift test).
The CNN is trained on the Apple GPU (MPS) if there is no CUDA GPU, see common.run_device.

    python -m tasks.q2_5.cnn      # about 1 hour with MPS; finished runs are cached

Writes results/q2_5/{search,top,cnn_best,cnn_perm}/, checkpoints/q2_5/{cnn_best,cnn_perm}/,
results/q2_5/summary.json and figures/q2_5/*.pdf.
"""

import argparse
import json
import math
import os
import shutil

import torch

from common import (CHECKPOINT_DIR, RESULTS_DIR, class_confusion, plot_confusion, ROOT, SEEDS, fmt, full_config, load_results, mean_std,
                    plot_learning_curves)
from tasks.q2_1.baseline import NAME as LINEAR
from tasks.q2_2.mlp_search import BEST as MLP_BEST
from tasks.q2_2.mlp_search import cached_run, line
from tasks.q2_4.inductive_biases import PERM as MLP_PERM
from tasks.q2_4.inductive_biases import SHIFTS, permutation_experiment, plot_shifts, shift_accuracies

OUT = os.path.join(RESULTS_DIR, "q2_5")
FIG = os.path.join(ROOT, "figures", "q2_5")
CNN_BEST = "q2_5/cnn_best"
CNN_PERM = "q2_5/cnn_perm"
CNN_FIXED = "q2_5/cnn_best_50ep"   # selected CNN, fixed 50-epoch budget, no early stopping

# Architectures as (channels per block, convolutions per block): 0.39M, 1.18M and 1.56M
# parameters. A 4.7M-parameter variant ((64, ..., 512) x 2) was dropped: on the 8 GB laptop it
# needed ~6 GB of GPU memory and ~30 s per epoch, and the task asks for a *small* CNN.
SEARCH_SPACE = {
    "architecture": [((32, 64, 128, 256), 1), ((32, 64, 128, 256), 2), ((64, 128, 256, 512), 1)],
    "dropout": [0.0, 0.3, 0.5],
    "optimizer": ["adam", "sgd"],
    "lr": {"adam": ("log", 3e-4, 3e-3), "sgd": ("log", 3e-3, 1e-1)},
    "weight_decay": [0.0, 1e-4, 5e-4],
    "batch_size": [64, 128],
    "scheduler": ["none", "cosine"],
}
FIXED = {"model": "cnn", "batchnorm": True, "activation": "relu", "epochs": 50, "patience": 8,
         "momentum": 0.9}
SAMPLING_SEED = 0
SEARCH_SEED = 0


def sample_config(generator):
    def choice(values):
        return values[torch.randint(len(values), (1,), generator=generator).item()]

    def log_uniform(low, high):
        u = torch.rand((), generator=generator).item()
        return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))

    s = SEARCH_SPACE
    optimizer = choice(s["optimizer"])
    _, low, high = s["lr"][optimizer]
    channels, convs_per_block = choice(s["architecture"])
    config = {
        "channels": channels,
        "convs_per_block": convs_per_block,
        "dropout": choice(s["dropout"]),
        "optimizer": optimizer,
        "lr": float(f"{log_uniform(low, high):.2g}"),
        "weight_decay": choice(s["weight_decay"]),
        "batch_size": choice(s["batch_size"]),
        "scheduler": choice(s["scheduler"]),
        **FIXED,
    }
    if optimizer == "adam":
        config["momentum"] = 0.0
    return config


def describe(config):
    c = full_config(config)
    return (f"channels={list(c['channels'])} x{c['convs_per_block']} dropout={c['dropout']} | "
            f"{c['optimizer']} lr={c['lr']:g} wd={c['weight_decay']:g} bs={c['batch_size']} "
            f"sched={c['scheduler']}")


def reuse_run(src, dst, config, seed):
    """Copy the run `src` (results and checkpoint) to `dst` if `dst` is missing. Runs are
    deterministic, so retraining the same config with the same seed would give the same result;
    this saves the time. Returns the result of `dst` (trained if `src` does not exist either)."""
    s_json = os.path.join(RESULTS_DIR, src, f"seed{seed}.json")
    s_ckpt = os.path.join(CHECKPOINT_DIR, src, f"seed{seed}.pt")
    d_json = os.path.join(RESULTS_DIR, dst, f"seed{seed}.json")
    if not os.path.exists(d_json) and os.path.exists(s_json) and os.path.exists(s_ckpt):
        with open(s_json) as f:
            result = json.load(f)
        if result["config"] == json.loads(json.dumps(full_config(config))):
            result["name"], result["copied_from"] = dst, src
            os.makedirs(os.path.dirname(d_json), exist_ok=True)
            with open(d_json, "w") as f:
                json.dump(result, f, indent=1)
            os.makedirs(os.path.join(CHECKPOINT_DIR, dst), exist_ok=True)
            shutil.copyfile(s_ckpt, os.path.join(CHECKPOINT_DIR, dst, f"seed{seed}.pt"))
    return cached_run(dst, config, seed, save_model=True)


def stats(runs):
    s = {k: mean_std([r[k] for r in runs]) for k in ("train_acc", "val_acc", "test_acc")}
    s["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
    s["num_parameters"] = runs[0]["num_parameters"]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=16, help="random-search trials")
    parser.add_argument("--top", type=int, default=3, help="configurations re-run with 3 seeds")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    print(f"== 1. random search: {args.trials} trials, seed {SEARCH_SEED} ==")
    generator = torch.Generator().manual_seed(SAMPLING_SEED)
    configs = [sample_config(generator) for _ in range(args.trials)]
    search = []
    for i, config in enumerate(configs):
        r = cached_run(f"q2_5/search/trial{i:03d}", config, SEARCH_SEED, save_model=True)
        print(f"[search {i + 1:2d}/{args.trials}] {line(r)} | {describe(config)}", flush=True)
        search.append(r)
    ranked = sorted(search, key=lambda r: (-r["val_acc"], r["num_parameters"]))

    print(f"\n== 2. re-evaluate the top {args.top} with seeds {list(SEEDS)} ==")
    scored = []
    for rank, r in enumerate(ranked[:args.top]):
        # seed 0 is the search run itself (same config, same seed)
        runs = [reuse_run(r["name"], f"q2_5/top/rank{rank}", r["config"], seed) if seed == SEARCH_SEED
                else cached_run(f"q2_5/top/rank{rank}", r["config"], seed, save_model=True)
                for seed in SEEDS]
        s = stats(runs)
        print(f"[top {rank + 1}] val {fmt(s['val_acc'])} | train {fmt(s['train_acc'])} | "
              f"{describe(r['config'])}", flush=True)
        scored.append((s, r["config"], f"q2_5/top/rank{rank}"))
    best_stats, best_config, best_top = max(scored, key=lambda sc: sc[0]["val_acc"]["mean"])
    print(f"selected: {describe(best_config)} (val {fmt(best_stats['val_acc'])})")

    print("\n== 3. final runs of the selected CNN ==")
    runs = [reuse_run(best_top, CNN_BEST, best_config, seed) for seed in SEEDS]
    for r in runs:
        print(f"  seed {r['seed']}: {line(r)} | test {100 * r['test_acc']:.2f}")
    plot_learning_curves(runs, os.path.join(FIG, "curves.pdf"),
                         title="Selected CNN: " + describe(best_config))
    comparison = {"linear (Q2.1)": stats(load_results(LINEAR)),
                  "MLP (Q2.2)": stats(load_results(MLP_BEST)),
                  "CNN (Q2.5)": stats(runs)}
    print("\nmodel            params       train            val              test")
    for name, s in comparison.items():
        print(f"{name:14s} {s['num_parameters']:>11,}  {fmt(s['train_acc'])}  {fmt(s['val_acc'])}  "
              f"{fmt(s['test_acc'])}")

    print("\n== 4a. shifted test images ==")
    shifts = {"CNN": shift_accuracies(CNN_BEST), "MLP": shift_accuracies(MLP_BEST),
              "linear": shift_accuracies(LINEAR)}
    for label, res in shifts.items():
        for direction in ("right", "right_mask", "right_lost", "down", "down_mask", "down_lost"):
            print(f"{label:7s} {direction:10s} " + " | ".join(
                f"{k:2d}px {100 * res[direction][k]['mean']:5.2f}±{100 * res[direction][k]['std']:.2f}"
                for k in SHIFTS))
    plot_shifts(shifts, os.path.join(FIG, "shifts.pdf"))

    print("\n== 4b. fixed pixel permutation, CNN retrained from scratch ==")
    base, perm, _ = permutation_experiment(CNN_BEST, CNN_PERM)
    mlp_perm = stats(load_results(MLP_PERM))
    for label, s in (("CNN original", base), ("CNN permuted", perm),
                     ("MLP original", comparison["MLP (Q2.2)"]), ("MLP permuted", mlp_perm)):
        print(f"{label:13s} train {fmt(s['train_acc'])} | val {fmt(s['val_acc'])} | "
              f"test {fmt(s['test_acc'])} | best epochs {s['best_epochs']}")

    print("\n== 5. selected CNN with a fixed budget: 50 epochs, no early stopping ==")
    # Extra check, not part of the selection protocol: with patience 8, early stopping sometimes
    # fired in the noisy first half of the cosine schedule (see the notes). Same config and
    # seeds, only patience=None; the best val epoch is still restored.
    fixed_runs = [cached_run(CNN_FIXED, {**best_config, "patience": None}, seed, save_model=True)
                  for seed in SEEDS]
    fixed = stats(fixed_runs)
    for r in fixed_runs:
        print(f"  seed {r['seed']}: {line(r)} | test {100 * r['test_acc']:.2f}")
    for label, s_ in (("patience 8", comparison["CNN (Q2.5)"]), ("50 epochs, no early stop", fixed)):
        print(f"{label:25s} train {fmt(s_['train_acc'])} | val {fmt(s_['val_acc'])} | "
              f"test {fmt(s_['test_acc'])} | best epochs {s_['best_epochs']}")
    plot_learning_curves(fixed_runs, os.path.join(FIG, "curves_50ep.pdf"),
                         title="Selected CNN, 50 epochs without early stopping: " + describe(best_config))

    print("\n== confusion matrices of the best CNN (test, 3 seeds) ==")
    conf = class_confusion(CNN_BEST)
    plot_confusion(conf["matrix"], os.path.join(FIG, "confusion_cnn.pdf"),
                   title="Best CNN (q2_5/cnn_best, protocol with patience 8)")
    conf_50 = class_confusion(CNN_FIXED)
    plot_confusion(conf_50["matrix"], os.path.join(FIG, "confusion_cnn_50ep.pdf"),
                   title="Best CNN, 50 epochs without early stopping (q2_5/cnn_best_50ep)")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"search_space": SEARCH_SPACE, "fixed": FIXED, "trials": args.trials,
                   "top": args.top, "selected_config": best_config,
                   "search_ranking": [{"name": r["name"], "val_acc": r["val_acc"],
                                       "num_parameters": r["num_parameters"],
                                       "config": r["config"]} for r in ranked],
                   "top_reevaluation": [{"name": n, "config": c, **s} for s, c, n in scored],
                   "comparison": comparison, "shifts": shifts,
                   "permutation": {"cnn_original": base, "cnn_permuted": perm,
                                   "mlp_original": comparison["MLP (Q2.2)"],
                                   "mlp_permuted": mlp_perm},
                   "fixed_budget_50ep": fixed, "confusion_cnn": conf["summary"],
                   "confusion_cnn_50ep": conf_50["summary"]}, f, indent=1)


if __name__ == "__main__":
    main()
