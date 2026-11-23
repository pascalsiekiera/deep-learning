"""Q2.2: tune an MLP on the validation set, then report the selected configuration over 3 seeds.

Protocol (all decisions use validation accuracy; the test set is only read for the final report):
  1. Random search: sample --trials configurations from SEARCH_SPACE (sampling seed 0) and train
     each once (training seed 0), with early stopping.
  2. Re-evaluation: retrain the --top best configurations with 3 seeds each and select the one
     with the highest MEAN validation accuracy (one seed is too noisy to pick a winner: with
     1000 validation images, one standard error of the accuracy is about 1.5 points).
  3. Report: learning curves and train/val/test accuracy (mean ± std over 3 seeds) of the
     selected configuration, next to the linear baseline of Q2.1.

    python -m tasks.q2_2                  # everything (about 1-2 hours on a laptop CPU)
    python -m tasks.q2_2 --trials 40      # a larger search

Finished runs are skipped when the script is restarted, so an interrupted search resumes.
Writes results/q2_2/{search,top,mlp_best}/, checkpoints/q2_2/mlp_best/,
results/q2_2/summary.json and figures/q2_2_curves.pdf.
"""

import argparse
import json
import math
import os

import torch

from common import (RESULTS_DIR, ROOT, SEEDS, fmt, full_config, load_results, mean_std,
                    plot_learning_curves, run)

# Each entry is a list of values sampled uniformly, or ("log", low, high) for a log-uniform float.
# The learning-rate range depends on the optimizer (Adam needs much smaller steps than SGD).
SEARCH_SPACE = {
    "depth": [1, 2, 3],
    "width": [256, 512, 1024],
    "activation": ["relu", "gelu", "tanh"],
    "dropout": [0.0, 0.2, 0.5],
    "optimizer": ["adam", "sgd"],
    "lr": {"adam": ("log", 1e-4, 3e-3), "sgd": ("log", 3e-3, 1e-1)},
    "weight_decay": [0.0, 1e-4, 1e-3],
    "batch_size": [64, 128],
    "scheduler": ["none", "cosine"],
}
FIXED = {
    "epochs": 50,     # maximum number of epochs
    "patience": 8,    # early stopping on validation accuracy
    "momentum": 0.9,  # used by SGD only
}
SAMPLING_SEED = 0
BEST = "q2_2/mlp_best"   # run name of the selected MLP ("the best MLP" in Q2.3-Q2.5)
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
    config = {
        "hidden": (choice(s["width"]),) * choice(s["depth"]),
        "activation": choice(s["activation"]),
        "dropout": choice(s["dropout"]),
        "optimizer": optimizer,
        "lr": float(f"{log_uniform(low, high):.2g}"),
        "weight_decay": choice(s["weight_decay"]),
        "batch_size": choice(s["batch_size"]),
        "scheduler": choice(s["scheduler"]),
        **FIXED,
    }
    if optimizer == "adam":
        config["momentum"] = 0.0   # unused by Adam; keep the saved config honest
    return config


def describe(config):
    c = full_config(config)
    return (f"hidden={list(c['hidden'])} {c['activation']} dropout={c['dropout']} | "
            f"{c['optimizer']} lr={c['lr']:g} wd={c['weight_decay']:g} bs={c['batch_size']} "
            f"sched={c['scheduler']}")


def cached_run(name, config, seed, save_model=False):
    """`run`, unless results/<name>/seed<seed>.json already exists with the same config."""
    path = os.path.join(RESULTS_DIR, name, f"seed{seed}.json")
    checkpoint = os.path.join(ROOT, "checkpoints", name, f"seed{seed}.pt")
    if os.path.exists(path) and (not save_model or os.path.exists(checkpoint)):
        with open(path) as f:
            result = json.load(f)
        # JSON turns tuples into lists, so compare after the same round trip.
        if result["config"] == json.loads(json.dumps(full_config(config))):
            return result
    result, _ = run(name, config, seed, save_model=save_model)
    return result


def line(result):
    return (f"val {100 * result['val_acc']:5.2f} | train {100 * result['train_acc']:5.2f} | "
            f"best epoch {result['history']['best_epoch']:2d} | "
            f"{result['num_parameters'] / 1e6:5.2f}M params")


def random_search(num_trials):
    generator = torch.Generator().manual_seed(SAMPLING_SEED)
    configs = [sample_config(generator) for _ in range(num_trials)]
    results = []
    for i, config in enumerate(configs):
        result = cached_run(f"q2_2/search/trial{i:03d}", config, SEARCH_SEED)
        print(f"[search {i + 1:3d}/{num_trials}] {line(result)} | {describe(config)}", flush=True)
        results.append(result)
    return results


def reevaluate(candidates):
    """Retrain each candidate with all seeds; return [(stats, config)] sorted by mean val acc."""
    scored = []
    for rank, result in enumerate(candidates):
        config = result["config"]
        runs = [cached_run(f"q2_2/top/rank{rank}", config, seed) for seed in SEEDS]
        stats = {key: mean_std([r[key] for r in runs]) for key in ("train_acc", "val_acc")}
        print(f"[top {rank + 1}] val {fmt(stats['val_acc'])} | train {fmt(stats['train_acc'])} "
              f"| {describe(config)}", flush=True)
        scored.append((stats, config))
    return sorted(scored, key=lambda sc: -sc[0]["val_acc"]["mean"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30, help="random-search trials")
    parser.add_argument("--top", type=int, default=3, help="configurations re-run with 3 seeds")
    args = parser.parse_args()

    print(f"== 1. random search: {args.trials} trials, seed {SEARCH_SEED} ==")
    search = random_search(args.trials)
    ranked = sorted(search, key=lambda r: (-r["val_acc"], r["num_parameters"]))
    print("\nbest trials (by validation accuracy, single seed):")
    for r in ranked[:10]:
        print(f"  {r['name']}: {line(r)} | {describe(r['config'])}")

    print(f"\n== 2. re-evaluate the top {args.top} with seeds {list(SEEDS)} ==")
    best_stats, best_config = reevaluate(ranked[:args.top])[0]
    print(f"selected: {describe(best_config)} (val {fmt(best_stats['val_acc'])})")

    print("\n== 3. final runs of the selected configuration ==")
    runs = [cached_run(BEST, best_config, seed, save_model=True) for seed in SEEDS]
    for r in runs:
        print(f"  seed {r['seed']}: {line(r)} | test {100 * r['test_acc']:.2f}")
    summary = {key: mean_std([r[key] for r in runs]) for key in ("train_acc", "val_acc", "test_acc")}

    print("\nmodel             params     train            val              test")
    rows = [("mlp_best", runs, summary)]
    try:
        linear = load_results("q2_1/linear")
        rows.insert(0, ("linear (Q2.1)", linear,
                        {k: mean_std([r[k] for r in linear]) for k in ("train_acc", "val_acc", "test_acc")}))
    except FileNotFoundError:
        print("(run `python -m tasks.q2_1` first to compare with the linear baseline)")
    for name, rs, s in rows:
        print(f"{name:16s} {rs[0]['num_parameters']:>9,}  {fmt(s['train_acc'])}  "
              f"{fmt(s['val_acc'])}  {fmt(s['test_acc'])}")

    with open(os.path.join(RESULTS_DIR, "q2_2", "summary.json"), "w") as f:
        json.dump({"search_space": SEARCH_SPACE, "fixed": FIXED, "trials": args.trials,
                   "top": args.top, "selected_config": best_config,
                   "search_ranking": [{"name": r["name"], "val_acc": r["val_acc"],
                                       "config": r["config"]} for r in ranked],
                   "final": summary}, f, indent=1)
    plot_learning_curves(runs, os.path.join(ROOT, "figures", "q2_2_curves.pdf"),
                         title="Selected MLP: " + describe(best_config))


if __name__ == "__main__":
    main()
