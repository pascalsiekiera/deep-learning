"""Q2.2 (extra): activation-function ablation on the selected MLP.

The random search in q2_2.py sampled ReLU, GELU and tanh. To test ELU and PReLU without
re-running (and changing) the whole search, we keep the selected configuration fixed and swap
only the activation, with 3 seeds each. Selection would again be by mean validation accuracy;
test accuracy is printed for completeness.

PReLU is torch's default nn.PReLU(): one learnable negative slope per hidden layer, initialized
to 0.25. Note that weight decay also acts on that slope (pulling it toward 0, i.e. ReLU).

    python -m tasks.q2_2_activations      # run after tasks.q2_2

Writes results/q2_2/activations/<activation>/ and results/q2_2/activations.json. The ReLU row
is the selected model itself (results/q2_2/mlp_best/), so it is not retrained.
"""

import json
import os

from common import RESULTS_DIR, SEEDS, fmt, mean_std
from tasks.q2_2 import BEST, cached_run, line

ACTIVATIONS = ["relu", "gelu", "tanh", "elu", "prelu"]


def main():
    with open(os.path.join(RESULTS_DIR, "q2_2", "summary.json")) as f:
        base = json.load(f)["selected_config"]
    base["hidden"] = tuple(base["hidden"])

    rows = {}
    for act in ACTIVATIONS:
        config = {**base, "activation": act}
        name = BEST if act == base["activation"] else f"q2_2/activations/{act}"
        runs = []
        for seed in SEEDS:
            r = cached_run(name, config, seed, save_model=name == BEST)
            print(f"[{act:5s} seed {seed}] {line(r)} | test {100 * r['test_acc']:5.2f}", flush=True)
            runs.append(r)
        rows[act] = {key: mean_std([r[key] for r in runs])
                     for key in ("train_acc", "val_acc", "test_acc")}
        rows[act]["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
        rows[act]["num_parameters"] = runs[0]["num_parameters"]

    print("\nactivation  params       train            val              test          best epochs")
    for act, s in sorted(rows.items(), key=lambda kv: -kv[1]["val_acc"]["mean"]):
        print(f"{act:10s} {s['num_parameters']:>11,}  {fmt(s['train_acc'])}  {fmt(s['val_acc'])}  "
              f"{fmt(s['test_acc'])}  {s['best_epochs']}")

    with open(os.path.join(RESULTS_DIR, "q2_2", "activations.json"), "w") as f:
        json.dump({"base_config": base, "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
