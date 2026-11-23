"""Q2.1 (extra): learning-rate sweep for the linear baseline, on the validation set.

The MLP of Q2.2 was tuned on validation, so for a fair comparison the baseline's learning rate
is tuned too. Everything else is the Q2.1 setup (plain SGD, batch size 64, 50 epochs, best val
epoch restored); each learning rate is trained with 3 seeds and selected by mean val accuracy.

    python -m tasks.q2_1_lr_sweep

Writes results/linear_lr/<lr>/ and results/q2_1_lr_sweep.json. The lr of tasks/q2_1.py
(0.001) is read from results/linear/ instead of being retrained.
"""

import json
import os

from common import RESULTS_DIR, SEEDS, fmt, mean_std
from tasks.q2_1 import CONFIG, NAME
from tasks.q2_2 import cached_run

LRS = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]


def main():
    rows = {}
    for lr in LRS:
        config = {**CONFIG, "lr": lr}
        name = NAME if lr == CONFIG["lr"] else f"linear_lr/{lr:g}"
        runs = [cached_run(name, config, seed) for seed in SEEDS]
        s = {key: mean_std([r[key] for r in runs]) for key in ("train_acc", "val_acc", "test_acc")}
        s["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
        s["final_train_loss"] = mean_std([r["history"]["train_loss"][-1] for r in runs])
        rows[f"{lr:g}"] = s
        print(f"lr {lr:<7g} train {fmt(s['train_acc'])} | val {fmt(s['val_acc'])} | "
              f"test {fmt(s['test_acc'])} | best epochs {s['best_epochs']} | "
              f"final train loss {s['final_train_loss']['mean']:.3f}", flush=True)

    best = max(rows, key=lambda lr: rows[lr]["val_acc"]["mean"])
    print(f"\nselected lr (mean val acc): {best}")
    with open(os.path.join(RESULTS_DIR, "q2_1_lr_sweep.json"), "w") as f:
        json.dump({"base_config": CONFIG, "lrs": LRS, "selected_lr": float(best),
                   "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
