"""Q2.4: inductive biases of MLPs.

  (a) First-layer weights of hidden units of the best MLP (mlp_best, seed 0), and the class
      templates of the linear baseline (linear, seed 0), shown as 64x64 RGB images, next to the
      mean training image of each class.
  (b) Shift test: the trained models (3 seeds) evaluated on test images shifted by k pixels
      (right and down, separately), the uncovered border filled with the training mean colour.
      No retraining; the linear baseline is shown for comparison.
  (c) Pixel permutation: one fixed random permutation of the pixel positions (seed 0) applied to
      all splits, and the mlp_best configuration retrained from scratch with the same protocol
      (3 seeds, early stopping on val, at most 50 epochs).

    python -m tasks.q2_4.inductive_biases

Writes results/q2_4/mlp_perm/, checkpoints/q2_4/mlp_perm/, results/q2_4/summary.json,
figures/q2_4/*.pdf. `shift_accuracies` and `permutation_experiment` are reused for the CNN in Q2.5.
"""

import json
import os

import torch

from common import (RESULTS_DIR, class_confusion, plot_confusion, ROOT, SEEDS, accuracy_on, build_model, fmt, full_config, get_data,
                    get_loaders, load_model, load_results, mean_std, mask_border, mask_lost,
                    pixel_permutation, run_device, shift_images)
from train import set_seed
from data import channel_stats
from load_data import CLASSES
from tasks.q2_1.baseline import NAME as LINEAR
from tasks.q2_2.mlp_search import BEST, cached_run

FIG = os.path.join(ROOT, "figures", "q2_4")
SHIFTS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
PERMUTATION_SEED = 0
PERM = "q2_4/mlp_perm"


# ----------------------------------------------------------------------------- (a) weights
def to_rgb(w):
    """A weight vector (3*64*64,) as an image (64, 64, 3) in [0, 1]: 0.5 = zero weight,
    symmetric scaling by the largest absolute value (sign is kept, magnitude is per image)."""
    w = w.reshape(3, 64, 64)
    return (0.5 + 0.5 * w / w.abs().max()).permute(1, 2, 0).clamp(0, 1)


def plot_weight_grid(weights, titles, path, cols, suptitle):
    import matplotlib.pyplot as plt

    rows = (len(weights) + cols - 1) // cols
    # constrained layout reserves room for every subplot title, so titles never overlap images
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows + 0.4), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, w, t in zip(axes.flat, weights, titles):
        ax.imshow(to_rgb(w))
        ax.set_title(t, fontsize=8)
    fig.suptitle(suptitle, fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def first_layer(name, seed):
    """(trained, initial) first-layer weights. The initialization is rebuilt exactly as in
    common.run (set_seed, then build_model), so W - W_init is what training changed."""
    config = full_config(load_results(name)[0]["config"])
    set_seed(seed)
    w_init = build_model(config).first_layer.weight.detach()
    return load_model(name, seed).first_layer.weight.detach(), w_init


def smoothness(w):
    """Mean correlation between horizontally neighbouring weights (0 for i.i.d. noise)."""
    w = w.reshape(-1, 3, 64, 64)
    a, b = w[..., :, :-1].reshape(len(w), -1), w[..., :, 1:].reshape(len(w), -1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean().item()


def channel_corr(w):
    """Mean correlation between the R, G and B weight maps of a unit (0 for i.i.d. noise,
    1 if the unit only responds to brightness, not colour)."""
    w = w.reshape(-1, 3, 64 * 64)
    w = w - w.mean(2, keepdim=True)
    w = w / w.norm(dim=2, keepdim=True)
    c = torch.einsum("ncp,ndp->ncd", w, w)
    return ((c[:, 0, 1] + c[:, 0, 2] + c[:, 1, 2]) / 3).mean().item()


def class_mean_figure(template_delta):
    """Mean training image of each class (top) and its deviation from the overall mean image
    (bottom, contrast-stretched like the weights). Shows what a position-wise template can at
    best pick up: the average scene of a class. Returns the correlation between each class's
    learned linear template (W - W_init) and each class's deviation image, in normalized
    input space (rows: templates, columns: class means)."""
    import matplotlib.pyplot as plt

    data = get_data()["train"]
    x, y = data["images"].float() / 255.0, data["labels"]
    mean, std = channel_stats(data["images"])
    overall = x.mean(0)
    means = torch.stack([x[y == c].mean(0) for c in range(len(CLASSES))])   # (10, 3, 64, 64)
    deviation = (means - overall) / std.view(1, 3, 1, 1)                      # normalized space

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(1.6 * len(CLASSES), 4.2), layout="constrained")
    for c, name in enumerate(CLASSES):
        axes[0, c].imshow(means[c].permute(1, 2, 0).clamp(0, 1))
        axes[0, c].set_title(name, fontsize=8)
        axes[1, c].imshow(to_rgb(deviation[c].reshape(-1)))
        axes[1, c].set_title("− overall mean", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Mean training image per class (top) and its deviation from the mean image of "
                 "all classes (bottom, contrast-stretched)", fontsize=9)
    path = os.path.join(FIG, "class_means.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)

    a = template_delta - template_delta.mean(1, keepdim=True)
    b = deviation.reshape(len(CLASSES), -1)
    b = b - b.mean(1, keepdim=True)
    corr = (a / a.norm(dim=1, keepdim=True)) @ (b / b.norm(dim=1, keepdim=True)).T
    print("corr(template W - W_init of class i, mean-image deviation of class j), diagonal:",
          {c: round(corr[i, i].item(), 3) for i, c in enumerate(CLASSES)})
    off = corr[~torch.eye(len(CLASSES), dtype=torch.bool)]
    print(f"mean diagonal {corr.diag().mean():.3f} | mean off-diagonal {off.mean():.3f} | "
          f"own class has the highest correlation for "
          f"{(corr.argmax(1) == torch.arange(len(CLASSES))).sum().item()}/10 templates")
    return {"diagonal": {c: corr[i, i].item() for i, c in enumerate(CLASSES)},
            "mean_diagonal": corr.diag().mean().item(), "mean_off_diagonal": off.mean().item(),
            "matrix": corr.tolist()}


def weight_figures():
    W, W0 = first_layer(BEST, 0)                # (1024, 12288)
    D = W - W0
    g = torch.Generator().manual_seed(0)
    random_units = torch.randperm(W.shape[0], generator=g)[:8].sort().values
    top_units = D.norm(dim=1).argsort(descending=True)[:8]
    plot_weight_grid([W[i] for i in random_units] + [D[i] for i in random_units],
                     [f"unit {i}" for i in random_units.tolist()]
                     + [f"unit {i}: W−W₀" for i in random_units.tolist()],
                     os.path.join(FIG, "mlp_weights_random.pdf"), 8,
                     "MLP (mlp_best, seed 0), 8 random hidden units: trained first-layer weights W "
                     "(top) and the learned change W − W_init (bottom)")
    plot_weight_grid([D[i] for i in top_units], [f"unit {i}" for i in top_units.tolist()],
                     os.path.join(FIG, "mlp_weights_top.pdf"), 8,
                     "MLP: learned change W − W_init of the 8 units that changed most")

    T, T0 = first_layer(LINEAR, 0)                  # (10, 12288)
    plot_weight_grid(list(T) + list(T - T0), list(CLASSES) + [f"{c}: W−W₀" for c in CLASSES],
                     os.path.join(FIG, "linear_templates.pdf"), 5,
                     "Linear baseline (seed 0): class templates W (rows 1-2) and the learned "
                     "change W − W_init (rows 3-4)")

    stats = {
        "relative_change": {"mlp": (D.norm() / W0.norm()).item(),
                            "linear": ((T - T0).norm() / T0.norm()).item()},
        "cosine_W_Winit": {"mlp": torch.nn.functional.cosine_similarity(W.flatten(), W0.flatten(), 0).item(),
                           "linear": torch.nn.functional.cosine_similarity(T.flatten(), T0.flatten(), 0).item()},
        "neighbour_corr": {"mlp_W": smoothness(W), "mlp_delta": smoothness(D), "mlp_init": smoothness(W0),
                           "linear_W": smoothness(T), "linear_delta": smoothness(T - T0)},
        "rgb_corr": {"mlp_W": channel_corr(W), "mlp_delta": channel_corr(D),
                     "linear_W": channel_corr(T), "linear_delta": channel_corr(T - T0)},
        "random_units": random_units.tolist(), "top_change_units": top_units.tolist(),
    }
    stats["template_vs_class_mean"] = class_mean_figure(T - T0)
    print("weight statistics:", json.dumps(stats, indent=1))
    return stats


# ----------------------------------------------------------------------------- (b) shifts
def shift_accuracies(name, seeds=SEEDS, shifts=SHIFTS):
    """Test accuracy (mean_std over seeds) of the saved models of `name` per shift k:
    "right"/"down" = image shifted by k pixels (uncovered border = training mean colour).
    Two controls, the image is not moved:
      "<dir>_mask": the border that the shift uncovers is blanked (same blank strip, at the same
                    place, as in the shifted image);
      "<dir>_lost": the strip whose content the shift pushes out of the frame is blanked (same
                    content lost as in the shifted image).
    A model whose output depends on absolute positions drops below both controls; a perfectly
    translation-invariant model has shift == "lost"."""
    config = full_config(load_results(name)[0]["config"])
    test = get_loaders(config)["test"]
    device = run_device(config)   # evaluate on the device the model was trained on
    models = [load_model(name, seed).to(device) for seed in seeds]
    out = {}
    for direction in ("right", "down"):
        for mode, fn in (("shift", shift_images), ("mask", mask_border), ("lost", mask_lost)):
            key = direction if mode == "shift" else f"{direction}_{mode}"
            out[key] = {}
            for k in shifts:
                dx, dy = (k, 0) if direction == "right" else (0, k)
                out[key][k] = mean_std([accuracy_on(m, test, lambda x: fn(x, dx, dy))
                                        for m in models])
    return out


def plot_shifts(curves, path):
    """One row per model, one column per direction: accuracy on shifted images and on the two
    controls (mean ± std over seeds)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves), 2, figsize=(9, 2.6 * len(curves) + 0.3), sharex=True,
                             squeeze=False, layout="constrained")
    styles = (("", "o-", "C0", "shifted"),
              ("_mask", "--", "C1", "not shifted, uncovered border blanked"),
              ("_lost", ":", "C2", "not shifted, pushed-out strip blanked"))
    for row, (label, res) in zip(axes, curves.items()):
        for ax, direction in zip(row, ("right", "down")):
            ks = sorted(res[direction])
            for suffix, style, color, name in styles:
                m = torch.tensor([100 * res[direction + suffix][k]["mean"] for k in ks])
                s = torch.tensor([100 * res[direction + suffix][k]["std"] for k in ks])
                ax.plot(ks, m, style, color=color, label=name, ms=3)
                ax.fill_between(ks, m - s, m + s, color=color, alpha=0.15)
            ax.set_title(f"{label}: shift {direction}", fontsize=9)
            ax.set_ylabel("test accuracy (%)")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("shift (pixels)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


# ----------------------------------------------------------------------------- (c) permutation
def permutation_experiment(base_name, perm_name, seeds=SEEDS):
    """Retrain the configuration of `base_name` on permuted pixels; return both summaries."""
    base_runs = load_results(base_name)
    config = {**base_runs[0]["config"], "permutation_seed": PERMUTATION_SEED}
    config["hidden"] = tuple(config["hidden"])
    perm_runs = []
    for seed in seeds:
        r = cached_run(perm_name, config, seed, save_model=True)
        print(f"[{perm_name} seed {seed}] val {100 * r['val_acc']:.2f} | test "
              f"{100 * r['test_acc']:.2f} | best epoch {r['history']['best_epoch']}", flush=True)
        perm_runs.append(r)

    def summary(runs):
        s = {k: mean_std([r[k] for r in runs]) for k in ("train_acc", "val_acc", "test_acc")}
        s["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
        return s

    return summary(base_runs), summary(perm_runs), perm_runs


def main():
    os.makedirs(FIG, exist_ok=True)
    print("== (a) first-layer weights ==")
    weight_stats = weight_figures()

    print("\n== (b) shifted test images ==")
    shifts = {"MLP (mlp_best)": shift_accuracies(BEST),
              "linear baseline": shift_accuracies(LINEAR)}
    for label, res in shifts.items():
        for direction in ("right", "right_mask", "right_lost", "down", "down_mask", "down_lost"):
            print(f"{label:16s} {direction:10s} " + " | ".join(
                f"{k:2d}px {100 * res[direction][k]['mean']:5.2f}±{100 * res[direction][k]['std']:.2f}"
                for k in SHIFTS))
    plot_shifts(shifts, os.path.join(FIG, "shifts.pdf"))

    print("\n== (c) fixed pixel permutation, retrained from scratch ==")
    base, perm, perm_runs = permutation_experiment(BEST, PERM)
    for label, s in (("original pixels", base), ("permuted pixels", perm)):
        print(f"{label:16s} train {fmt(s['train_acc'])} | val {fmt(s['val_acc'])} | "
              f"test {fmt(s['test_acc'])} | best epochs {s['best_epochs']}")

    # The permuted model's weights, shown as they are and mapped back to the original layout.
    Wp, Wp0 = first_layer(PERM, 0)
    Dp = (Wp - Wp0).reshape(-1, 3, 64 * 64)
    inverse = torch.argsort(pixel_permutation(PERMUTATION_SEED))
    units = Dp.reshape(len(Dp), -1).norm(dim=1).argsort(descending=True)[:8]
    plot_weight_grid([Dp[i].reshape(-1) for i in units] + [Dp[i][:, inverse].reshape(-1) for i in units],
                     [f"unit {i}" for i in units.tolist()] + [f"{i} unpermuted" for i in units.tolist()],
                     os.path.join(FIG, "perm_weights.pdf"), 8,
                     "MLP trained on permuted pixels (seed 0), 8 units that changed most: W − W_init "
                     "as learned (top) and mapped back to the original pixel layout (bottom)")
    weight_stats["perm_neighbour_corr"] = {
        "as_learned": smoothness(Dp.reshape(len(Dp), -1)),
        "unpermuted": smoothness(Dp[:, :, inverse].reshape(len(Dp), -1))}
    print("permuted model, neighbour correlation of W - W_init:", weight_stats["perm_neighbour_corr"])

    print("\n== confusion matrix of the best MLP (test, 3 seeds) ==")
    conf = class_confusion(BEST)
    plot_confusion(conf["matrix"], os.path.join(FIG, "confusion_mlp.pdf"),
                   title="Best MLP (q2_2/mlp_best)")

    with open(os.path.join(RESULTS_DIR, "q2_4", "summary.json"), "w") as f:
        json.dump({"weights": weight_stats, "shifts": shifts,
                   "permutation": {"seed": PERMUTATION_SEED, "original": base, "permuted": perm},
                   "confusion_best_mlp": conf["summary"]},
                  f, indent=1)


if __name__ == "__main__":
    main()
