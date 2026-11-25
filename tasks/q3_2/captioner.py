"""Q3.2: an RNN captioner (CNN encoder pretrained on classification, frozen; GRU decoder).

Protocol
  1. A small decoder search with seed 0 (frozen CNN encoder from Q2.5): each trial changes one
     decoder/optimization setting from a base configuration. Early stopping on validation loss
     (patience 4, at most 20 epochs); the epoch with the lowest validation loss is restored.
  2. Selection by validation BLEU-4 (greedy decoding), the metric we report.
  3. The selected configuration with seeds 0, 1, 2: loss / perplexity curves, validation and
     test BLEU-4, and generated captions for test images (good and bad cases).

    python -m tasks.q3_2.captioner

Writes results/q3_2/<trial>/, results/q3_2/summary.json, figures/q3_2/{curves,examples}.pdf.
"""

import json
import math
import os
import shutil

import torch

from captioning import (CAP_DEFAULTS, cached_caption_run, cap_config, caption_line, references)
from common import CHECKPOINT_DIR, RESULTS_DIR, ROOT, SEEDS, get_data, mean_std
from load_data import CLASSES

OUT = os.path.join(RESULTS_DIR, "q3_2")
FIG = os.path.join(ROOT, "figures", "q3_2")

BASE = {"encoder": "cnn", "pretrained": True, "freeze": True}
TRIALS = {                                   # one change at a time w.r.t. BASE + CAP_DEFAULTS
    "base": {},
    "concat_image": {"concat_image": True},
    "dropout_0.5": {"dropout": 0.5},
    "lr_3e-4": {"lr": 3e-4},
    "hidden_256": {"hidden_dim": 256},
}
SEARCH_SEED = 0
FINAL = "q3_2/final"


def copy_run(src, dst, seed):
    """Copy a finished run (JSON + checkpoint), e.g. the seed-0 search run to the final runs."""
    s_json = os.path.join(RESULTS_DIR, src, f"seed{seed}.json")
    d_json = os.path.join(RESULTS_DIR, dst, f"seed{seed}.json")
    if os.path.exists(d_json) or not os.path.exists(s_json):
        return
    with open(s_json) as f:
        r = json.load(f)
    r["name"], r["copied_from"] = dst, src
    os.makedirs(os.path.dirname(d_json), exist_ok=True)
    with open(d_json, "w") as f:
        json.dump(r, f)
    os.makedirs(os.path.join(CHECKPOINT_DIR, dst), exist_ok=True)
    shutil.copyfile(os.path.join(CHECKPOINT_DIR, src, f"seed{seed}.pt"),
                    os.path.join(CHECKPOINT_DIR, dst, f"seed{seed}.pt"))


def summarize(runs):
    s = {}
    for split in ("validation", "test"):
        for k in ("bleu4", "class_mention", "mean_length", "distinct_1", "distinct_2"):
            s[f"{split}_{k}"] = mean_std([r["metrics"][split][k] for r in runs])
    for split in ("train", "validation", "test"):
        s[f"{split}_loss"] = mean_std([r["loss"][split] for r in runs])
        s[f"{split}_ppl"] = mean_std([r["perplexity"][split] for r in runs])
    s["best_epochs"] = [r["history"]["best_epoch"] for r in runs]
    s["num_parameters"] = runs[0]["num_parameters"]
    return s


def plot_curves(runs, path, title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for r in runs:
        h = r["history"]
        ep = list(range(1, len(h["train_loss"]) + 1))
        line, = axes[0].plot(ep, h["train_loss"], "--", lw=1.2, label=f"seed {r['seed']} train")
        axes[0].plot(ep, h["val_loss"], "-", color=line.get_color(), label=f"seed {r['seed']} val")
        axes[1].plot(ep, [math.exp(x) for x in h["train_loss"]], "--", lw=1.2, color=line.get_color())
        axes[1].plot(ep, [math.exp(x) for x in h["val_loss"]], "-", color=line.get_color())
        b = h["best_epoch"]
        axes[0].plot(b, h["val_loss"][b - 1], "o", color=line.get_color())
        axes[1].plot(b, math.exp(h["val_loss"][b - 1]), "o", color=line.get_color())
    axes[0].set_ylabel("cross-entropy per token")
    axes[1].set_ylabel("perplexity")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + "  (dashed: train, with dropout; solid: validation; dot: restored epoch)",
                 fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def sentence_score(hyp, refs):
    """Smoothed sentence-level BLEU-2 (add-one on the bigram precision), only used to *rank*
    test images for picking good and bad examples (corpus BLEU-4 is the reported metric)."""
    from collections import Counter
    if not hyp:
        return 0.0
    score = 1.0
    for n in (1, 2):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        m = Counter()
        for ref in refs:
            m |= Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        p = (sum((h & m).values()) + (n > 1)) / (sum(h.values()) + (n > 1))
        score *= p ** 0.5
    ref_len = min((len(r) for r in refs), key=lambda l: abs(l - len(hyp)))
    return score * min(1.0, math.exp(1 - ref_len / len(hyp)))


def pick_examples(captions, k_good=4, k_bad=4, k_random=4, seed=0):
    """Indices of test images: the k best and k worst by sentence score, plus k random ones."""
    refs = references("test")
    scores = torch.tensor([sentence_score(c, r) for c, r in zip(captions, refs)])
    order = scores.argsort(descending=True).tolist()
    g = torch.Generator().manual_seed(seed)
    rest = [i for i in torch.randperm(len(captions), generator=g).tolist()
            if i not in order[:k_good] + order[-k_bad:]]
    return ({"good": order[:k_good], "bad": order[-k_bad:][::-1], "random": rest[:k_random]},
            scores)


def plot_examples(picks, captions, scores, path):
    import matplotlib.pyplot as plt
    import textwrap

    data = get_data()["test"]
    items = [(kind, i) for kind in ("good", "random", "bad") for i in picks[kind]]
    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.25 * rows), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (kind, i) in zip(axes.flat, items):
        ax.imshow(data["images"][i].permute(1, 2, 0))
        gen = " ".join(captions[i])
        ref = data["captions"][i][0]
        label = CLASSES[int(data["labels"][i])]
        ax.set_title(f"[{kind}] #{i} ({label}), score {scores[i]:.2f}\n"
                     + "\n".join(textwrap.wrap("gen: " + gen, 42)) + "\n"
                     + "\n".join(textwrap.wrap("ref: " + ref, 42)), fontsize=6.5, loc="left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"== 1. decoder search (frozen CNN encoder, seed {SEARCH_SEED}) ==")
    search = {}
    for name, change in TRIALS.items():
        r = cached_caption_run(f"q3_2/search/{name}", {**BASE, **change}, SEARCH_SEED)
        print(f"[{name:13s}] {caption_line(r)}", flush=True)
        search[name] = r
    best = max(search, key=lambda n: search[n]["metrics"]["validation"]["bleu4"])
    best_config = {**BASE, **TRIALS[best]}
    print(f"selected by validation BLEU-4: {best} ({search[best]['metrics']['validation']['bleu4']:.2f})")

    print("\n== 2. selected configuration with 3 seeds ==")
    copy_run(f"q3_2/search/{best}", FINAL, SEARCH_SEED)
    runs = [cached_caption_run(FINAL, best_config, seed) for seed in SEEDS]
    for r in runs:
        print(f"  seed {r['seed']}: {caption_line(r)}")
    s = summarize(runs)
    for split in ("validation", "test"):
        print(f"{split:10s} BLEU-4 {s[f'{split}_bleu4']['mean']:.2f} ± {s[f'{split}_bleu4']['std']:.2f} | "
              f"loss {s[f'{split}_loss']['mean']:.3f} ± {s[f'{split}_loss']['std']:.3f} | "
              f"perplexity {s[f'{split}_ppl']['mean']:.2f} ± {s[f'{split}_ppl']['std']:.2f} | "
              f"class mention {100 * s[f'{split}_class_mention']['mean']:.1f}% | "
              f"mean length {s[f'{split}_mean_length']['mean']:.2f}")
    print(f"train      loss {s['train_loss']['mean']:.3f} ± {s['train_loss']['std']:.3f} | "
          f"perplexity {s['train_ppl']['mean']:.2f} (eval mode, no dropout)")
    print("parameters:", runs[0]["num_parameters"])
    plot_curves(runs, os.path.join(FIG, "curves.pdf"), f"Selected captioner ({best})")

    print("\n== 3. example captions (seed 0, test) ==")
    captions = runs[0]["captions"]["test"]
    picks, scores = pick_examples(captions)
    data = get_data()["test"]
    for kind in ("good", "random", "bad"):
        for i in picks[kind]:
            print(f"[{kind:6s}] #{i:3d} {CLASSES[int(data['labels'][i])]:10s} score {scores[i]:.2f} | "
                  f"gen: {' '.join(captions[i])}")
            for ref in data["captions"][i]:
                print(f"{'':36s}ref: {ref}")
    plot_examples(picks, captions, scores, os.path.join(FIG, "examples.pdf"))

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"defaults": CAP_DEFAULTS, "base": BASE, "trials": TRIALS,
                   "search": {n: {"config": r["config"], "loss": r["loss"], "perplexity": r["perplexity"],
                                  "metrics": r["metrics"], "best_epoch": r["history"]["best_epoch"]}
                              for n, r in search.items()},
                   "selected": best, "selected_config": cap_config(best_config), "final": s,
                   "examples": {k: [{"index": i, "class": CLASSES[int(data["labels"][i])],
                                     "score": scores[i].item(), "generated": " ".join(captions[i]),
                                     "references": data["captions"][i]} for i in v]
                                for k, v in picks.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
