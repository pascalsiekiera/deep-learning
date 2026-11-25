"""Q3.3 (b): class mention accuracy of the captioners of Q3.3 (a), related to the encoder's
classification accuracy and to BLEU-4. Run after tasks.q3_3.encoders (uses its saved captions
and checkpoints; no captioner is retrained).

  * Class mention accuracy with the original CLASS_SYNONYMS and with an extended list (our
    extension: possessives such as "dog's", which our tokenizer keeps as one token, and a few
    frequent words for the classes found in the training captions). metrics.py is not changed.
  * Classification accuracy of each encoder: for the pretrained encoders, the test accuracy of
    the Part 2 classifier; for every encoder (also the jointly trained ones, which have no
    trained classification layer), a linear probe: a softmax classifier trained on the frozen
    encoder features of the training images (epochs chosen on validation), test accuracy.
  * Image level (frozen encoders): how often the caption names the right class when the
    classifier is right vs. wrong.

    python -m tasks.q3_3.class_mention

Writes results/q3_3/class_mention.json and figures/q3_3/{metrics_scatter,mention_per_class}.pdf.
"""

import json
import os

import torch
import torch.nn.functional as F

from captioning import device_for_captioning, image_loader, load_captioner
from common import RESULTS_DIR, ROOT, SEEDS, get_data, load_model, mean_std, predict
from load_data import CLASSES
from metrics import CLASS_SYNONYMS, class_mention_accuracy
from tasks.q3_3.encoders import CLASSIFIER_RUNS, ENCODERS, MODES, variant_name

OUT = os.path.join(RESULTS_DIR, "q3_3")
FIG = os.path.join(ROOT, "figures", "q3_3")

EXTRA_SYNONYMS = {
    "airplane": ["biplane"],
    "dog": ["retriever", "husky", "shepherd", "collie", "terrier"],
    "horse": ["horseback"],
    "motorcycle": ["motorcyclist"],
    "train": ["subway", "trolley", "tram"],
    "truck": ["firetruck", "pickup"],
}


def mention_hits(captions, labels, synonyms):
    """Per caption: does it contain a word naming the image's class? (like
    metrics.class_mention_accuracy, with a given synonym list and possessive forms)."""
    hits = []
    for hyp, label in zip(captions, labels):
        words = synonyms[CLASSES[label]]
        names = {w + suffix for w in words for suffix in ("", "s", "'s")}
        hits.append(any(t in names for t in hyp))
    return torch.tensor(hits, dtype=torch.float)


def extended_synonyms():
    return {c: list(ws) + EXTRA_SYNONYMS.get(c, []) for c, ws in CLASS_SYNONYMS.items()}


@torch.no_grad()
def encoder_features(encoder, split):
    encoder.eval()
    params = list(encoder.parameters())   # (a cached pretrained encoder of Q4.2 has none)
    device = params[0].device if params else device_for_captioning()
    return torch.cat([encoder.features(x.to(device)).cpu() for x, _ in image_loader(split)])


def linear_probe(encoder, epochs=300, lr=1e-2, weight_decay=1e-4, seed=0):
    """Softmax classifier on standardized frozen features (full batch Adam); returns the test
    accuracy at the epoch with the best validation accuracy."""
    data = get_data()
    feats = {s: encoder_features(encoder, s) for s in ("train", "validation", "test")}
    mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-6
    feats = {s: (f - mu) / sd for s, f in feats.items()}
    y = {s: data[s]["labels"] for s in feats}
    g = torch.Generator().manual_seed(seed)
    W = (0.01 * torch.randn(feats["train"].shape[1], len(CLASSES), generator=g)).requires_grad_()
    b = torch.zeros(len(CLASSES), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    best = (-1.0, None, None)
    for epoch in range(epochs):
        opt.zero_grad()
        F.cross_entropy(feats["train"] @ W + b, y["train"]).backward()
        opt.step()
        with torch.no_grad():
            acc = {s: ((feats[s] @ W + b).argmax(1) == y[s]).float().mean().item()
                   for s in ("validation", "test")}
        if acc["validation"] > best[0]:
            best = (acc["validation"], acc["test"], epoch + 1)
    return {"val_acc": best[0], "test_acc": best[1], "epoch": best[2]}


def main():
    os.makedirs(FIG, exist_ok=True)
    data = get_data()
    labels = {s: data[s]["labels"].tolist() for s in ("validation", "test")}
    ext = extended_synonyms()

    # Effect of the extension on the human references (upper reference point).
    from text import tokenize
    ref_mention = {}
    for split in ("validation", "test"):
        caps = [tokenize(c) for ic in data[split]["captions"] for c in ic]
        lab = [l for l in labels[split] for _ in range(5)]
        ref_mention[split] = {"original": class_mention_accuracy(caps, lab),
                              "extended": mention_hits(caps, lab, ext).mean().item()}
    print("human references, class mention (original / extended list):",
          {s: f"{100 * v['original']:.2f} / {100 * v['extended']:.2f}" for s, v in ref_mention.items()})

    # Classification accuracy of the Part 2 classifiers (pretrained encoders) and their test
    # predictions (for the image-level analysis).
    clf_acc, clf_pred = {}, {}
    for enc, run in CLASSIFIER_RUNS.items():
        clf_acc[enc] = {}
        for seed in SEEDS:
            model = load_model(run, seed).to(device_for_captioning())
            preds, _ = predict(model, image_loader("test"))
            clf_pred[(enc, seed)] = preds
            clf_acc[enc][seed] = (preds == data["test"]["labels"]).float().mean().item()

    rows, per_run = {}, []
    for enc in ENCODERS:
        for mode in MODES:
            name = variant_name(enc, mode)
            accs = {k: [] for k in ("bleu4", "mention", "mention_ext", "probe", "classifier",
                                    "mention_if_clf_right", "mention_if_clf_wrong")}
            per_class = []
            for seed in SEEDS:
                with open(os.path.join(RESULTS_DIR, name, f"seed{seed}.json")) as f:
                    r = json.load(f)
                caps = r["captions"]["test"]
                hits_ext = mention_hits(caps, labels["test"], ext)
                model = load_captioner(name, seed)
                probe = linear_probe(model.encoder, seed=seed)
                accs["bleu4"].append(r["metrics"]["test"]["bleu4"])
                accs["mention"].append(class_mention_accuracy(caps, labels["test"]))
                accs["mention_ext"].append(hits_ext.mean().item())
                accs["probe"].append(probe["test_acc"])
                per_class.append(torch.stack([hits_ext[torch.tensor(labels["test"]) == c].mean()
                                              for c in range(len(CLASSES))]))
                if mode == "frozen":
                    right = clf_pred[(enc, seed)] == data["test"]["labels"]
                    accs["classifier"].append(clf_acc[enc][seed])
                    accs["mention_if_clf_right"].append(hits_ext[right].mean().item())
                    accs["mention_if_clf_wrong"].append(hits_ext[~right].mean().item())
                per_run.append({"encoder": enc, "mode": mode, "seed": seed,
                                "bleu4": accs["bleu4"][-1], "mention_ext": accs["mention_ext"][-1],
                                "probe": accs["probe"][-1]})
                print(f"[{enc} {mode:9s} seed {seed}] BLEU-4 {accs['bleu4'][-1]:.2f} | mention "
                      f"{100 * accs['mention'][-1]:.1f}% (ext. {100 * accs['mention_ext'][-1]:.1f}%) | "
                      f"probe acc {100 * probe['test_acc']:.1f}% (epoch {probe['epoch']})", flush=True)
                del model
            rows[f"{enc}_{mode}"] = {k: mean_std(v) for k, v in accs.items() if v}
            rows[f"{enc}_{mode}"]["per_class_mention_ext"] = {
                c: mean_std(torch.stack(per_class)[:, i].tolist()) for i, c in enumerate(CLASSES)}

    pct = lambda s: f"{100 * s['mean']:.1f} ± {100 * s['std']:.1f}"
    print("\n| encoder | mode | test BLEU-4 | class mention | class mention (ext.) | linear probe acc "
          "| classifier acc | mention if clf right / wrong |")
    for key, s in rows.items():
        enc, mode = key.split("_")
        clf = pct(s["classifier"]) if "classifier" in s else "–"
        rw = (f"{pct(s['mention_if_clf_right'])} / {pct(s['mention_if_clf_wrong'])}"
              if "mention_if_clf_right" in s else "–")
        print(f"| {enc} | {mode} | {s['bleu4']['mean']:.2f} ± {s['bleu4']['std']:.2f} | "
              f"{pct(s['mention'])} | {pct(s['mention_ext'])} | {pct(s['probe'])} | {clf} | {rw} |")

    # Correlations over the 18 runs.
    t = lambda k: torch.tensor([r[k] for r in per_run])
    corr = lambda a, b: torch.corrcoef(torch.stack([t(a), t(b)]))[0, 1].item()
    correlations = {"bleu4_vs_mention": corr("bleu4", "mention_ext"),
                    "probe_vs_mention": corr("probe", "mention_ext"),
                    "probe_vs_bleu4": corr("probe", "bleu4")}
    print("Pearson correlations over the 18 runs:", {k: round(v, 3) for k, v in correlations.items()})

    plot_scatter(per_run, os.path.join(FIG, "metrics_scatter.pdf"))
    plot_per_class(rows, os.path.join(FIG, "mention_per_class.pdf"))
    with open(os.path.join(OUT, "class_mention.json"), "w") as f:
        json.dump({"extra_synonyms": EXTRA_SYNONYMS, "reference_mention": ref_mention,
                   "rows": rows, "per_run": per_run, "correlations": correlations}, f, indent=1)


def plot_scatter(per_run, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    markers = {"frozen": "o", "finetuned": "s", "scratch": "^"}
    colors = {"cnn": "C0", "mlp": "C1"}
    pairs = (("probe", "mention_ext", "linear-probe accuracy (%)", "class mention (%)"),
             ("mention_ext", "bleu4", "class mention (%)", "BLEU-4"),
             ("probe", "bleu4", "linear-probe accuracy (%)", "BLEU-4"))
    for ax, (xk, yk, xl, yl) in zip(axes, pairs):
        for r in per_run:
            x = r[xk] * (100 if xk != "bleu4" else 1)
            y = r[yk] * (100 if yk != "bleu4" else 1)
            ax.scatter(x, y, marker=markers[r["mode"]], color=colors[r["encoder"]], s=28)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    for enc, c in colors.items():
        for mode, m in markers.items():
            axes[0].scatter([], [], marker=m, color=c, label=f"{enc.upper()} {mode}")
    axes[0].legend(fontsize=7)
    fig.suptitle("Test set, one point per run (3 seeds × 6 captioners); class mention with the "
                 "extended synonym list", fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def plot_per_class(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    keys = list(rows)
    width = 0.8 / len(keys)
    for k, key in enumerate(keys):
        m = [100 * rows[key]["per_class_mention_ext"][c]["mean"] for c in CLASSES]
        ax.bar([i + (k - len(keys) / 2 + 0.5) * width for i in range(len(CLASSES))], m, width,
               label=key.replace("_", " "))
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylabel("class mention, test (%)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
