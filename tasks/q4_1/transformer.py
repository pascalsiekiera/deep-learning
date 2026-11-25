"""Q4.1 (bonus): a Transformer captioner, compared with the RNN captioner of Q3.2.

Model: models.TransformerCaptioner, a small pre-LayerNorm Transformer decoder (own decoder
blocks; attention from nn.MultiheadAttention) trained from scratch, with causal self-attention
over the caption and cross-attention to the 8 x 8 grid of features of the CNN of Q2.5
(pretrained on classification, frozen; the same encoder as the RNN captioner of Q3.2).

Protocol (as in Q3.2)
  1. Small search with seed 0: a base configuration and one change at a time; same budget as
     the RNN (at most 20 epochs, early stopping on validation loss with patience 4). Selection by
     validation BLEU-4 (greedy decoding).
  2. The selected configuration with seeds 0, 1, 2.
  3. Comparison with the RNN captioner (q3_2/final, seeds 0, 1, 2): caption metrics with greedy
     decoding and beam search (beam 5), number of parameters, training time per epoch (recorded,
     and a timing of 100 training steps of both models in the same session), decoding time.
  4. Cross-attention over the image regions for a few generated captions.

    python -m tasks.q4_1.transformer
    python -m tasks.q4_1.transformer --report-figure     # figures/q4_1/attention_report.pdf

Writes results/q4_1/..., figures/q4_1/{curves,attention_*}.pdf.
"""

import json
import os
import time

import torch
import torch.nn.functional as F

from captioning import (CAP_DEFAULTS, cached_caption_run, caption_line, caption_loss,
                        caption_metrics, device_for_captioning, generate, get_caption_loaders,
                        get_vocab, image_loader, load_captioner, build_captioner, build_optimizer,
                        cap_config)
from common import RESULTS_DIR, ROOT, SEEDS, get_data, mean_std, release_device_memory
from load_data import CLASSES
from tasks.q3_2.captioner import FINAL as RNN_FINAL
from tasks.q3_2.captioner import copy_run, plot_curves
from train import set_seed

OUT = os.path.join(RESULTS_DIR, "q4_1")
FIG = os.path.join(ROOT, "figures", "q4_1")

BASE = {"encoder": "cnn", "pretrained": True, "freeze": True, "decoder": "transformer"}
TRIALS = {                      # one change at a time w.r.t. BASE + TRANSFORMER_DEFAULTS
    "base": {},                 # d_model 256, 3 layers, 4 heads, d_ff 1024, dropout 0.1, lr 5e-4
    "dropout_0.3": {"dropout": 0.3},
    "lr_1e-3": {"lr": 1e-3},
    "layers_2": {"n_layers": 2},
    "layers_4": {"n_layers": 4},
}
SEARCH_SEED = 0
FINAL = "q4_1/final"
BEAM = 5


def decode_eval(model, split, beam):
    """Caption metrics and decoding time for greedy decoding (beam=1) or beam search."""
    vocab = get_vocab()
    decode = None if beam == 1 else (lambda m, x: m.beam_search(x, beam_size=beam)[0])
    device = next(model.parameters()).device
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    caps = generate(model, split, vocab, decode=decode)
    if device.type == "mps":
        torch.mps.synchronize()
    m = caption_metrics(caps, split)
    m["seconds"] = time.time() - t0
    m["distinct_captions"] = len(set(map(tuple, caps)))
    return m, caps


def evaluate_runs(name, out_dir=OUT):
    """Greedy and beam-search metrics of results/<name>/seed<k> (cached in <out_dir>/decoding/)."""
    path = os.path.join(out_dir, "decoding", name.replace("/", "_") + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = {}
    for seed in SEEDS:
        model = load_captioner(name, seed)
        out[str(seed)] = {}
        for beam in (1, BEAM):
            for split in ("validation", "test"):
                m, caps = decode_eval(model, split, beam)
                out[str(seed)][f"beam{beam}_{split}"] = m
                if split == "test":
                    out[str(seed)][f"beam{beam}_test_captions"] = caps
        del model
        release_device_memory()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def time_training_steps(config, steps=100, seed=0):
    """Seconds per training step (forward, backward, update), measured after 10 warm-up steps."""
    release_device_memory()
    set_seed(seed)
    config = cap_config(config)
    device = device_for_captioning()
    model = build_captioner(config, seed).to(device).train()
    opt = build_optimizer(config, model)
    pad = get_vocab().pad_id
    batches = iter(get_caption_loaders(config)["train"])
    for i in range(10 + steps):
        if i == 10:
            torch.mps.synchronize() if device.type == "mps" else None
            t0 = time.time()
        images, caps = next(batches)
        loss = caption_loss(model, (images.to(device), caps.to(device)), pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.mps.synchronize() if device.type == "mps" else None
    return (time.time() - t0) / steps


def plot_attention(model, indices, path, layer=-1, max_words=14, source="CNN"):
    """For each test image: the image and the cross-attention map (averaged over heads; `layer`
    = block index, or "mean" over blocks) used to generate each word, upsampled to 64 x 64."""
    import matplotlib.pyplot as plt

    vocab = get_vocab()
    data = get_data()["test"]
    device = next(model.parameters()).device
    images = torch.stack([image_loader("test").dataset[i][0] for i in indices]).to(device)
    ids, weights = model.attention(images)
    w = torch.stack(weights).mean(0) if layer == "mean" else weights[layer]
    w = w.cpu()
    rows = len(indices)
    fig, axes = plt.subplots(rows, max_words + 1, figsize=(1.25 * (max_words + 1), 1.45 * rows),
                             layout="constrained")
    for r, i in enumerate(indices):
        img = data["images"][i].permute(1, 2, 0)
        words = vocab.decode(ids[r], strip_special=False)
        words = [x for x in words if x not in ("<pad>",)] + (["<eos>"] if vocab.eos_id in ids[r] else [])
        for ax in axes[r]:
            ax.axis("off")
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"#{i} ({CLASSES[int(data['labels'][i])]})", fontsize=7)
        for t, word in enumerate(words[:max_words]):
            a = w[r, t][None, None]
            a = F.interpolate(a, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            ax = axes[r, t + 1]
            ax.imshow(img)
            ax.imshow(a, cmap="jet", alpha=0.5)
            ax.set_title(word, fontsize=8)
    title = "mean over blocks" if layer == "mean" else f"block {layer % len(weights) + 1} of {len(weights)}"
    fig.suptitle(f"Cross-attention over the {model.grid_size} × {model.grid_size} {source} grid when generating each word "
                 f"({title}, mean over heads; greedy decoding, seed 0)", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def attention_stats(model):
    """How focused the cross-attention is: mean entropy of the attention distribution over the
    64 regions (uniform = log 64 = 4.16) and mean maximum weight, per block, on the test set."""
    stats = {}
    with torch.no_grad():
        for images, _ in image_loader("test"):
            ids, weights = model.attention(images.to(next(model.parameters()).device))
            valid = torch.cat([torch.ones_like(ids[:, :1], dtype=torch.bool),
                               ids[:, :-1] != model.eos_id], 1) & (ids != model.pad_id)
            for b, w in enumerate(weights):
                w = w.flatten(2)[valid]                                   # (N, R)
                ent = -(w * w.clamp_min(1e-12).log()).sum(-1)
                s = stats.setdefault(b, {"entropy": 0.0, "max": 0.0, "n": 0})
                s["entropy"] += ent.sum().item()
                s["max"] += w.max(-1).values.sum().item()
                s["n"] += w.shape[0]
    return {f"block{b + 1}": {"entropy": s["entropy"] / s["n"], "max_weight": s["max"] / s["n"]}
            for b, s in stats.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    print(f"== 1. search (Transformer decoder, frozen CNN encoder, seed {SEARCH_SEED}) ==", flush=True)
    search = {}
    for name, change in TRIALS.items():
        r = cached_caption_run(f"q4_1/search/{name}", {**BASE, **change}, SEARCH_SEED)
        print(f"[{name:12s}] {caption_line(r)} | {r['num_parameters']['trainable']:,} trainable | "
              f"{sum(r['history']['seconds']) / len(r['history']['seconds']):.0f} s/epoch", flush=True)
        search[name] = r
    best = max(search, key=lambda n: search[n]["metrics"]["validation"]["bleu4"])
    best_config = {**BASE, **TRIALS[best]}
    print(f"selected by validation BLEU-4: {best} ({search[best]['metrics']['validation']['bleu4']:.2f})")

    print("\n== 2. selected configuration with 3 seeds ==", flush=True)
    copy_run(f"q4_1/search/{best}", FINAL, SEARCH_SEED)
    runs = [cached_caption_run(FINAL, best_config, seed) for seed in SEEDS]
    for r in runs:
        print(f"  seed {r['seed']}: {caption_line(r)}", flush=True)
    plot_curves(runs, os.path.join(FIG, "curves.pdf"), f"Transformer captioner ({best})")

    print("\n== 3. comparison with the RNN captioner of Q3.2 ==", flush=True)
    rnn_runs = []
    for seed in SEEDS:
        with open(os.path.join(RESULTS_DIR, RNN_FINAL, f"seed{seed}.json")) as f:
            rnn_runs.append(json.load(f))
    models = {"RNN (GRU, Q3.2)": (RNN_FINAL, rnn_runs, {"encoder": "cnn", "concat_image": True}),
              f"Transformer ({best})": (FINAL, runs, best_config)}
    comparison = {}
    for label, (name, rs, config) in models.items():
        dec = evaluate_runs(name)
        step_s = time_training_steps(config)
        c = {"params_total": rs[0]["num_parameters"]["total"],
             "params_trainable": rs[0]["num_parameters"]["trainable"],
             "best_epochs": [r["history"]["best_epoch"] for r in rs],
             "seconds_per_epoch_recorded": mean_std([sum(r["history"]["seconds"]) / len(r["history"]["seconds"])
                                                     for r in rs]),
             "seconds_to_best_epoch": mean_std([sum(r["history"]["seconds"][:r["history"]["best_epoch"]])
                                                for r in rs]),
             "seconds_per_train_step_timed": step_s,
             "val_ppl": mean_std([r["perplexity"]["validation"] for r in rs]),
             "test_ppl": mean_std([r["perplexity"]["test"] for r in rs])}
        for key in ("beam1_validation", "beam1_test", f"beam{BEAM}_validation", f"beam{BEAM}_test"):
            for k in ("bleu4", "class_mention", "mean_length", "distinct_1", "distinct_2",
                      "distinct_captions", "seconds"):
                c[f"{key}_{k}"] = mean_std([dec[str(s)][key][k] for s in SEEDS])
        comparison[label] = c
        print(f"\n{label}: params {c['params_total']:,} (trainable {c['params_trainable']:,}) | "
              f"best epochs {c['best_epochs']} | {c['seconds_per_epoch_recorded']['mean']:.0f} s/epoch recorded | "
              f"timed {1000 * step_s:.0f} ms/step | val ppl {c['val_ppl']['mean']:.2f} ± {c['val_ppl']['std']:.2f} | "
              f"test ppl {c['test_ppl']['mean']:.2f}")
        for key in ("beam1_validation", "beam1_test", f"beam{BEAM}_validation", f"beam{BEAM}_test"):
            f_ = lambda k, sc=1, d=2: f"{sc * c[f'{key}_{k}']['mean']:.{d}f} ± {sc * c[f'{key}_{k}']['std']:.{d}f}"
            print(f"  {key:18s} BLEU-4 {f_('bleu4')} | mention {f_('class_mention', 100, 1)}% | "
                  f"len {c[f'{key}_mean_length']['mean']:.2f} | distinct 1/2 {c[f'{key}_distinct_1']['mean']:.0f}/"
                  f"{c[f'{key}_distinct_2']['mean']:.0f} | distinct captions {c[f'{key}_distinct_captions']['mean']:.0f} | "
                  f"decoding {c[f'{key}_seconds']['mean']:.1f} s")

    print("\n== 4. cross-attention (seed 0) ==", flush=True)
    model = load_captioner(FINAL, 0)
    att = attention_stats(model)
    print("attention entropy (uniform = 4.16) and mean max weight (uniform = 0.016) per block:",
          {b: {k: round(v, 3) for k, v in s.items()} for b, s in att.items()})
    data = get_data()["test"]
    # Two images per class for a few classes, chosen by index (first images of each class).
    picks = []
    for cls in ("airplane", "dog", "elephant", "giraffe", "motorcycle", "truck"):
        c = CLASSES.index(cls)
        picks += [i for i in range(len(data["labels"])) if int(data["labels"][i]) == c][:1]
    for layer in (-1, "mean"):
        plot_attention(model, picks, os.path.join(FIG, f"attention_{'last' if layer == -1 else 'mean'}.pdf"),
                       layer=layer)

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"defaults": CAP_DEFAULTS, "base": BASE, "trials": TRIALS,
                   "search": {n: {"config": r["config"], "loss": r["loss"], "perplexity": r["perplexity"],
                                  "metrics": r["metrics"], "best_epoch": r["history"]["best_epoch"],
                                  "num_parameters": r["num_parameters"]}
                              for n, r in search.items()},
                   "selected": best, "selected_config": cap_config(best_config),
                   "comparison": comparison, "attention": att, "attention_images": picks}, f, indent=1)


def report_figure():
    """Smaller attention figure for the report (3 of the 6 images; run after main())."""
    with open(os.path.join(OUT, "summary.json")) as f:
        picks = json.load(f)["attention_images"]          # airplane, dog, elephant, giraffe, motorcycle, truck
    model = load_captioner(FINAL, 0)
    plot_attention(model, [picks[0], picks[3], picks[5]], os.path.join(FIG, "attention_report.pdf"),
                   max_words=11)


if __name__ == "__main__":
    import sys
    report_figure() if "--report-figure" in sys.argv else main()
