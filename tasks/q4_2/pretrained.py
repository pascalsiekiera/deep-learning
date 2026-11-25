"""Q4.2 (bonus): pretrained image encoders, compared with our encoder trained from scratch.

Encoders (all frozen):
  * ours: the CNN of Q2.5 (1.2M parameters, trained by us on the 6,500 classification images),
  * ResNet-50 trained on ImageNet-1k (torchvision, 25.6M parameters; supervised classification),
  * the image encoder of CLIP ViT-B/32 (Hugging Face transformers, 87.8M parameters; trained
    contrastively on 400M image-text pairs from the web).
The pretrained encoders see our 64 x 64 images upsampled to 224 x 224 (see pretrained.py).

Decoders: our two decoders with exactly the configurations selected for our CNN (no new tuning):
the GRU of Q3.2 (image vector as initial state and at every step) and the Transformer of Q4.1
(cross-attention to the 7 x 7 grid of the pretrained encoder). Same protocol as before (at most
20 epochs, early stopping on validation loss with patience 4, seeds 0, 1, 2).

Also: linear-probe classification accuracy of each encoder's image vector (as in Q3.3 b), the
cost of each encoder (parameters, GFLOPs per image, feature-extraction time), training time per
step, greedy and beam-search (5) caption metrics, example captions, and CLIP cross-attention.

    python -m tasks.q4_2.pretrained

Writes results/q4_2/..., figures/q4_2/{examples,attention_clip}.pdf; feature stores in
checkpoints/q4_2/ (not part of the submission).
"""

import json
import os
import textwrap

import torch

from captioning import (cached_caption_run, caption_line, image_loader, load_captioner, norm_stats)
from common import RESULTS_DIR, ROOT, SEEDS, get_data, load_model, mean_std, release_device_memory
from load_data import CLASSES
from pretrained import BACKBONES, PretrainedBackbone, build_feature_store, cached_encoder
from tasks.q3_3.class_mention import linear_probe
from tasks.q4_1 import transformer as q41

OUT = os.path.join(RESULTS_DIR, "q4_2")
FIG = os.path.join(ROOT, "figures", "q4_2")
ENCODERS = ["resnet50", "clip"]
DECODERS = {"gru": {"concat_image": True}, "transformer": {"decoder": "transformer"}}
OWN = {"gru": "q3_2/final", "transformer": "q4_1/final"}       # our CNN encoder
OWN_CNN_RUN = "q2_5/cnn_best_50ep"
LABELS = {"cnn": "our CNN", "resnet50": "ResNet-50 (ImageNet)", "clip": "CLIP ViT-B/32"}
BEAM = q41.BEAM


def run_name(encoder, decoder):
    return OWN[decoder] if encoder == "cnn" else f"q4_2/{encoder}_{decoder}"


def config_of(encoder, decoder):
    return {"encoder": encoder, "pretrained": True, "freeze": True, **DECODERS[decoder]}


def gflops(fn, x):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            fn(x)
    return counter.get_total_flops() / 1e9


def encoder_costs():
    """Parameters, GFLOPs per 64 x 64 test image (forward pass, multiply-add = 2 FLOPs), and the
    time to compute the features of all 8,500 images on the Apple GPU."""
    x = next(iter(image_loader("test", batch_size=1)))[0]
    costs = {}
    cnn = load_model(OWN_CNN_RUN, 0).eval()
    costs["cnn"] = {"parameters": sum(p.numel() for p in cnn.parameters()),
                    "gflops_per_image": gflops(cnn.features, x), "input": "64 x 64"}
    for enc in ENCODERS:
        store = build_feature_store(enc)
        backbone = PretrainedBackbone(enc, *norm_stats()).eval()
        costs[enc] = {"parameters": sum(p.numel() for p in backbone.parameters()),
                      "gflops_per_image": gflops(backbone, x), "input": "224 x 224 (upsampled)",
                      "extraction_seconds_8500_images": store["seconds"],
                      "pooled_dim": BACKBONES[enc][0], "grid": f"{BACKBONES[enc][1]}x{BACKBONES[enc][1]}x{BACKBONES[enc][2]}"}
        del backbone
    return costs


def probes():
    """Linear-probe test accuracy of the image vector (as in Q3.3 b), 3 probe seeds."""
    out = {}
    with open(os.path.join(RESULTS_DIR, "q3_3", "class_mention.json")) as f:
        q33 = json.load(f)
    out["cnn"] = q33["rows"]["cnn_frozen"]["probe"]
    for enc in ENCODERS:
        accs = [linear_probe(cached_encoder(enc), seed=s) for s in SEEDS]
        out[enc] = mean_std([a["test_acc"] for a in accs])
        out[enc]["val"] = mean_std([a["val_acc"] for a in accs])
        out[enc]["epochs"] = [a["epoch"] for a in accs]
    return out


def plot_examples(indices, systems, path):
    """Test images with the greedy captions (seed 0) of several captioners."""
    import matplotlib.pyplot as plt

    data = get_data()["test"]
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.9 * rows), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        ax.imshow(data["images"][i].permute(1, 2, 0))
        lines = [f"#{i} ({CLASSES[int(data['labels'][i])]})"]
        for label, caps in systems.items():
            lines += textwrap.wrap(f"{label}: {' '.join(caps[i])}", 46)
        lines += textwrap.wrap("ref: " + data["captions"][i][0], 46)
        ax.set_title("\n".join(lines), fontsize=6.3, loc="left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    print("== 1. encoders: cost and linear-probe accuracy ==", flush=True)
    costs = encoder_costs()
    probe = probes()
    for enc in ["cnn"] + ENCODERS:
        c = costs[enc]
        print(f"{LABELS[enc]:22s} params {c['parameters']:>11,} | {c['gflops_per_image']:.2f} GFLOPs/image "
              f"({c['input']}) | extraction {c.get('extraction_seconds_8500_images', float('nan')):.0f} s | "
              f"linear probe test acc {100 * probe[enc]['mean']:.1f} ± {100 * probe[enc]['std']:.1f} %", flush=True)

    print("\n== 2. captioners with pretrained (frozen) encoders ==", flush=True)
    runs = {}
    for dec in DECODERS:
        for enc in ENCODERS:
            name = run_name(enc, dec)
            runs[(enc, dec)] = [cached_caption_run(name, config_of(enc, dec), s) for s in SEEDS]
            for r in runs[(enc, dec)]:
                print(f"[{enc:8s} {dec:11s} seed {r['seed']}] {caption_line(r)}", flush=True)
            release_device_memory()

    print("\n== 3. comparison (test unless stated; mean ± std over 3 seeds) ==", flush=True)
    table = {}
    for dec in DECODERS:
        for enc in ["cnn"] + ENCODERS:
            name = run_name(enc, dec)
            rs = []
            for s in SEEDS:
                with open(os.path.join(RESULTS_DIR, name, f"seed{s}.json")) as f:
                    rs.append(json.load(f))
            out_dir = q41.OUT if enc == "cnn" else OUT       # own runs: decoded in Q4.1
            decoded = q41.evaluate_runs(name, out_dir)
            step = q41.time_training_steps({"encoder": enc, **DECODERS[dec]} if enc == "cnn"
                                           else config_of(enc, dec))
            row = {"decoder_trainable_params": rs[0]["num_parameters"]["trainable"],
                   "encoder_params": costs[enc]["parameters"],
                   "best_epochs": [r["history"]["best_epoch"] for r in rs],
                   "val_ppl": mean_std([r["perplexity"]["validation"] for r in rs]),
                   "test_ppl": mean_std([r["perplexity"]["test"] for r in rs]),
                   "seconds_per_train_step": step,
                   "seconds_per_epoch_recorded": mean_std([sum(r["history"]["seconds"]) / len(r["history"]["seconds"]) for r in rs]),
                   "seconds_to_best_epoch": mean_std([sum(r["history"]["seconds"][:r["history"]["best_epoch"]]) for r in rs])}
            for key in ("beam1_validation", "beam1_test", f"beam{BEAM}_validation", f"beam{BEAM}_test"):
                for k in ("bleu4", "class_mention", "mean_length", "distinct_1", "distinct_2", "distinct_captions"):
                    row[f"{key}_{k}"] = mean_std([decoded[str(s)][key][k] for s in SEEDS])
            table[f"{enc}_{dec}"] = row
            f_ = lambda k, sc=1, d=2: f"{sc * row[k]['mean']:.{d}f} ± {sc * row[k]['std']:.{d}f}"
            print(f"{LABELS[enc]:22s} + {dec:11s} | val ppl {f_('val_ppl')} | BLEU-4 greedy val "
                  f"{f_('beam1_validation_bleu4')} test {f_('beam1_test_bleu4')} | beam {BEAM} val "
                  f"{f_(f'beam{BEAM}_validation_bleu4')} test {f_(f'beam{BEAM}_test_bleu4')} | mention "
                  f"{f_('beam1_test_class_mention', 100, 1)}% | distinct captions {row['beam1_test_distinct_captions']['mean']:.0f} | "
                  f"bigrams {row['beam1_test_distinct_2']['mean']:.0f} | len {row['beam1_test_mean_length']['mean']:.2f} | "
                  f"best epochs {row['best_epochs']} | {1000 * step:.0f} ms/step | "
                  f"{row['seconds_to_best_epoch']['mean']:.0f} s to best epoch", flush=True)
            release_device_memory()

    print("\n== 4. examples (seed 0, greedy, test) ==", flush=True)
    with open(os.path.join(RESULTS_DIR, "q3_2", "summary.json")) as f:
        q32 = json.load(f)["examples"]
    indices = [e["index"] for e in q32["bad"]] + [e["index"] for e in q32["random"]]
    systems = {}
    for label, name in (("CNN+GRU", run_name("cnn", "gru")), ("ResNet+GRU", run_name("resnet50", "gru")),
                        ("CLIP+GRU", run_name("clip", "gru")), ("CLIP+Transf.", run_name("clip", "transformer"))):
        with open(os.path.join(RESULTS_DIR, name, "seed0.json")) as f:
            systems[label] = json.load(f)["captions"]["test"]
    data = get_data()["test"]
    for i in indices:
        print(f"#{i} ({CLASSES[int(data['labels'][i])]}) ref: {data['captions'][i][0]}")
        for label, caps in systems.items():
            print(f"    {label:12s} {' '.join(caps[i])}")
    plot_examples(indices, systems, os.path.join(FIG, "examples.pdf"))

    print("\n== 5. CLIP cross-attention (Transformer, seed 0) ==", flush=True)
    model = load_captioner(run_name("clip", "transformer"), 0)
    att = q41.attention_stats(model)
    print("attention entropy (uniform = log 49 = 3.89) and mean max weight (uniform = 0.020) per block:",
          {b: {k: round(v, 3) for k, v in s.items()} for b, s in att.items()})
    with open(os.path.join(RESULTS_DIR, "q4_1", "summary.json")) as f:
        att_images = json.load(f)["attention_images"]
    q41.plot_attention(model, att_images, os.path.join(FIG, "attention_clip.pdf"), source="CLIP")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"encoders": costs, "linear_probe": probe, "table": table, "attention": att,
                   "example_indices": indices,
                   "examples": {str(i): {label: " ".join(c[i]) for label, c in systems.items()}
                                for i in indices}}, f, indent=1)


if __name__ == "__main__":
    main()
