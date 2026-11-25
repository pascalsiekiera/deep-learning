"""Q3.4 (a): beam search vs greedy decoding.

Decodes the validation and test images with the selected captioner of Q3.2 (frozen CNN,
concat_image; seeds 0, 1, 2; no retraining) using beam sizes 1, 3, 5 (and 10 as an extra point),
ranking the finished captions by log p(y | x) or by log p(y | x) / L (length normalization).
Beam size 1 is greedy decoding (checked against the stored greedy captions).

Per setting: BLEU-4, mean length, distinct words / bigrams (metrics.distinct_ngrams), class
mention, mean log p(y | x) of the returned captions, number of distinct captions, fraction of
captions with a repeated bigram, fraction cut at max_len.

    python -m tasks.q3_4.decoding

Writes results/q3_4/seed<k>.json, results/q3_4/summary.json, figures/q3_4/tradeoffs.pdf.
"""

import json
import os
import time

import torch

from captioning import caption_metrics, get_vocab, image_loader, load_captioner
from common import RESULTS_DIR, ROOT, SEEDS, get_data, mean_std
from load_data import CLASSES
from tasks.q3_2.captioner import FINAL

OUT = os.path.join(RESULTS_DIR, "q3_4")
FIG = os.path.join(ROOT, "figures", "q3_4")
MAX_LEN = 25
# (beam size, length normalization); beam 1 is the same with and without normalization.
SETTINGS = [(1, False), (3, False), (3, True), (5, False), (5, True), (10, False), (10, True)]
SPLITS = ("validation", "test")


def setting_name(beam, norm):
    return f"beam{beam}" + ("_norm" if norm else "")


@torch.no_grad()
def decode_split(model, split, vocab, beam, norm):
    model.eval()
    device = next(model.parameters()).device
    captions, logps, cut = [], [], 0
    for images, _ in image_loader(split):
        ids, lp = model.beam_search(images.to(device), beam_size=beam, length_norm=norm,
                                    max_len=MAX_LEN)
        for row in ids:
            row = row.tolist()
            cut += vocab.eos_id not in row
            captions.append(vocab.decode(torch.tensor(row)))
        logps += lp.tolist()
    return captions, logps, cut / len(captions)


def repeated_bigram(caption):
    bigrams = list(zip(caption, caption[1:]))
    return len(bigrams) != len(set(bigrams))


def evaluate(captions, logps, cut, split):
    m = caption_metrics(captions, split)
    m.update({"mean_logp": sum(logps) / len(logps),
              "mean_logp_per_token": sum(lp / (len(c) + 1) for lp, c in zip(logps, captions)) / len(captions),
              "distinct_captions": len(set(map(tuple, captions))),
              "repeated_bigram": sum(map(repeated_bigram, captions)) / len(captions),
              "cut_at_max_len": cut,
              "length_hist": {str(l): n for l, n in sorted(
                  {l: sum(len(c) == l for c in captions) for l in set(map(len, captions))}.items())}})
    return m


def run_seed(seed):
    path = os.path.join(OUT, f"seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    vocab = get_vocab()
    model = load_captioner(FINAL, seed)
    with open(os.path.join(RESULTS_DIR, FINAL, f"seed{seed}.json")) as f:
        greedy = json.load(f)["captions"]
    result = {"seed": seed, "model": FINAL, "max_len": MAX_LEN, "settings": {}}
    for beam, norm in SETTINGS:
        name = setting_name(beam, norm)
        result["settings"][name] = {"beam": beam, "length_norm": norm}
        for split in SPLITS:
            t0 = time.time()
            caps, logps, cut = decode_split(model, split, vocab, beam, norm)
            m = evaluate(caps, logps, cut, split)
            m["same_as_greedy"] = sum(c == g for c, g in zip(caps, greedy[split])) / len(caps)
            m["seconds"] = time.time() - t0
            result["settings"][name][split] = {"metrics": m, "captions": caps}
            print(f"[seed {seed}] {name:11s} {split:10s} BLEU-4 {m['bleu4']:.2f} | len "
                  f"{m['mean_length']:.2f} | distinct 1/2 {m['distinct_1']}/{m['distinct_2']} | "
                  f"mention {100 * m['class_mention']:.1f}% | log p {m['mean_logp']:.2f} | "
                  f"= greedy {100 * m['same_as_greedy']:.1f}% | {m['seconds']:.0f}s", flush=True)
    if any(result["settings"]["beam1"][s]["metrics"]["same_as_greedy"] != 1.0 for s in SPLITS):
        print("WARNING: beam size 1 differs from the stored greedy captions")
    with open(path, "w") as f:
        json.dump(result, f)
    return result


def plot_tradeoffs(summary, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), layout="constrained")
    keys = (("bleu4", "BLEU-4"), ("mean_length", "mean length (tokens)"),
            ("distinct_2", "distinct bigrams"), ("mean_logp", "mean log p(y | x)"))
    beams = sorted({b for b, _ in SETTINGS})
    for ax, (k, label) in zip(axes, keys):
        for norm, style in ((False, "o-"), (True, "s--")):
            bs = [b for b in beams if (b, norm) in SETTINGS or b == 1]
            names = [setting_name(b, norm and b > 1) for b in bs]
            m = [summary[n]["test"][k]["mean"] for n in names]
            s = [summary[n]["test"][k]["std"] for n in names]
            ax.errorbar(bs, m, yerr=s, fmt=style, capsize=3, ms=4,
                        label="length-normalized" if norm else "log-probability")
        ax.set_xscale("log")
        ax.set_xticks(beams, [str(b) for b in beams])
        ax.set_xlabel("beam size")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Test set, captioner of Q3.2 (mean ± std over 3 seeds); beam size 1 = greedy",
                 fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [run_seed(seed) for seed in SEEDS]
    keys = ("bleu4", "mean_length", "distinct_1", "distinct_2", "class_mention", "mean_logp",
            "mean_logp_per_token", "distinct_captions", "repeated_bigram", "cut_at_max_len",
            "same_as_greedy", "seconds")
    summary = {}
    for beam, norm in SETTINGS:
        name = setting_name(beam, norm)
        summary[name] = {split: {k: mean_std([r["settings"][name][split]["metrics"][k] for r in runs])
                                 for k in keys} for split in SPLITS}

    for split in SPLITS:
        print(f"\n== {split} (mean ± std over seeds {SEEDS}) ==")
        print("| setting | BLEU-4 | mean length | distinct words | distinct bigrams | class mention "
              "| mean log p | distinct captions | repeated bigram | = greedy | s |")
        for name, s in summary.items():
            v = s[split]
            f = lambda k, d=2, sc=1: f"{sc * v[k]['mean']:.{d}f} ± {sc * v[k]['std']:.{d}f}"
            print(f"| {name} | {f('bleu4')} | {f('mean_length')} | {f('distinct_1', 0)} | "
                  f"{f('distinct_2', 0)} | {f('class_mention', 1, 100)} | {f('mean_logp')} | "
                  f"{f('distinct_captions', 0)} | {f('repeated_bigram', 1, 100)} | "
                  f"{f('same_as_greedy', 1, 100)} | {v['seconds']['mean']:.0f} |")

    # Examples (seed 0, test): images where the settings differ most.
    r0 = runs[0]["settings"]
    data = get_data()["test"]
    caps = {n: r0[n]["test"]["captions"] for n in r0}
    differ = [i for i in range(len(data["labels"]))
              if len({tuple(caps[n][i]) for n in ("beam1", "beam5", "beam5_norm", "beam10")}) >= 3]
    print(f"\n== examples (seed 0, test): {len(differ)} images with >= 3 different captions "
          "among beam1 / beam5 / beam5_norm / beam10 ==")
    examples = []
    for i in differ[:8]:
        print(f"#{i} ({CLASSES[int(data['labels'][i])]}) ref: {data['captions'][i][0]}")
        for n in ("beam1", "beam3", "beam5", "beam5_norm", "beam10", "beam10_norm"):
            print(f"    {n:11s} {' '.join(caps[n][i])}")
        examples.append({"index": i, "class": CLASSES[int(data["labels"][i])],
                         "reference": data["captions"][i][0],
                         **{n: " ".join(caps[n][i]) for n in caps}})

    plot_tradeoffs(summary, os.path.join(FIG, "tradeoffs.pdf"))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"model": FINAL, "max_len": MAX_LEN, "settings": SETTINGS, "summary": summary,
                   "examples": examples}, f, indent=1)


if __name__ == "__main__":
    main()
