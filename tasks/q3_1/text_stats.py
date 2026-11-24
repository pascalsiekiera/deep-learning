"""Q3.1: tokenizer and vocabulary statistics.

    python -m tasks.q3_1.text_stats

Reports the vocabulary size (and how it depends on the frequency threshold), the fraction of
<unk> tokens in the validation captions, the distribution of caption lengths, how the provided
loaders batch captions of different lengths, and checks that padding does not change the loss.
Writes results/q3_1/{summary.json, vocab.json, log.txt via the shell} and
figures/q3_1/caption_lengths.pdf.
"""

import json
import os

import torch
import torch.nn.functional as F

from common import ROOT, RESULTS_DIR, get_data
from data import caption_loaders, channel_stats
from load_data import SPLITS
from metrics import class_mention_accuracy
from text import Vocabulary, tokenize

MIN_COUNT = 5                       # chosen threshold (see the comparison below)
THRESHOLDS = [1, 2, 3, 5, 10, 20]
OUT = os.path.join(RESULTS_DIR, "q3_1")
FIG = os.path.join(ROOT, "figures", "q3_1")


def unk_stats(vocab, captions):
    """Fraction of <unk> among the word tokens (without <bos>/<eos>), and fraction of captions
    with at least one <unk>."""
    n_tokens = n_unk = n_caps_with_unk = n_caps = 0
    for image_captions in captions:
        for c in image_captions:
            ids = vocab.encode(c)[1:-1]
            unk = sum(i == vocab.unk_id for i in ids)
            n_tokens, n_unk = n_tokens + len(ids), n_unk + unk
            n_caps, n_caps_with_unk = n_caps + 1, n_caps_with_unk + (unk > 0)
    return {"unk_token_fraction": n_unk / n_tokens, "captions_with_unk": n_caps_with_unk / n_caps,
            "tokens": n_tokens}


def length_stats(captions):
    """Caption lengths in tokens (words and punctuation, without <bos>/<eos>)."""
    lengths = torch.tensor([len(tokenize(c)) for ic in captions for c in ic], dtype=torch.float)
    q = torch.quantile(lengths, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    return lengths, {"mean": lengths.mean().item(), "std": lengths.std().item(),
                     "min": int(lengths.min()), "max": int(lengths.max()),
                     "quantiles": dict(zip(["5%", "25%", "50%", "75%", "95%", "99%"], q.tolist())),
                     "fraction_le_15": (lengths <= 15).float().mean().item()}


def padding_demo(vocab, data):
    """How caption_loaders batches captions: shapes and padding of the first training batches,
    the fraction of padding over one epoch, and a check that the loss with ignore_index is
    unchanged when extra padding is appended."""
    mean, std = channel_stats(data["train"]["images"])
    torch.manual_seed(0)
    train_loader, _, _ = caption_loaders(data, vocab.encode, vocab.pad_id, batch_size=64,
                                         mean=mean, std=std)
    shapes, pad_fracs, total, pads = [], [], 0, 0
    for b, (images, captions) in enumerate(train_loader):
        n_pad = (captions == vocab.pad_id).sum().item()
        total, pads = total + captions.numel(), pads + n_pad
        if b < 3:
            shapes.append([list(images.shape), list(captions.shape)])
            pad_fracs.append(n_pad / captions.numel())
            if b == 0:
                example = {"longest": vocab.decode(captions[(captions != vocab.pad_id).sum(1).argmax()],
                                                   strip_special=False),
                           "padded_row": [vocab.itos[i] for i in
                                          captions[(captions != vocab.pad_id).sum(1).argmin()].tolist()]}
    # Padding check: random logits for a padded batch; appending more <pad> columns must not
    # change the loss when the pads are ignored, but does change it when they are not.
    g = torch.Generator().manual_seed(0)
    caps = captions                                              # last batch (B, T)
    extra = torch.cat([caps, torch.full((caps.shape[0], 5), vocab.pad_id)], dim=1)
    logits = torch.randn(caps.shape[0], extra.shape[1] - 1, len(vocab), generator=g)

    def loss(c, ignore):
        targets = c[:, 1:]
        lg = logits[:, :targets.shape[1]]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), targets.reshape(-1),
                               ignore_index=vocab.pad_id if ignore else -100).item()

    return {"first_batches_shapes_images_captions": shapes, "first_batches_pad_fraction": pad_fracs,
            "epoch_pad_fraction": pads / total, "example_batch0": example,
            "loss_ignore_pad": {"original": loss(caps, True), "with_5_extra_pad_columns": loss(extra, True)},
            "loss_counting_pad": {"original": loss(caps, False), "with_5_extra_pad_columns": loss(extra, False)}}


def plot_lengths(lengths, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
    for split, l in lengths.items():
        bins = torch.arange(0.5, int(max(x.max() for x in lengths.values())) + 1.5)
        hist = torch.histc(l, bins=len(bins) - 1, min=bins[0].item(), max=bins[-1].item())
        ax.plot(bins[:-1] + 0.5, 100 * hist / hist.sum(), "o-", ms=2.5, label=f"{split} ({len(l)} captions)")
    ax.set_xlim(0, 35)
    ax.set_xlabel("caption length (tokens incl. punctuation, without <bos>/<eos>)")
    ax.set_ylabel("% of captions")
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = get_data()
    train_caps, val_caps = data["train"]["captions"], data["validation"]["captions"]

    print("== vocabulary size vs. frequency threshold (built from the training captions) ==")
    print("min_count | vocab size (incl. 4 special) | val <unk> tokens | val captions with <unk>")
    by_threshold = {}
    for k in THRESHOLDS:
        v = Vocabulary.from_captions(train_caps, min_count=k)
        u = unk_stats(v, val_caps)
        by_threshold[k] = {"vocab_size": len(v), **u}
        print(f"{k:9d} | {len(v):28d} | {100 * u['unk_token_fraction']:15.2f}% | "
              f"{100 * u['captions_with_unk']:6.2f}%")

    vocab = Vocabulary.from_captions(train_caps, min_count=MIN_COUNT)
    counts = vocab.counts
    print(f"\nchosen min_count = {MIN_COUNT}: vocabulary size {len(vocab)} "
          f"({len(vocab) - 4} words + <pad>, <bos>, <eos>, <unk>)")
    print(f"distinct training tokens: {len(counts)}, of which {len(counts) - (len(vocab) - 4)} "
          f"occur < {MIN_COUNT} times; they cover "
          f"{100 * sum(n for w, n in counts.items() if n < MIN_COUNT) / sum(counts.values()):.2f}% "
          f"of the training tokens")
    unk = {split: unk_stats(vocab, data[split]["captions"]) for split in SPLITS}
    for split, u in unk.items():
        print(f"{split:10s} <unk> tokens {100 * u['unk_token_fraction']:.2f}% of "
              f"{u['tokens']} tokens | captions with <unk> {100 * u['captions_with_unk']:.2f}%")
    print("20 most frequent words:", [(w, counts[w]) for w in vocab.itos[4:24]])

    print("\n== caption lengths (tokens incl. punctuation, without <bos>/<eos>) ==")
    lengths, length_summary = {}, {}
    for split in SPLITS:
        lengths[split], length_summary[split] = length_stats(data[split]["captions"])
        s = length_summary[split]
        print(f"{split:10s} mean {s['mean']:.2f} ± {s['std']:.2f} | min {s['min']} | max {s['max']} | "
              + " ".join(f"{k} {v:.0f}" for k, v in s["quantiles"].items())
              + f" | <= 15 tokens: {100 * s['fraction_le_15']:.1f}%")
    plot_lengths(lengths, os.path.join(FIG, "caption_lengths.pdf"))

    # Class mentions in the reference captions: an upper reference point for Q3.3 (b).
    mention = {}
    for split in SPLITS:
        caps = [tokenize(c) for ic in data[split]["captions"] for c in ic]
        labels = [int(l) for l in data[split]["labels"] for _ in range(5)]
        mention[split] = class_mention_accuracy(caps, labels)
    print("\nclass mention accuracy of the human reference captions:",
          {k: round(100 * v, 2) for k, v in mention.items()})

    print("\n== batching with caption_loaders (batch size 64) ==")
    demo = padding_demo(vocab, data)
    print("first batches (images, captions) shapes:", demo["first_batches_shapes_images_captions"])
    print("pad fraction in these batches:", [round(x, 3) for x in demo["first_batches_pad_fraction"]],
          f"| over one epoch: {100 * demo['epoch_pad_fraction']:.1f}%")
    print("example from batch 0, longest caption:", demo["example_batch0"]["longest"])
    print("example from batch 0, shortest caption as padded row:", demo["example_batch0"]["padded_row"])
    print("loss with ignore_index=<pad>:", demo["loss_ignore_pad"])
    print("loss counting <pad> as a target:", demo["loss_counting_pad"])

    with open(os.path.join(OUT, "vocab.json"), "w") as f:
        json.dump({"min_count": MIN_COUNT, "itos": vocab.itos}, f)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"min_count": MIN_COUNT, "vocab_size": len(vocab), "by_threshold": by_threshold,
                   "unk": unk, "lengths": length_summary, "reference_class_mention": mention,
                   "batching": demo}, f, indent=1)


if __name__ == "__main__":
    main()
