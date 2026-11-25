"""Shared code for Part 3 (image captioning): building a captioner from a config dict, training
it with the provided training loop (teacher forcing, token-level cross-entropy that ignores
<pad>), computing losses / perplexities, greedy decoding, and caption metrics. Every run is
saved as JSON in results/<name>/seed<k>.json (and the model in checkpoints/<name>/).

    from captioning import cached_caption_run
    result = cached_caption_run("q3_2/example", {"encoder": "cnn", "freeze": True}, seed=0)
    result["metrics"]["test"]["bleu4"]
"""

import json
import math
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CHECKPOINT_DIR, RESULTS_DIR, build_model, full_config, get_data, load_model,
                    load_results, release_device_memory)
from data import caption_loaders, channel_stats, image_datasets
from metrics import bleu4, class_mention_accuracy, distinct_ngrams
from models import Captioner
from text import Vocabulary, tokenize
from train import count_parameters, get_device, set_seed, train

# Encoders from Part 2 (runs with checkpoints for seeds 0, 1, 2).
ENCODER_RUNS = {"cnn": "q2_5/cnn_best_50ep", "mlp": "q2_2/mlp_best"}

CAP_DEFAULTS = {
    # encoder
    "encoder": "cnn",          # "cnn" or "mlp" (architecture of ENCODER_RUNS[encoder])
    "pretrained": True,        # start from the classifier trained in Part 2 (same seed)
    "freeze": True,            # keep the encoder fixed; False = train it jointly with the decoder
    # decoder
    "embed_dim": 256,
    "hidden_dim": 512,
    "dropout": 0.3,
    "concat_image": False,     # also feed the image vector at every step
    # optimization
    "lr": 1e-3,                # Adam, decoder (and encoder unless encoder_lr is set)
    "encoder_lr": None,        # learning rate of a jointly trained encoder (None = lr)
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 20,
    "patience": 4,             # early stopping on validation loss
    "grad_clip": 5.0,
    # text
    "min_count": 5,
    "max_len": 25,             # maximum generated length (99 % of the captions have <= 20 tokens)
}


def cap_config(config):
    unknown = set(config) - set(CAP_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**CAP_DEFAULTS, **config}


@lru_cache(maxsize=4)
def get_vocab(min_count=5):
    return Vocabulary.from_captions(get_data()["train"]["captions"], min_count=min_count)


@lru_cache(maxsize=1)
def norm_stats():
    return channel_stats(get_data()["train"]["images"])


def device_for_captioning():
    device = get_device()
    if device.type == "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return device


def build_captioner(config, seed):
    """The captioner of `config`. The encoder has the architecture of ENCODER_RUNS[encoder]; with
    pretrained=True its weights are the Part 2 classifier trained with the same seed."""
    run = ENCODER_RUNS[config["encoder"]]
    if config["pretrained"]:
        encoder = load_model(run, seed)
    else:
        encoder = build_model(full_config(load_results(run)[0]["config"]))
    vocab = get_vocab(config["min_count"])
    return Captioner(encoder, len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id,
                     embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
                     dropout=config["dropout"], concat_image=config["concat_image"],
                     freeze_encoder=config["freeze"])


def build_optimizer(config, model):
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": decoder, "lr": config["lr"]}]
    if encoder:
        groups.append({"params": encoder, "lr": config["encoder_lr"] or config["lr"]})
    return torch.optim.Adam(groups, weight_decay=config["weight_decay"])


def get_caption_loaders(config):
    data = get_data()
    vocab = get_vocab(config["min_count"])
    mean, std = norm_stats()
    return dict(zip(("train", "validation", "test"),
                    caption_loaders(data, vocab.encode, vocab.pad_id, config["batch_size"],
                                    mean=mean, std=std)))


def image_loader(split, batch_size=100):
    """One item per image, in dataset order (for generating captions)."""
    mean, std = norm_stats()
    return DataLoader(image_datasets(get_data(), mean, std)[split], batch_size=batch_size)


def caption_loss(model, batch, pad_id):
    """Teacher forcing: input captions[:, :-1], targets captions[:, 1:]; mean token-level
    cross-entropy over the non-<pad> targets."""
    images, captions = batch
    logits = model(images, captions[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), captions[:, 1:].reshape(-1),
                           ignore_index=pad_id)


@torch.no_grad()
def dataset_loss(model, loader, pad_id):
    """Mean cross-entropy per real (non-<pad>) target token over a whole loader, in eval mode."""
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)
        logits = model(images, captions[:, :-1])
        targets = captions[:, 1:]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                 ignore_index=pad_id, reduction="sum").item()
        count += (targets != pad_id).sum().item()
    return total / count


@torch.no_grad()
def generate(model, split, vocab, decode=None, max_len=25):
    """Generated captions (lists of words, without special tokens) for every image of `split`.
    `decode(model, images)` returns token ids (B, T); default: greedy decoding."""
    model.eval()
    device = next(model.parameters()).device
    decode = decode or (lambda m, x: m.greedy(x, max_len=max_len))
    captions = []
    for images, _ in image_loader(split):
        for ids in decode(model, images.to(device)).cpu():
            captions.append(vocab.decode(ids))
    return captions


@lru_cache(maxsize=4)
def references(split):
    """The 5 reference captions of every image of `split`, tokenized with our tokenizer."""
    return [[tokenize(c) for c in image_captions] for image_captions in get_data()[split]["captions"]]


def caption_metrics(captions, split):
    """BLEU-4, class mention accuracy, mean length and diversity of generated captions."""
    labels = get_data()[split]["labels"]
    return {"bleu4": bleu4(captions, references(split)),
            "class_mention": class_mention_accuracy(captions, labels),
            "mean_length": sum(map(len, captions)) / len(captions),
            "distinct_1": distinct_ngrams(captions, 1), "distinct_2": distinct_ngrams(captions, 2),
            "unk_fraction": sum(c.count("<unk>") for c in captions) / max(1, sum(map(len, captions)))}


def caption_run(name, config, seed, save_model=True, verbose=False):
    """Train one captioner configuration with one seed and evaluate it. Saves
    results/<name>/seed<seed>.json (incl. the generated val/test captions) and, if save_model,
    checkpoints/<name>/seed<seed>.pt. Returns (result, model)."""
    config = cap_config(config)
    release_device_memory()
    set_seed(seed)
    vocab = get_vocab(config["min_count"])
    loaders = get_caption_loaders(config)
    model = build_captioner(config, seed)
    optimizer = build_optimizer(config, model)
    device = device_for_captioning()
    history = train(model, optimizer, lambda m, b: caption_loss(m, b, vocab.pad_id),
                    loaders["train"],
                    val_score_fn=lambda m: -dataset_loss(m, loaders["validation"], vocab.pad_id),
                    epochs=config["epochs"], patience=config["patience"],
                    grad_clip=config["grad_clip"], device=device, verbose=verbose)
    # `train` has restored the epoch with the lowest validation loss.
    losses = {split: dataset_loss(model, loaders[split], vocab.pad_id)
              for split in ("train", "validation", "test")}
    generated = {split: generate(model, split, vocab, max_len=config["max_len"])
                 for split in ("validation", "test")}
    result = {
        "name": name, "seed": seed, "config": config, "device": str(device),
        "num_parameters": {"total": sum(p.numel() for p in model.parameters()),
                           "trainable": count_parameters(model),
                           "encoder": sum(p.numel() for p in model.encoder.parameters())},
        "history": {"train_loss": history["train_loss"],
                    "val_loss": [-s for s in history["val_score"]],
                    "seconds": history["seconds"], "best_epoch": history["best_epoch"]},
        "loss": losses, "perplexity": {k: math.exp(v) for k, v in losses.items()},
        "metrics": {split: caption_metrics(c, split) for split, c in generated.items()},
        "captions": generated,
    }
    os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name, f"seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if save_model:
        os.makedirs(os.path.join(CHECKPOINT_DIR, name), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, name, f"seed{seed}.pt"))
    return result, model


def cached_caption_run(name, config, seed, save_model=True, verbose=False):
    """caption_run, unless results/<name>/seed<seed>.json exists with the same config (and the
    checkpoint, if save_model)."""
    path = os.path.join(RESULTS_DIR, name, f"seed{seed}.json")
    ckpt = os.path.join(CHECKPOINT_DIR, name, f"seed{seed}.pt")
    if os.path.exists(path) and (not save_model or os.path.exists(ckpt)):
        with open(path) as f:
            result = json.load(f)
        if result["config"] == json.loads(json.dumps(cap_config(config))):
            return result
    return caption_run(name, config, seed, save_model=save_model, verbose=verbose)[0]


def load_captioner(name, seed):
    with open(os.path.join(RESULTS_DIR, name, f"seed{seed}.json")) as f:
        result = json.load(f)
    model = build_captioner(result["config"], seed)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name, f"seed{seed}.pt"),
                                     weights_only=True))
    return model.to(device_for_captioning()).eval()


def caption_line(r):
    m = r["metrics"]
    return (f"val loss {r['loss']['validation']:.3f} (ppl {r['perplexity']['validation']:.1f}) | "
            f"BLEU-4 val {m['validation']['bleu4']:.2f} test {m['test']['bleu4']:.2f} | "
            f"class mention test {100 * m['test']['class_mention']:.1f}% | "
            f"len {m['test']['mean_length']:.1f} | best epoch {r['history']['best_epoch']}")
