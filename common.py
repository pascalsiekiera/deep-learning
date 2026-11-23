"""Shared helpers for Part 2: data loading, building models and optimizers from a config dict,
running one configuration over several seeds, and saving every run as JSON in results/.

A config is a plain dict; any key left out takes its value from DEFAULTS. Example:

    from common import run_seeds
    results = run_seeds("q2_2/my_mlp", {"hidden": (512,), "optimizer": "adam", "lr": 1e-3})
    print(results["test_acc"])        # {"mean": ..., "std": ..., "values": [...]}
"""

import json
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ImageDataset, channel_stats
from load_data import SPLITS, load_coco_dominant10
from models import MLP
from train import count_parameters, get_device, set_seed, train

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")   # not submitted (see .gitignore)
SEEDS = (0, 1, 2)

DEFAULTS = {
    # model
    "model": "mlp",
    "hidden": (),             # () = linear baseline
    "activation": "relu",
    "dropout": 0.0,
    # optimization
    "optimizer": "sgd",       # "sgd" or "adam"
    "lr": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "scheduler": "none",      # "none" or "cosine"
    "batch_size": 64,
    "epochs": 30,
    "patience": None,         # None = always train for `epochs` epochs
    # data
    "normalize": True,        # per-channel standardization with training-set statistics
    "flip": False,            # random horizontal flips on the training set
    "permutation_seed": None, # if set, apply one fixed random pixel permutation to all splits
}


# ----------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def get_data():
    """The dataset (cached in memory after the first call)."""
    return load_coco_dominant10()


def pixel_permutation(seed, height=64, width=64):
    """A fixed random permutation of the height * width pixel positions (Q2.4 (ii))."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(height * width, generator=generator)


def permute_pixels(images, perm):
    """Apply the same permutation of pixel positions to every image and every channel.
    images: (N, C, H, W) -> (N, C, H, W)."""
    n, c, h, w = images.shape
    return images.reshape(n, c, h * w)[:, :, perm].reshape(n, c, h, w)


def shift_images(images, dx=0, dy=0, fill=0.0):
    """Translate a batch (N, C, H, W) by dx pixels to the right and dy pixels down (negative =
    left / up), without wrap-around: the uncovered border is set to `fill`. Applied to
    normalized images, fill=0 is the training-set mean colour (Q2.4 (i), Q2.5)."""
    n, c, h, w = images.shape
    out = torch.full_like(images, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[:, :, yd, xd] = images[:, :, ys, xs]
    return out


def mask_border(images, dx=0, dy=0, fill=0.0):
    """Control for shift_images: blank exactly the border that shift_images(images, dx, dy)
    would uncover, but leave the rest of the image where it is (same information loss, no
    misalignment)."""
    out = images.clone()
    if dx > 0:
        out[..., :, :dx] = fill
    elif dx < 0:
        out[..., :, dx:] = fill
    if dy > 0:
        out[..., :dy, :] = fill
    elif dy < 0:
        out[..., dy:, :] = fill
    return out


def get_datasets(config):
    """{split: ImageDataset}, plus an un-augmented copy of the training set ("train_eval") used
    to measure training accuracy."""
    data = get_data()
    images = {split: data[split]["images"] for split in SPLITS}
    if config["permutation_seed"] is not None:
        perm = pixel_permutation(config["permutation_seed"])
        images = {split: permute_pixels(x, perm) for split, x in images.items()}
    # Per-channel statistics do not change under a permutation of pixel positions.
    mean, std = channel_stats(images["train"]) if config["normalize"] else (None, None)
    datasets = {split: ImageDataset(images[split], data[split]["labels"], mean, std,
                                    flip=config["flip"] and split == "train")
                for split in SPLITS}
    datasets["train_eval"] = ImageDataset(images["train"], data["train"]["labels"], mean, std)
    return datasets


def get_loaders(config):
    """{split: DataLoader}; only "train" is shuffled (and augmented, if config["flip"])."""
    return {split: DataLoader(ds, batch_size=config["batch_size"] if split == "train" else 500,
                              shuffle=split == "train")
            for split, ds in get_datasets(config).items()}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, loader, transform=None):
    """Predicted classes and gold labels over a loader, as two long tensors (N,).
    `transform` (optional) is applied to each batch of images before the model, e.g. a shift."""
    model.eval()
    device = next(model.parameters()).device
    preds, labels = [], []
    for x, y in loader:
        if transform is not None:
            x = transform(x)
        preds.append(model(x.to(device)).argmax(-1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def accuracy_on(model, loader, transform=None):
    preds, labels = predict(model, loader, transform)
    return (preds == labels).float().mean().item()


def mean_std(values):
    """{"mean", "std", "values"} of a list of floats (sample std, computed with torch)."""
    t = torch.tensor(values, dtype=torch.float64)
    std = t.std().item() if len(values) > 1 else 0.0
    return {"mean": t.mean().item(), "std": std, "values": list(values)}


# ----------------------------------------------------------------------------- building blocks
def full_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return {**DEFAULTS, **config}


def build_model(config):
    if config["model"] == "mlp":
        return MLP(hidden=tuple(config["hidden"]), activation=config["activation"],
                   dropout=config["dropout"])
    raise ValueError(f"unknown model {config['model']!r}")


def build_optimizer(config, model):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                               weight_decay=config["weight_decay"])
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"],
                                weight_decay=config["weight_decay"])
    raise ValueError(f"unknown optimizer {config['optimizer']!r}")


def build_scheduler(config, optimizer):
    if config["scheduler"] == "none":
        return None
    if config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    raise ValueError(f"unknown scheduler {config['scheduler']!r}")


def classification_loss(model, batch):
    images, labels = batch
    return F.cross_entropy(model(images), labels)


# ----------------------------------------------------------------------------- experiments
def run(name, config, seed, save_model=False, verbose=False):
    """Train one configuration with one seed. Saves results/<name>/seed<seed>.json and, if
    save_model, checkpoints/<name>/seed<seed>.pt. Returns (result dict, trained model)."""
    config = full_config(config)
    set_seed(seed)
    loaders = get_loaders(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    history = train(model, optimizer, classification_loss, loaders["train"],
                    val_score_fn=lambda m: accuracy_on(m, loaders["validation"]),
                    epochs=config["epochs"], scheduler=build_scheduler(config, optimizer),
                    patience=config["patience"], device=get_device(), verbose=verbose)
    # `train` has restored the epoch with the best validation accuracy.
    result = {
        "name": name, "seed": seed, "config": config,
        "num_parameters": count_parameters(model),
        "history": history,
        "train_acc": accuracy_on(model, loaders["train_eval"]),
        "val_acc": accuracy_on(model, loaders["validation"]),
        "test_acc": accuracy_on(model, loaders["test"]),
    }
    os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name, f"seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if save_model:
        os.makedirs(os.path.join(CHECKPOINT_DIR, name), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, name, f"seed{seed}.pt"))
    return result, model


def run_seeds(name, config, seeds=SEEDS, save_model=False, verbose=False):
    """Run one configuration over several seeds; returns mean/std of train/val/test accuracy
    and the per-seed results."""
    runs = []
    for seed in seeds:
        result, _ = run(name, config, seed, save_model=save_model, verbose=verbose)
        print(f"{name} seed {seed}: train {result['train_acc']:.4f} | val {result['val_acc']:.4f} "
              f"| test {result['test_acc']:.4f} | best epoch {result['history']['best_epoch']}")
        runs.append(result)
    summary = {key: mean_std([r[key] for r in runs]) for key in ("train_acc", "val_acc", "test_acc")}
    summary["runs"] = runs
    return summary


def load_results(name):
    """All saved runs of an experiment, sorted by seed."""
    folder = os.path.join(RESULTS_DIR, name)
    runs = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                runs.append(json.load(f))
    return sorted(runs, key=lambda r: r["seed"])


def load_model(name, seed):
    """Rebuild a model from its saved config and checkpoint."""
    with open(os.path.join(RESULTS_DIR, name, f"seed{seed}.json")) as f:
        config = json.load(f)["config"]
    model = build_model(config)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name, f"seed{seed}.pt"),
                                     weights_only=True))
    return model.eval()


def fmt(stats, scale=100):
    """'mean ± std' in percent."""
    return f"{scale * stats['mean']:.2f} ± {scale * stats['std']:.2f}"


# ----------------------------------------------------------------------------- plots
def plot_learning_curves(runs, path, title=None):
    """Training loss and validation accuracy per epoch, one line per seed (with early stopping
    the runs can have different lengths); a dot marks the restored best epoch."""
    import matplotlib.pyplot as plt   # only needed for plots

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for r in runs:
        h = r["history"]
        epochs = list(range(1, len(h["train_loss"]) + 1))
        best = h["best_epoch"]
        line, = axes[0].plot(epochs, h["train_loss"], label=f"seed {r['seed']}")
        axes[0].plot(best, h["train_loss"][best - 1], "o", color=line.get_color())
        val = [100 * v for v in h["val_score"]]
        axes[1].plot(epochs, val, color=line.get_color())
        axes[1].plot(best, val[best - 1], "o", color=line.get_color())
    for ax, label in zip(axes, ("training loss", "validation accuracy (%)")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
