# Deep Learning 2026-27: Course Project

This repository contains our code for the course project. The first section describes our code
and how to reproduce every table and figure of the report; the second section is the original
description of the provided skeleton code.

## Our code (Parts 2–4: image classification, image captioning, Transformer captioner)

### Setup

Python 3.10 or newer, then, from the repository root:

```
pip install -r requirements.txt
python load_data.py          # downloads the dataset (about 80 MB) into data/
```

### Reproducing the results

Run the scripts from the repository root, in this order (later questions read the results of
earlier ones):

| Question | Command | What it does | Approx. time* |
|---|---|---|---|
| Q2.1 | `python -m tasks.q2_1.baseline` | data statistics, linear baseline (3 seeds) | 1 min |
| Q2.1 | `python -m tasks.q2_1.lr_sweep` | learning-rate sweep of the linear baseline on validation | 3 min |
| Q2.2 | `python -m tasks.q2_2.mlp_search` | MLP random search (30 configs), top-3 re-run with 3 seeds, selected MLP | 1–2 h |
| Q2.2 | `python -m tasks.q2_2.activations` | extra: activation ablation (ReLU, GELU, tanh, ELU, PReLU) | 15 min |
| Q2.3 | `python -m tasks.q2_3.study` | one-factor-at-a-time study (7 factors, 45 runs), per-class accuracy, confusion matrix | 1.5 h |
| Q2.4 | `python -m tasks.q2_4.inductive_biases` | weight visualizations, shifted test images, pixel permutation, MLP confusion matrix | 5 min |
| Q2.5 | `python -m tasks.q2_5.cnn` | CNN random search (16 configs), top-3 re-run, selected CNN, shift and permutation experiments, confusion matrices | 1.5–2 h |
| Q3.1 | `python -m tasks.q3_1.text_stats` | tokenizer and vocabulary statistics, `<unk>` rate, caption lengths, padding check | 1 min |
| Q3.2 | `python -m tasks.q3_2.captioner` | GRU captioner on the frozen CNN: decoder search (5 trials), selected captioner with 3 seeds, loss curves, example captions | 1.5 h |
| Q3.3 | `python -m tasks.q3_3.encoders` | MLP vs CNN encoder, each frozen / fine-tuned / trained from scratch (3 seeds each) | 5 h |
| Q3.3 | `python -m tasks.q3_3.class_mention` | class mention vs classification accuracy (linear probes), correlations, per-class plot | 5 min |
| Q3.4 | `python -m tasks.q3_4.decoding` | greedy vs beam search (beam 1, 3, 5, 10; with and without length normalization) | 5 min |
| Q4.1 | `python -m tasks.q4_1.transformer` | Transformer captioner: search (5 trials), 3 seeds, comparison with the RNN (greedy and beam 5, timing), cross-attention maps | 1.5 h |

\* On an Apple M2 laptop (8 GB RAM); MLPs on the CPU, CNNs and all captioners on the Apple GPU (MPS). With a CUDA
GPU everything uses the GPU.

- Every run is saved as JSON in `results/qX_Y/<run>/seed<k>.json` (config, per-epoch history,
  train/val/test accuracy, number of parameters; for captioners also losses, perplexities, caption
  metrics and the generated validation/test captions). A script skips runs whose JSON already exists
  with the same config, so an interrupted script can simply be restarted.
- Each script also writes a summary JSON and a log to `results/qX_Y/`, and its figures to
  `figures/qX_Y/`.
- Model checkpoints are written to `checkpoints/qX_Y/` and are **not** part of the submission
  (`.gitignore`); they are recreated by running the scripts in the order above.
- Seeds 0, 1 and 2 are used everywhere; with `train.set_seed` the runs are deterministic, so
  re-running a script reproduces the same numbers. Exception: captioners whose encoder is trained
  jointly on the Apple GPU (Q3.3 fine-tuned / scratch) are not bit-reproducible there and can differ
  slightly between re-runs.

### Layout

| Path | Content |
|---|---|
| `models.py` | `MLP` (configurable depth, width, activation, dropout; no hidden layer = linear baseline) and `CNN` (VGG-style blocks of 3×3 conv, BatchNorm, ReLU, max pooling; global average pooling), both with `features()` for the image vector; `Captioner` (encoder + GRU decoder, image vector as initial state and optionally at every step) and `TransformerCaptioner` (own pre-LN decoder blocks with causal self-attention and cross-attention to the 8×8 CNN feature grid); greedy decoding and beam search shared by both (`CaptionDecoding`) |
| `text.py` | tokenizer (lowercase, words and punctuation as tokens) and `Vocabulary` (frequency threshold, `<pad>`, `<bos>`, `<eos>`, `<unk>`) |
| `captioning.py` | Part 3 helpers: building a captioner from a config dict (encoder from Part 2, frozen or trained jointly), teacher-forcing loss with `ignore_index=<pad>`, perplexity, caption generation, caption metrics, `caption_run` (train and evaluate one config with one seed and save the result) |
| `common.py` | shared helpers: building models/optimizers/schedulers from a config dict, data loaders with per-channel normalization, `run` (train one config with one seed and save the result), evaluation, image shifts and border controls, pixel permutation, confusion matrices, plots |
| `tasks/qX_Y/` | one folder per question with the scripts above |
| `results/qX_Y/` | all run results (JSON), summaries and logs |
| `figures/qX_Y/` | all figures used in the report |

### Changes to the skeleton code

None: `load_data.py`, `data.py`, `train.py` and `metrics.py` are unchanged. Our code only imports
them (`channel_stats`, `ImageDataset`, `image_datasets`, `caption_loaders`, `train`, `set_seed`,
`count_parameters`, `get_device`, `confusion_matrix`, `per_class_accuracy`, `bleu4`,
`class_mention_accuracy`, `distinct_ngrams`, …). Choosing the Apple GPU (MPS) for CNNs and
captioners is done in `common.run_device` and `captioning.device_for_captioning`, outside
`train.py`. The tokenizer, vocabulary, captioner, and greedy and beam search decoding are our own
code (`text.py`, `models.py`, `captioning.py`). Our extended class synonym list for Q3.3 (b) is
defined in `tasks/q3_3/class_mention.py`, not in `metrics.py`.
No `numpy` is used.

## Provided skeleton code

This folder contains the code we provide for Parts 2 and 3 of the project: data loading, a
generic training loop, and the evaluation metrics. **Everything else is up to you**: the models
(MLPs, CNNs, RNN decoders), the tokenizer and vocabulary, the loss functions, greedy and beam
search decoding, the evaluation code that calls the metrics, the experiments, and the plots.
You may modify these files, but say what you changed in your report.

### Setup

Python 3.10 or newer, then:

```
pip install -r requirements.txt
python load_data.py
```

The second command downloads the dataset (about 80 MB) into `data/` and prints the size of each
split. On Google Colab, upload this folder (or clone it) and run the same commands in a cell,
with a `!` in front. Part 2 runs fine on a laptop CPU; for Part 3 we recommend a GPU
(e.g., the free Colab GPU: *Runtime > Change runtime type*).

### Files

| File | What it gives you |
|---|---|
| `load_data.py` | `load_coco_dominant10()`: downloads and caches the dataset; returns, for each split, uint8 images `(N, 3, 64, 64)`, labels `(N,)`, 5 caption strings per image, and license metadata. `CLASSES` holds the class names. |
| `data.py` | `channel_stats`, `ImageDataset`, and `image_loaders` for classification; `CaptionDataset`, `PadCollate`, and `caption_loaders` for captioning, which take *your* `encode` function (caption string to token ids) and the id of your padding token. |
| `train.py` | `train(...)`: a training loop with optional learning-rate scheduler, gradient clipping, early stopping, and restoring the best epoch on the validation set; you pass it your `loss_fn` and `val_score_fn`. Also `set_seed`, `count_parameters`, `get_device`. |
| `metrics.py` | `accuracy`, `confusion_matrix`, `per_class_accuracy`; `bleu4`, `class_mention_accuracy` (with the synonym list `CLASS_SYNONYMS`), `distinct_ngrams`. |

### Example

```python
import torch
import torch.nn.functional as F
from load_data import load_coco_dominant10
from data import channel_stats, image_loaders
from metrics import accuracy
from train import set_seed, train

data = load_coco_dominant10()
mean, std = channel_stats(data["train"]["images"])
train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)

@torch.no_grad()
def predict(model, loader):
    device = next(model.parameters()).device
    return torch.cat([model(x.to(device)).argmax(-1).cpu() for x, _ in loader])

set_seed(0)
model = ...   # your model: images (B, 3, 64, 64) -> class scores (B, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
history = train(model, optimizer,
                loss_fn=lambda model, batch: F.cross_entropy(model(batch[0]), batch[1]),
                train_loader=train_loader,
                val_score_fn=lambda model: accuracy(predict(model, val_loader), data["validation"]["labels"]),
                epochs=20)
```

For captioning, `caption_loaders(data, encode, pad_id, batch_size, mean=mean, std=std)` yields
`(images, captions)` mini-batches, with captions padded to the longest one in the mini-batch.
To generate captions, iterate over `image_loaders` (one item per image, in dataset order) and
compare them with `data[split]["captions"]`, tokenized with your tokenizer.

### Rules (see the project statement)

Allowed: `torch`, `torchvision.transforms`, `matplotlib`, `tqdm`, and this code. Not allowed in
Parts 2 and 3: `numpy`, `scikit-learn`, high-level training frameworks, `torchvision.models`,
`timm`, Hugging Face `transformers`, pretrained weights, and pretrained tokenizers. `pyarrow` and
`Pillow` are used only inside `load_data.py`, to read the data files.
