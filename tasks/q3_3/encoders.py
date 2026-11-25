"""Q3.3 (a): MLP vs CNN encoder, and frozen pretrained vs jointly trained encoder.

Six captioners, each with 3 seeds, all with the decoder configuration selected in Q3.2 and the
same budget (at most 20 epochs, early stopping on validation loss with patience 4):

    encoder  x  {frozen:    pretrained on classification (Part 2, same seed), not updated,
                 finetuned: pretrained, then trained jointly with the decoder (encoder lr 1e-4),
                 scratch:   randomly initialized, trained jointly with the decoder (lr 1e-3)}

The frozen-CNN variant is the captioner of Q3.2 (its runs are copied, not retrained).
The class mention analysis (Q3.3 b) uses these runs, see tasks/q3_3/class_mention.py.

    python -m tasks.q3_3.encoders      # about 5 hours on the laptop (MPS); runs are cached

Writes results/q3_3/<variant>/, checkpoints/q3_3/<variant>/, results/q3_3/encoders.json.
"""

import json
import os

from captioning import cached_caption_run, caption_line
from common import RESULTS_DIR, SEEDS, load_results, mean_std
from tasks.q3_2.captioner import FINAL as Q32_FINAL
from tasks.q3_2.captioner import copy_run

OUT = os.path.join(RESULTS_DIR, "q3_3")
DECODER = {"concat_image": True}           # selected in Q3.2 (other settings: CAP_DEFAULTS)
MODES = {
    "frozen": {"pretrained": True, "freeze": True},
    "finetuned": {"pretrained": True, "freeze": False, "encoder_lr": 1e-4},
    "scratch": {"pretrained": False, "freeze": False},
}
ENCODERS = ["cnn", "mlp"]
# Classification test accuracy of the pretrained encoders (Part 2), per seed.
CLASSIFIER_RUNS = {"cnn": "q2_5/cnn_best_50ep", "mlp": "q2_2/mlp_best"}
# Cheapest first, so that partial results arrive early.
ORDER = [("mlp", "frozen"), ("cnn", "frozen"), ("cnn", "finetuned"), ("mlp", "finetuned"),
         ("cnn", "scratch"), ("mlp", "scratch")]


def variant_name(encoder, mode):
    return f"q3_3/{encoder}_{mode}"


def config_of(encoder, mode):
    return {"encoder": encoder, **MODES[mode], **DECODER}


def stats(runs):
    s = {"num_parameters": runs[0]["num_parameters"],
         "best_epochs": [r["history"]["best_epoch"] for r in runs],
         "seconds_per_epoch": sum(sum(r["history"]["seconds"]) / len(r["history"]["seconds"])
                                  for r in runs) / len(runs)}
    for split in ("validation", "test"):
        for k in ("bleu4", "class_mention", "mean_length", "distinct_1", "distinct_2"):
            s[f"{split}_{k}"] = mean_std([r["metrics"][split][k] for r in runs])
        s[f"{split}_loss"] = mean_std([r["loss"][split] for r in runs])
        s[f"{split}_ppl"] = mean_std([r["perplexity"][split] for r in runs])
    s["train_loss"] = mean_std([r["loss"]["train"] for r in runs])
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    for seed in SEEDS:                       # frozen CNN = the Q3.2 captioner
        copy_run(Q32_FINAL, variant_name("cnn", "frozen"), seed)

    results = {}
    for encoder, mode in ORDER:
        name = variant_name(encoder, mode)
        runs = []
        for seed in SEEDS:
            r = cached_caption_run(name, config_of(encoder, mode), seed)
            print(f"[{encoder} {mode:9s} seed {seed}] {caption_line(r)}", flush=True)
            runs.append(r)
        results[f"{encoder}_{mode}"] = stats(runs)

    classifier = {e: mean_std([r["test_acc"] for r in load_results(run)])
                  for e, run in CLASSIFIER_RUNS.items()}
    print("\n| encoder | mode | params total | trainable | val BLEU-4 | test BLEU-4 | test class mention "
          "| val ppl | train loss | mean len | best epochs |")
    for encoder in ENCODERS:
        for mode in MODES:
            s = results[f"{encoder}_{mode}"]
            p = s["num_parameters"]
            f = lambda k, scale=1: f"{scale * s[k]['mean']:.2f} ± {scale * s[k]['std']:.2f}"
            print(f"| {encoder} | {mode} | {p['total']:,} | {p['trainable']:,} | {f('validation_bleu4')} | "
                  f"{f('test_bleu4')} | {f('test_class_mention', 100)} | {f('validation_ppl')} | "
                  f"{f('train_loss')} | {s['test_mean_length']['mean']:.2f} | {s['best_epochs']} |")
    print("classification test accuracy of the pretrained encoders:",
          {e: f"{100 * c['mean']:.2f} ± {100 * c['std']:.2f}" for e, c in classifier.items()})

    with open(os.path.join(OUT, "encoders.json"), "w") as f:
        json.dump({"decoder": DECODER, "modes": MODES, "results": results,
                   "classifier_test_acc": classifier}, f, indent=1)


if __name__ == "__main__":
    main()
