# Report Notes

Bullet points for the report, extended after each finished question. All numbers are accuracies in %, given as mean ± std (sample std, n = 3) over seeds 0, 1 and 2, unless stated otherwise.

## General (method, code)

- **Seeds:** 0, 1, 2. With `set_seed(seed)` from `train.py` the runs are deterministic: rerunning the same config with the same seed gives exactly the same numbers.
- **Model selection:** on validation only. Test is read only for the final numbers. At the end, `train()` restores the epoch with the best val accuracy.
- **Loss:** cross-entropy (`F.cross_entropy` on the logits, averaged over the batch). Rationale: it is the negative log-likelihood of the softmax model, i.e. maximum likelihood. The gradient w.r.t. the logits is `softmax(z) − onehot(y)`. In the linear case the loss is convex. MSE on the softmax output would have a vanishing gradient for confident wrong predictions.
- **Normalization:** per channel, with statistics from the training set only (`channel_stats`), applied unchanged to validation and test.
- **Skeleton code:** `data.py`, `train.py`, `metrics.py` and `load_data.py` are **unchanged**. Our own code is in `models.py`, `common.py` and `tasks/q2_x/*.py` (one folder per question) (incl. the extra experiments `tasks/q2_1/lr_sweep.py` and `tasks/q2_2/activations.py`), `tasks/q2_3/study.py` and `tasks/q2_4/inductive_biases.py`. Added to `common.py` for 2.4: `shift_images`, `mask_border`.
- **No numpy:** all statistics (mean, std, class counts) are computed with torch.
- **Reproduce:** from the project folder run `python -m tasks.q2_1.baseline`, `python -m tasks.q2_2.mlp_search`, etc. Everything is organized per question: scripts in `tasks/q2_x/`, results in `results/q2_x/`, figures in `figures/q2_x/`, checkpoints in `checkpoints/q2_x/`. Each run is saved as JSON in `results/q2_x/<run>/seed<k>.json` (config, per-epoch history, accuracies, parameter count).

## Q2.1 Data and linear baseline

**Data** (source: `results/q2_1/data_stats.json`)

| Split | Images | per class |
|---|---|---|
| Train | 6500 | 650 |
| Validation | 1000 | 100 |
| Test | 1000 | 100 |

- All 10 classes are exactly balanced in every split. Chance level is 10 %, and accuracy needs no class weighting.
- Channel statistics on train (pixels in [0, 1]):
  - Mean R, G, B = 0.4634, 0.4530, 0.4173
  - Std R, G, B = 0.2585, 0.2571, 0.2713

**Model and training**
- `nn.Flatten` → `nn.Linear(12288, 10)`, i.e. multinomial logistic regression.
- 122,890 parameters: 12,288 × 10 weights + 10 biases.
- Plain SGD (no momentum), **learning rate 0.0003** (selected on validation, see the sweep below), batch size 64, 50 epochs, no early stopping (so the full curve is visible). Afterwards the best val epoch is restored.

**Learning-rate sweep** (`python -m tasks.q2_1.lr_sweep`, summary in `results/q2_1/lr_sweep.json`)
- Why: the MLP of 2.2 is tuned on validation, so the baseline's learning rate is tuned too. Otherwise part of the MLP's advantage could just come from tuning only one of the two models.
- Setup: the 2.1 config above, only the lr changes, 3 seeds each, same budget of 50 epochs. Selection by mean val accuracy (same rule as in 2.2). The grid was first 0.0003–0.03; since the best value was at the lower edge, it was extended to 0.00003 and 0.0001 so that the optimum lies inside the grid.

| lr | Train | Val | Test | best epochs | final train loss |
|---|---|---|---|---|---|
| 0.00003 | 31.77 ± 0.65 | 30.13 ± 0.38 | 25.57 ± 1.00 | 50, 49, 50 | 1.952 |
| 0.0001 | 37.02 ± 2.34 | 31.67 ± 0.55 | 26.63 ± 0.85 | 22, 38, 48 | 1.803 |
| **0.0003** (selected) | 42.14 ± 3.49 | **32.20 ± 0.50** | 26.83 ± 0.83 | 30, 14, 28 | 1.591 |
| 0.001 (first guess) | 42.83 ± 2.29 | 31.57 ± 0.75 | 26.70 ± 0.53 | 10, 7, 7 | 1.237 |
| 0.003 | 47.64 ± 6.71 | 29.33 ± 0.42 | 25.50 ± 0.40 | 4, 11, 6 | 0.826 |
| 0.01 | 50.46 ± 5.43 | 25.97 ± 0.49 | 22.30 ± 1.90 | 4, 8, 13 | 0.425 |
| 0.03 | 46.41 ± 5.39 | 24.87 ± 1.42 | 22.47 ± 1.04 | 4, 11, 9 | 0.488 |

- Between 0.0001 and 0.001 the val accuracy is flat (31.6–32.2, differences within about one std, and below the ~1.5-point standard error of a 1000-image val set). The choice of 0.0003 follows the rule but is not a meaningful improvement over 0.001: test changes only from 26.70 to 26.83.
- With lr 0.00003 the best epoch is the last one (50, 49, 50): the model is still underfitting and the 50-epoch budget is the limit, not overfitting.
- With lr ≥ 0.003 the train loss gets much lower but val drops by up to 7 points: large steps reach the overfitted region within a few epochs (best epoch 4–13), and with lr 0.03 the final train loss is even higher than with 0.01 (0.488 vs 0.425), a sign of unstable steps.
- Early stopping mostly undoes the effect of the lr: the restored epoch has about the same train loss in each case (about 1.7–1.85 for lr 0.0003 and 0.001), so smaller lrs reach the same point later. This is why the val accuracy is so flat across a 10× range.
- Conclusion for the comparison in 2.2: the MLP's advantage does **not** come from tuning only the MLP. With the tuned baseline it is +4.3 points on val and +2.8 on test.

**Results** (lr 0.0003)

| Seed | Train | Val | Test | best epoch |
|---|---|---|---|---|
| 0 | 44.49 | 31.70 | 27.10 | 30 |
| 1 | 38.12 | 32.70 | 27.50 | 14 |
| 2 | 43.80 | 32.20 | 25.90 | 28 |
| **Mean** | **42.14 ± 3.49** | **32.20 ± 0.50** | **26.83 ± 0.83** | |

**Observations**
- The best epoch is in the middle of the run (14–30 of 50). After that the model overfits: train loss keeps falling (1.59 at epoch 50), val accuracy drops by 1–3 points (final val 29.3–30.5). Plot: `figures/q2_1/curves.pdf` (mean ± std over seeds).
- Test is about 5 points below val. Part of this is expected, because the best epoch (and now also the lr) is selected on val, which makes the val value optimistic. The gap is still large, so it deserves a sentence.
- The first-guess runs with lr 0.001 are kept in `results/q2_1/lr_sweep/0.001/` (identical to the earlier numbers 26.70 ± 0.53 on test, since the runs are deterministic).
- The seed 0 weights are in `checkpoints/q2_1/linear/seed0.pt` and are needed for the class templates in 2.4.

## Q2.2 MLP and hyperparameter search

**Search protocol**
1. **Random search:** 30 configurations, sampling seed 0, each trained once with seed 0. At most 50 epochs, early stopping on val accuracy with patience 8.
2. **Re-check:** the top 3 of the search are run with all 3 seeds. The choice is made by **mean** val accuracy. Rationale: with 1000 val images the standard error of the accuracy is about 1.5 points, so a single seed is too noisy.
3. **Report:** the chosen configuration with 3 seeds; test is read only here.

**Search space**

| Hyperparameter | Values |
|---|---|
| Depth | 1, 2, 3 (all layers the same width) |
| Width | 256, 512, 1024 |
| Activation | ReLU, GELU, tanh |
| Dropout | 0, 0.2, 0.5 |
| Optimizer | Adam (learning rate log-uniform in [1e-4, 3e-3]) or SGD with momentum 0.9 (learning rate log-uniform in [3e-3, 1e-1]) |
| Weight decay | 0, 1e-4, 1e-3 |
| Batch size | 64, 128 |
| Scheduler | none or cosine (T_max = 50) |

**Re-check of the top 3** (val, 3 seeds)

| Search rank | Configuration | Val (1 seed) | Val (3 seeds) | Train (3 seeds) |
|---|---|---|---|---|
| 1 | [512, 512] GELU, dropout 0.2, SGD lr 0.0056, cosine | 37.40 | 36.10 ± 1.18 | 58.29 ± 17.50 |
| 2 | [1024, 1024] ReLU, dropout 0.5, SGD lr 0.013, no scheduler | 37.20 | **36.50 ± 0.61** | 53.49 ± 8.41 |
| 3 | [1024, 1024, 1024] GELU, dropout 0.5, Adam lr 0.00023, cosine | 35.30 | 35.80 ± 0.46 | 48.16 ± 12.78 |

All three use weight decay 1e-4 and batch size 128.

**Chosen configuration**
- MLP with 2 hidden layers of 1024 units each, ReLU, dropout 0.5
- SGD with momentum 0.9, learning rate 0.013, weight decay 1e-4, batch size 128, no scheduler
- At most 50 epochs, patience 8
- 13,643,786 parameters

**Results**

| Seed | Train | Val | Test | best epoch |
|---|---|---|---|---|
| 0 | 62.31 | 37.20 | 30.90 | 12 |
| 1 | 45.57 | 36.20 | 29.10 | 6 |
| 2 | 52.58 | 36.10 | 29.00 | 8 |
| **Mean** | **53.49 ± 8.41** | **36.50 ± 0.61** | **29.67 ± 1.07** | |

**Comparison with the baseline**

| Model | Parameters | Train | Val | Test |
|---|---|---|---|---|
| Linear (2.1, lr tuned) | 122,890 | 42.14 ± 3.49 | 32.20 ± 0.50 | 26.83 ± 0.83 |
| MLP (2.2) | 13,643,786 | 53.49 ± 8.41 | 36.50 ± 0.61 | 29.67 ± 1.07 |

**Observations**
- The MLP is about 2.8 points better on test and 4.3 on val than the lr-tuned baseline, but has more than 100 times as many parameters. The gain is small relative to the model size. Both models are now tuned on val, so the comparison is fair (see the lr sweep in 2.1); against the untuned first guess (lr 0.001) the gap would be about the same (+3.0 test, +4.9 val).
- The large gap between train and val shows strong overfitting. The model sees only 6500 images with 12,288 inputs each. This motivates 2.3 (regularization) and 2.4 (inductive biases).
- The best single configuration of the search (37.4 % with 1 seed) lost with 3 seeds (36.10 ± 1.18). This shows why the multi-seed re-check is needed.
- Train accuracy varies a lot (± 8.4) because early stopping triggers at epoch 6, 8 or 12 depending on the seed. Val accuracy is stable (± 0.6).
- In the top 10 of the search almost all models are large (512–1024 wide) and use weight decay 1e-4. SGD with momentum appears more often than Adam.
- Val and test are about 7 points apart for the MLP too, as for the baseline.

**Activation ablation (ELU, PReLU)**
- Goal: also test ELU and PReLU without re-running the random search (adding them to the search space would change all 30 sampled configurations).
- Setup: the chosen configuration is kept fixed, only the activation is swapped, 3 seeds each. ReLU is the chosen model itself (`results/q2_2/mlp_best/`).
- PReLU = torch default `nn.PReLU()`: one learnable negative slope per hidden layer (init 0.25), i.e. only +2 parameters. Weight decay also acts on the slope (pulls it toward 0, i.e. ReLU).

| Activation | Train | Val | Test | best epochs |
|---|---|---|---|---|
| **ReLU** (chosen) | 53.49 ± 8.41 | **36.50 ± 0.61** | **29.67 ± 1.07** | 12, 6, 8 |
| PReLU | 57.63 ± 14.69 | 35.90 ± 0.95 | 29.13 ± 0.42 | 10, 13, 4 |
| GELU | 48.75 ± 10.17 | 34.13 ± 0.50 | 28.57 ± 0.57 | 10, 4, 4 |
| ELU | 53.61 ± 5.90 | 33.97 ± 0.93 | 28.73 ± 0.99 | 10, 10, 7 |
| tanh | 61.65 ± 11.03 | 33.50 ± 0.10 | 27.10 ± 0.66 | 10, 7, 14 |

- ReLU stays the best on val, so the selected model is unchanged. PReLU is within one std of ReLU (0.6 points on val, about the noise level); ELU is about 2.5 points worse, similar to GELU.
- Caveat for the report: the other hyperparameters (lr, dropout, weight decay) were selected together with ReLU, so this comparison is biased toward ReLU. A fair claim is "with this tuned setup, swapping the activation does not help", not "ReLU is the best activation".
- The ranking on test is the same as on val except that ELU and GELU swap places (differences there are below the noise level).
- Script: `python -m tasks.q2_2.activations`; results in `results/q2_2/activations/<activation>/`, summary in `results/q2_2/activations.json`, log in `results/q2_2/activations_log.txt`. No checkpoints are saved for these runs.

**Files**
- Learning curves of the 3 seeds: `figures/q2_2/curves.pdf` (a dot marks the best epoch of each)
- Full search with ranking: `results/q2_2/summary.json`
- Individual runs: `results/q2_2/search/`, `results/q2_2/top/`, `results/q2_2/mlp_best/`
- Checkpoints for 2.3–2.5: `checkpoints/q2_2/mlp_best/seed{0,1,2}.pt`

## Q2.3 One-factor-at-a-time study

**Protocol** (`python -m tasks.q2_3.study`, summary in `results/q2_3/summary.json`, log in `results/q2_3/log.txt`)
- Base = the MLP selected in 2.2: [1024, 1024] ReLU, dropout 0.5, SGD lr 0.013 with momentum 0.9, weight decay 1e-4, batch size 128, no scheduler, normalized inputs, no augmentation.
- One factor changed at a time, everything else fixed. 7 factors (at least 3 required).
- **Same budget for every run: 50 epochs, no early stopping.** The best val epoch is restored at the end (as everywhere). The base run with this budget gives exactly the same numbers as `mlp_best` in 2.2 (all best epochs ≤ 12, so patience 8 never cut off a later improvement).
- 3 seeds per setting, 45 runs in total. Batch size was not varied: with a fixed number of epochs it changes the number of updates, which would break the "same budget" rule.
- Columns: **train** = train accuracy of the restored epoch (so it depends on where the best epoch is; this explains the large stds); **val** = best val accuracy within 50 epochs; **final val** = val accuracy after epoch 50 (best val − final val shows how much the model overfits when trained to the end); **final train loss** = mean train loss in epoch 50.
- **Meaningful difference rule:** Δval vs base, with standard error SE = sqrt(s_a²/3 + s_b²/3); marked * if |Δ| > 2·SE. With n = 3 this is a rough rule, not a proper test.
- Test is not used in the study; it is read only for the best model (per-class accuracy and confusion matrix below).
- Plots: `figures/q2_3/curves.pdf` (per factor: train loss (log) and val accuracy per epoch, mean ± std over seeds).

**Depth** (width 1024)

| Hidden layers | Params | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|---|
| 1 | 12,594,186 | 34.15 ± 1.20 | 29.73 ± 0.64 | −6.77 ± 0.51 * | 23.10 ± 1.30 | 1, 3, 2 | 4.7 × 10⁸ (diverged) |
| **2 (base)** | 13,643,786 | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |
| 3 | 14,693,386 | 60.64 ± 13.42 | 35.47 ± 0.80 | −1.03 ± 0.58 | 32.43 ± 0.68 | 15, 10, 22 | 0.561 |

**Width** (depth 2)

| Width | Params | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|---|
| 256 | 3,214,346 | 51.99 ± 13.12 | 34.73 ± 0.85 | −1.77 ± 0.60 * | 31.47 ± 1.93 | 4, 20, 16 | 0.946 |
| 512 | 6,559,754 | 60.84 ± 14.50 | 35.23 ± 0.90 | −1.27 ± 0.63 * | 32.40 ± 0.17 | 13, 8, 25 | 0.722 |
| **1024 (base)** | 13,643,786 | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |
| 2048 | 29,384,714 | 65.16 ± 13.74 | 35.30 ± 0.96 | −1.20 ± 0.66 | 32.87 ± 1.38 | 13, 6, 14 | 0.403 |

**Dropout**

| Dropout | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|
| 0 | 52.46 ± 0.45 | 34.17 ± 0.92 | −2.33 ± 0.64 * | 32.40 ± 0.46 | 3, 3, 3 | 0.000 |
| 0.2 | 68.95 ± 27.09 | 34.47 ± 1.08 | −2.03 ± 0.71 * | 31.40 ± 1.13 | 3, 30, 5 | 0.043 |
| **0.5 (base)** | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |

**Weight decay**

| Weight decay | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|
| 0 | 66.88 ± 23.36 | 35.47 ± 0.42 | −1.03 ± 0.43 * | 33.23 ± 0.83 | 12, 6, 37 | 0.535 |
| **1e-4 (base)** | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |
| 1e-3 | 83.76 ± 12.03 | 35.63 ± 0.21 | −0.87 ± 0.37 * | 33.00 ± 1.54 | 24, 48, 15 | 0.521 |
| 1e-2 | 62.48 ± 21.60 | 35.43 ± 0.25 | −1.07 ± 0.38 * | 32.73 ± 0.86 | 43, 5, 12 | 0.884 |

**Learning rate** (SGD, momentum 0.9)

| lr | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|
| 0.004 | 57.31 ± 11.51 | 36.47 ± 0.55 | −0.03 ± 0.47 | 32.67 ± 0.31 | 12, 6, 11 | 0.270 |
| **0.013 (base)** | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |
| 0.04 | 26.97 ± 1.78 | 26.80 ± 2.12 | −9.70 ± 1.27 * | 20.70 ± 0.46 | 6, 5, 2 | 1.951 |

**Input normalization** (per-channel standardization with train statistics vs. raw pixels in [0, 1])

| Normalization | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|
| off | 31.55 ± 2.36 | 31.23 ± 0.15 | −5.27 ± 0.36 * | 27.07 ± 1.50 | 32, 13, 16 | 2.016 |
| **on (base)** | 53.49 ± 8.41 | **36.50 ± 0.61** | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |

**Data augmentation** (random horizontal flips of the training images)

| Flips | Train | Val | Δval vs base | Final val | Best epochs | Final train loss |
|---|---|---|---|---|---|---|
| **off (base)** | 53.49 ± 8.41 | 36.50 ± 0.61 | – | 32.60 ± 1.10 | 12, 6, 8 | 0.531 |
| on | 44.13 ± 1.93 | **37.07 ± 0.85** | +0.57 ± 0.60 | 34.87 ± 1.55 | 13, 13, 10 | 1.149 |

**Which differences are meaningful**
- Clearly meaningful (|Δ| ≫ 2·SE, several points): depth 1 (−6.8), lr 0.04 (−9.7), no normalization (−5.3). All three are optimization failures (see below), not capacity effects.
- Moderately meaningful (about 2 points, > 3·SE): less dropout (0 or 0.2: −2.3 / −2.0).
- Borderline (≈ 1–1.8 points, just above 2·SE): smaller widths (−1.8 / −1.3), every change of weight decay (≈ −1). **Caveat:** the base was selected in 2.2 using these same 3 seeds and the same val set (winner's curse), so its val accuracy is optimistically biased by an unknown amount, probably about 1 point. Small negative Δs of all neighbours around the base are what this bias predicts, so they should not be read as real effects. Also, the seed std does not include the sampling noise of the 1000-image val set (≈ 1.5 points SE), which is shared by all runs.
- Not meaningful (< 2·SE): depth 3, width 2048, lr 0.004, flips.

**Under- and overfitting**
- **Overfitting everywhere** at the base settings: train loss keeps falling to 0.53 after 50 epochs while val drops from 36.5 (best, epoch 6–12) to 32.6 (final). The model memorizes 6500 images with 12,288 inputs each.
- **Dropout is the strongest regularizer:** without dropout the train loss reaches 0.000 (perfect memorization), the best epoch is 3 for all seeds, and val peaks at 34.2. Dropout 0.2 is almost as bad (final train loss 0.043).
- **Weight decay barely matters** in the range 0–1e-2 (all within ≈ 1 point, final train losses 0.52–0.88). With dropout 0.5 already in place and only 50 epochs, the L2 penalty is too weak to change the result. (Train acc. of wd 1e-3 looks high because its best epochs are late, 15–48.)
- **Capacity:** more width or depth does not help beyond 1024 × 2. Width 2048 and depth 3 fit the train set better (lower final train loss, higher train acc.) but val does not improve: extra capacity is used for memorization. Width 256 is slightly worse, so 1024 is not much too large either.
- **Flips** act as a regularizer: lower train acc. (44 vs 53), higher final train loss (1.15 vs 0.53) and much less decay after the best epoch (final val 34.87 vs 32.60, Δ = +2.27 ± 1.10, about 2·SE). The best val is only +0.57 (not meaningful), so flips make the model more robust to overtraining rather than better at its best epoch.
- **Underfitting / optimization failures:**
  - *Depth 1 diverged*: train loss rises from epoch 2 (2.32 → 5.3 by epoch 6 → ~5 × 10⁸ at epoch 50) for all seeds; the best val (≈ 30) is reached in epoch 1–3, before the divergence. The lr 0.013 (effective step lr/(1−momentum) ≈ 0.13) was tuned for depth 2 and is too large for one hidden layer. This is a confound of the one-factor-at-a-time design: the result says nothing about the capacity of a 1-layer MLP. (Possible follow-up: depth 1 with lr 0.004.)
  - *lr 0.04*: the train loss stalls at ≈ 2.17–2.19 from epoch 3 on (chance level is ln 10 = 2.30), train ≈ val ≈ 27: underfitting, the step is too large to make progress (probably many dead ReLUs). lr 0.004 gives the same val as 0.013 and a lower final train loss (0.27): the val accuracy is flat in lr below the instability point, as for the linear baseline in 2.1.
  - *No normalization*: train ≈ val (31.6 vs 31.2) and final train loss 2.02: clear underfitting. Raw pixels are all positive with mean ≈ 0.45, so the inputs are far from centered; the gradients of the first layer are then dominated by the mean and the problem is badly conditioned, and with the same lr training progresses much more slowly (best epochs 13–32 instead of 6–12).

**Best model of the study**
- By mean val accuracy (same rule as in 2.2) the best setting is **base + horizontal flips**: val 37.07 ± 0.85 (seeds: 37.10, 37.90, 36.20). Its advantage over the base (+0.57) is not meaningful, see above.
- **Test accuracy: 31.27 ± 0.65** (31.30, 31.90, 30.60). For comparison, the base (`mlp_best`, same 50-epoch budget) has 29.67 ± 1.07 on test. The test difference (+1.6) is larger than on val, but test was not used for the choice and the val difference is within noise, so do not over-interpret it.
- Checkpoints: `checkpoints/q2_3/flip_True/seed{0,1,2}.pt`.
- **Decision: `mlp_best` (2.2) stays "the best MLP" for 2.4 and 2.5.** Reasons: the flip model is not meaningfully better on val (+0.57 ± 0.60), and keeping the 2.2 model keeps the story consistent. Also, a model trained with flips has seen (horizontally) transformed images, which would mix augmentation into the shift experiment of 2.4. The flip result is reported as a finding of 2.3 only. Note that per-class accuracies and the confusion matrix above are of the flip model (the study's best by val).

**Per-class test accuracy** (flip model, mean ± std over 3 seeds, 100 test images per class)

| Class | Accuracy |
|---|---|
| airplane | 58.0 ± 3.6 |
| bus | 41.7 ± 6.5 |
| cat | 41.3 ± 20.8 |
| giraffe | 40.0 ± 3.6 |
| motorcycle | 37.7 ± 6.4 |
| elephant | 27.7 ± 4.7 |
| dog | 20.0 ± 12.3 |
| truck | 20.0 ± 5.2 |
| train | 17.3 ± 7.6 |
| horse | 9.0 ± 6.2 |

**Confusion matrix** (`figures/q2_3/confusion.pdf`, rows = true class, summed over 3 seeds, i.e. 300 test predictions per true class; raw counts in `results/q2_3/summary.json`)

Most frequent confusions (true → predicted, count out of 300):

| True | Predicted | Count |
|---|---|---|
| dog | giraffe | 87 |
| horse | giraffe | 80 |
| dog | cat | 71 |
| elephant | giraffe | 65 |
| train | bus | 54 |
| cat | dog | 53 |
| truck | bus | 52 |
| horse | elephant | 50 |
| motorcycle | cat | 43 |
| giraffe | elephant | 41 |

**Observations on the classes**
- **Errors stay mostly within the super-class:** of the 2062 errors, 61 % stay among animals (cat, dog, elephant, giraffe, horse) or among vehicles (airplane, bus, motorcycle, train, truck). With uniform errors it would be 4/9 = 44 %. So the MLP gets the coarse category right more often than the fine class.
- **Airplane is the easiest class (58 %):** planes are usually seen against a large, uniform sky (blue/grey), a strong global colour cue that a flattened-pixel model can use without knowing the object's shape.
- **Horse is the hardest (9 %), and "giraffe" acts as a sink:** giraffe is predicted 495 times for 300 true giraffes; horse only 120 times. Horses, dogs and elephants are often photographed outdoors on grass, sand or savanna-like backgrounds, as are giraffes, so the MLP, which mostly sees global colour and layout, maps these scenes to giraffe/elephant. Likely explanation: an MLP on raw pixels has no translation invariance and no notion of local shape, so it relies on the background and colour distribution rather than the animal.
- **Visually similar pairs:** dog ↔ cat (71 and 53; similar size, fur, indoor scenes), train/truck → bus (54, 52; large boxy vehicles on streets/rails), horse → elephant (50). Train and truck are much worse than bus (17 % and 20 % vs 42 %): bus is the "sink" for large road/rail vehicles (346 predictions).
- **Cat has a very large seed std (± 20.8):** the boundary between cat and dog moves between seeds (dog ± 12.3 as well). Per-class numbers with 100 images per class and 3 seeds are noisy; only the large differences (airplane vs horse) are clearly meaningful.
- These are hypotheses about the cause; 2.4 (weight visualization, shifts, pixel permutation) can test the "global colour/layout, no local structure" explanation.

## Q2.4 Inductive biases of MLPs

Script: `python -m tasks.q2_4.inductive_biases` (summary in `results/q2_4/summary.json`, log in `results/q2_4/log.txt`). Model: `mlp_best` (decision in 2.3). Helpers `shift_images` and `mask_border` are in `common.py`; `shift_accuracies` and `permutation_experiment` in `tasks/q2_4/inductive_biases.py` are reused for the CNN in 2.5.

**(a) First-layer weights**
- Figures: `figures/q2_4/linear_templates.pdf` (10 class templates of the linear baseline, seed 0), `figures/q2_4/class_means.pdf` (mean training image per class), `figures/q2_4/mlp_weights_random.pdf` (8 random hidden units of `mlp_best`, seed 0), `figures/q2_4/mlp_weights_top.pdf` (the 8 units whose weights changed most). Each weight vector is shown as a 64×64 RGB image, grey = 0, scaled per image by its largest |w|. The weights act on normalized inputs, so colours are relative to the training mean colour.
- **The raw weights look like noise.** Reason: training changes them only a little relative to the random initialization. We rebuilt the exact initialization (same seed, same `set_seed` → `build_model` order as in training) and found: cosine(W, W_init) = 0.90 (MLP) and 0.95 (linear); ‖W − W_init‖ / ‖W_init‖ = 0.49 (MLP) and 0.32 (linear). With 6–30 epochs of SGD the init noise (PyTorch default: uniform ±1/√12288) dominates the picture. So the figures also show the **learned change W − W_init**, which is where the structure is. (Worth one sentence in the report; otherwise the reader sees noise and wonders.)
- **Quantified with two simple statistics** (mean over units):

| | neighbour corr. (horizontal) | R–G–B corr. |
|---|---|---|
| MLP init W_init | 0.000 | – |
| MLP trained W | 0.155 | 0.147 |
| MLP learned change W − W_init | **0.818** | **0.809** |
| Linear trained W | 0.044 | 0.033 |
| Linear learned change W − W_init | **0.741** | **0.660** |

  Neighbour corr. = correlation between horizontally adjacent weights (0 for i.i.d. noise, near 1 for smooth images). R–G–B corr. = correlation between the three colour channels of a unit (1 = the unit only sees brightness).
- **Linear templates (W − W_init):** smooth, low-frequency colour blobs that look like the *average scene* of each class, not like the object. Airplane: blue-grey with a horizontal band (sky, horizon); giraffe, elephant, horse: green/yellow-brown (grass, savanna, trees); cat: warm reddish tones (indoor scenes); bus: reddish/pink lower half (red buses, street). No edges, no object shape. This fits the confusion matrix of 2.3: the classifier separates *backgrounds and colour distributions*, so animals on similar outdoor backgrounds get confused (horse → giraffe).
- **Why the templates show no recognizable objects** (`figures/q2_4/class_means.pdf`): even the *mean training image* of each class shows no object, only a blurred colour/brightness layout (top row). Objects appear at every position, scale and pose, and the cluttered scene covers more of the image than the object, so averaging over 650 images per class leaves only the typical scene. Subtracting the mean image of all classes (bottom row, contrast-stretched) makes the differences visible: airplane = light-blue sky everywhere; bus/train/truck = blue sky at the top, dark road/rails at the bottom; elephant = dark central blob on green/sand; horse = brown blob on green/yellow; giraffe = green top (trees), yellow bottom (savanna); cat/dog = brown/beige indoor tones; motorcycle = dark central shape. A linear classifier has one weight per pixel position, so the best it can do is respond to these position-wise averages. (On centred, aligned data such as MNIST the templates do look like the class; here the data are not aligned.)
- **The learned templates match the class means quantitatively:** correlation between the template W − W_init of class i and the deviation image (class mean − overall mean, in normalized input space) of class j: mean 0.37 on the diagonal (0.25 train … 0.52 giraffe) vs −0.04 off the diagonal; **for all 10/10 templates the own class has the highest correlation.** So the linear model has essentially learned "class mean scene minus average scene", plus the fine-grained differences between classes. A template is still not a prototype: it is discriminative (positive where the class differs from the others, negative where it lacks something), which is why the correlation is moderate, not ~1.
- **MLP hidden units (W − W_init):** also smooth and blob-like (neighbour corr. 0.82), mostly global brightness/colour layouts over the whole image (top vs bottom, centre vs border, some colour-opponent blue vs brown units). No localized edge or part detectors as in a CNN: each unit sees all 12,288 pixels and nothing in the architecture prefers local patterns.
- Interpretation: the smoothness is learned from the data (natural images are smooth, so the gradient, which is a weighted sum of input images, is smooth), not built in. Part (c) confirms this.

**(b) Shifted test images** (no retraining; `figures/q2_4/shifts.pdf`)
- Test images shifted by k pixels to the right or down, the uncovered border filled with 0 (= training mean colour after normalization). Mean ± std over the 3 seeds.
- **Two controls** (image **not** moved), to separate the effect of *misalignment* from the *information loss*:
  - "border only" (`mask_border`): the k-pixel border that the shift uncovers is blanked (e.g. the top k rows for a shift down), i.e. the same blank strip at the same place as in the shifted image;
  - "lost strip" (`mask_lost`): the k-pixel strip whose content the shift pushes out of the frame is blanked (e.g. the bottom k rows for a shift down), i.e. exactly the content the shifted image has lost. For a perfectly translation-invariant model, accuracy on shifted images = accuracy on this control. (This second control was added in 2.5; the first alone does not remove the same content as the shift.)

| Shift (px) | 0 | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| MLP, right | 29.67 ± 1.07 | 29.10 ± 1.25 | 29.10 ± 0.82 | 29.07 ± 1.27 | 28.03 ± 0.90 | 26.10 ± 1.01 | 24.10 ± 0.85 |
| MLP, right, border only | 29.67 | 29.97 ± 0.72 | 29.67 ± 1.08 | 29.03 ± 0.46 | 28.97 ± 0.21 | 29.60 ± 0.26 | 28.97 ± 0.32 |
| MLP, right, lost strip | 29.67 | 29.63 ± 1.18 | 29.67 ± 0.74 | 29.47 ± 0.87 | 29.93 ± 0.75 | 29.37 ± 0.49 | 29.40 ± 0.44 |
| MLP, down | 29.67 ± 1.07 | 29.67 ± 1.36 | 29.20 ± 0.75 | 28.20 ± 1.31 | 23.93 ± 1.12 | 21.20 ± 0.17 | 19.30 ± 1.01 |
| MLP, down, border only | 29.67 | 29.43 ± 1.70 | 29.13 ± 1.46 | 28.67 ± 1.01 | 28.80 ± 0.79 | 28.37 ± 1.33 | 27.60 ± 1.87 |
| MLP, down, lost strip | 29.67 | 29.90 ± 0.82 | 29.97 ± 0.55 | 29.87 ± 0.65 | 28.10 ± 0.53 | 27.90 ± 0.30 | 26.83 ± 0.59 |
| Linear, right | 26.83 ± 0.83 | 26.47 ± 0.91 | 26.47 ± 1.06 | 25.20 ± 0.53 | 24.20 ± 1.51 | 23.67 ± 0.31 | 21.43 ± 0.61 |
| Linear, down | 26.83 ± 0.83 | 27.17 ± 0.67 | 27.43 ± 0.71 | 24.80 ± 0.95 | 22.17 ± 0.06 | 19.60 ± 1.44 | 18.43 ± 0.31 |

(Linear border-only control at 8–16 px: 25.1–25.9 for right, 24.1–25.4 for down; full numbers incl. 3 and 6 px in the JSON/log.)

- **Both controls cost little** (MLP up to 16 px: border only ≤ 2.1 points, lost strip ≤ 2.8 points, the largest for the bottom rows). So the drop under shifting is due to **misalignment**, not to missing image content: at 16 px, shifting costs −5.3 (right) and −7.5 (down) points more than losing the same content without moving the image (lost-strip control).
- **Small shifts (1–4 px) barely hurt** (≤ 1.5 points, within the seed std). This is *not* a built-in invariance: it follows from the learned weights being smooth (neighbour corr. 0.82). A 1-pixel shift barely changes the dot product of a smooth template with the image. Accuracy degrades steadily with the shift size, as expected for a model whose weights are tied to absolute pixel positions.
- **Vertical shifts hurt about twice as much as horizontal ones.** Likely reason: the classes are recognized from the global vertical layout of the scene (sky at the top, ground/grass at the bottom, horizon height), which the templates in (a) show; horizontal position is more variable in photos (objects appear left or right), so the learned templates are more tolerant horizontally.
- The linear baseline behaves the same way (at 16 px: −5.4 right, −8.4 down vs. unshifted; −4.5 and −7.0 vs. its lost-strip control), so this is a property of "one weight per pixel position", not of the hidden layers.
- The MLP stays far above chance (10 %) even at 16 px: much of what it uses (global colour statistics) survives a shift.

**(c) Fixed random pixel permutation** (retrained from scratch)
- One fixed random permutation of the 64×64 pixel positions (`pixel_permutation(seed=0)`, the same for all 3 channels) applied to train, val and test. The `mlp_best` configuration retrained with the same protocol (3 seeds, at most 50 epochs, early stopping with patience 8 on val). Per-channel normalization statistics are unchanged by a permutation. Results in `results/q2_4/mlp_perm/`, checkpoints in `checkpoints/q2_4/mlp_perm/`.

| Pixels | Train | Val | Test | Best epochs |
|---|---|---|---|---|
| original (`mlp_best`) | 53.49 ± 8.41 | 36.50 ± 0.61 | 29.67 ± 1.07 | 12, 6, 8 |
| permuted (`mlp_perm`) | 51.62 ± 11.64 | 35.60 ± 0.80 | 30.20 ± 1.15 | 12, 8, 4 |

- **No meaningful difference** (val −0.90 ± 0.58, test +0.53 ± 0.89; both < 2·SE). The permuted images are unrecognizable for a human, but the MLP learns them just as well.
- **Why, exactly:** the first layer computes W x. For a permutation matrix P, W (P x) = (W P) x, so an MLP with weights W P on the original images computes the same function as an MLP with weights W on permuted images. The initialization is i.i.d. over the inputs (its distribution does not change if the columns of W are permuted), and SGD updates commute with the permutation. So training on permuted pixels is the same learning problem up to relabelling the input dimensions: the MLP is **invariant to the order of its inputs as a learning algorithm**. The small differences come only from different random draws (the init is not permuted along with the data).
- `figures/q2_4/perm_weights.pdf`: the learned change W − W_init of the permuted model looks like noise as learned (neighbour corr. 0.033), but after mapping it back to the original pixel layout it is smooth again (0.819, the same as the original MLP's 0.818) and shows the same kind of colour blobs. The model learned the same thing, just in a scrambled coordinate system.

**(d) Confusion matrix of the best MLP** (`figures/q2_4/confusion_mlp.pdf`; `q2_2/mlp_best`, test, summed over 3 seeds = 300 predictions per class; numbers in `results/q2_4/summary.json`). Added for comparison with the CNN in 2.5; 2.3 reported the confusion matrix of the study's best setting (the flip model), which is very similar.

| Class | Test acc. (mean ± std) |
|---|---|
| airplane | 64.0 ± 5.0 |
| motorcycle | 44.7 ± 7.5 |
| cat | 33.0 ± 8.5 |
| bus | 31.7 ± 2.1 |
| giraffe | 29.3 ± 4.9 |
| train | 23.3 ± 8.7 |
| elephant | 22.3 ± 6.8 |
| dog | 16.7 ± 7.8 |
| horse | 16.7 ± 11.0 |
| truck | 15.0 ± 8.2 |

- Most frequent confusions (count / 300): dog → cat 82, dog → giraffe 55, truck → motorcycle 53, cat → motorcycle 51, elephant → horse 47, elephant → giraffe 47, cat → dog 47, truck → train 44, train → airplane 41, elephant → motorcycle 41.
- Same picture as in 2.3: airplane (sky) is easiest; 60 % of the errors stay within animals/vehicles (44 % expected for uniform errors); several confusions only make sense via colour/background (dog → giraffe, cat → motorcycle, train → airplane: grey/blue sky-like backgrounds). The "sink" class differs from the flip model (here motorcycle 454 and airplane 405 predictions for 300 true images, there giraffe), i.e. which class absorbs the uncertain images is not stable across models.

**What this says about the MLP's assumptions (for the report)**
- An MLP makes **no assumption about the spatial structure of images**: it does not know which pixels are neighbours (c), and it has no weight sharing across positions, so a pattern learned at one position does not transfer to another (b). Every pixel position is an independent input feature.
- What looks like structure in its weights (smooth templates, robustness to 1–4 px shifts) is learned from the data, and it costs data: with 6500 images the MLP can only learn coarse global colour/layout templates (a), which explains the confusions between classes with similar backgrounds (2.3) and the strong overfitting (12.6M first-layer parameters, one per pixel and unit).
- A CNN builds in exactly the two missing assumptions, locality (small kernels) and translation equivariance (weight sharing). The prediction for 2.5: the CNN should handle shifts much better and should lose a lot of accuracy on permuted pixels, because the permutation destroys the local neighbourhoods its kernels rely on.

## Q2.5 From MLPs to CNNs

Script: `python -m tasks.q2_5.cnn` (summary in `results/q2_5/summary.json`, log in `results/q2_5/log.txt`). Model class `CNN` in `models.py`; CNN-specific config keys (`channels`, `convs_per_block`, `batchnorm`) in `common.CNN_DEFAULTS`.

**Architecture** (`models.CNN`)
- `len(channels)` blocks of [Conv 3×3, padding 1 → BatchNorm → ReLU] × `convs_per_block`, each block followed by 2×2 max pooling (64 → 32 → 16 → 8 → 4), then **global average pooling** over the 4×4 map, dropout, and one linear layer to the 10 classes.
- Design rationale (links to Q1.1): 3×3 convolutions = locality + weight sharing (a constrained, Toeplitz/circulant-like linear layer, Q1.1(a)); convolutions + ReLU are translation equivariant (Q1.1(b)); global average pooling turns equivariance into (approximate) invariance (Q1.1(b)). `features(x)` returns the pooled vector, for reuse as an image encoder in Part 3.

**Protocol** (same as for the MLP in 2.2)
- Random search, 16 configurations (sampling seed 0), each trained once with seed 0; at most 50 epochs, early stopping on val accuracy with patience 8, best val epoch restored. Then the top 3 re-run with seeds 0, 1, 2; selection by **mean** val accuracy; test read only for the selected model. No data augmentation (as for the MLP; also keeps the shift experiment clean).
- Search space:

| Hyperparameter | Values |
|---|---|
| Architecture | channels (32, 64, 128, 256) with 1 conv/block (391,466 params) or 2 conv/block (1,175,786), or (64, 128, 256, 512) with 1 conv/block (1,557,066) |
| Dropout (before the output layer) | 0, 0.3, 0.5 |
| Optimizer | Adam (lr log-uniform in [3e-4, 3e-3]) or SGD with momentum 0.9 (lr log-uniform in [3e-3, 1e-1]) |
| Weight decay | 0, 1e-4, 5e-4 |
| Batch size | 64, 128 |
| Scheduler | none or cosine (T_max = 50) |

- Fixed: BatchNorm on, ReLU. A larger variant ((64, …, 512) with 2 conv/block, 4.7M params) was in the first version of the search space but dropped before any result was used: on the 8 GB laptop it needed ~6 GB of GPU memory and ~30 s/epoch. All CNNs are far smaller than the MLP (13.6M), as required.
- Hardware note: CNNs were trained on the Apple GPU (MPS) via `common.run_device` (≈ 8 s/epoch for the selected CNN vs ≈ 17+ s on CPU); repeated MPS runs gave bit-identical results. MLPs stayed on the CPU. Evaluation batch size 100 for CNNs (memory), which does not change predictions. Runs that are identical by construction (top-3 seed 0 = search run; final runs = top-3 runs of the selected config) are copied instead of retrained (`reuse_run`), marked with `copied_from` in the JSON.

**Search results** (val, seed 0; full ranking in the summary JSON): the best trials were all SGD + cosine or Adam + cosine and trained long (best epochs 28–47, train acc ≈ 100 %): 76.5 (1.18M, no dropout, SGD lr 0.01), 73.5 (1.18M, dropout 0.5, Adam lr 3.2e-4), 73.4 (1.18M, dropout 0.3, SGD lr 0.0031), 72.1 (1.56M), 71.8 (0.39M). The worst trial still reached 57.4 (vs MLP 36.5).

**Re-evaluation of the top 3** (3 seeds)

| Search rank | Configuration | Val per seed | Val (mean ± std) | Best epochs |
|---|---|---|---|---|
| 1 | 1.18M, dropout 0, SGD lr 0.01, wd 5e-4, bs 64, cosine | 76.5, 62.5, 67.2 | 68.73 ± 7.12 | 31, 8, 11 |
| 2 | 1.18M, dropout 0.5, Adam lr 3.2e-4, wd 5e-4, bs 64, cosine | 73.5, 73.2, 62.6 | 69.77 ± 6.21 | 47, 40, 5 |
| **3** | **1.18M, dropout 0.3, SGD lr 0.0031, wd 5e-4, bs 64, cosine** | 73.4, 66.8, 72.9 | **71.03 ± 3.67** | 30, 14, 33 |

- Again the best single-seed trial (76.5) lost after the 3-seed re-check, as with the MLP in 2.2.

**Selected CNN**: 4 blocks × 2 conv (32, 64, 128, 256 channels), BatchNorm, ReLU, global average pooling, dropout 0.3; SGD momentum 0.9, lr 0.0031, weight decay 5e-4, batch size 64, cosine schedule; 1,175,786 parameters. Learning curves: `figures/q2_5/curves.pdf`. Checkpoints: `checkpoints/q2_5/cnn_best/`.

**Main comparison** (test = final numbers)

| Model | Parameters | Train | Val | Test |
|---|---|---|---|---|
| Linear (2.1) | 122,890 | 42.14 ± 3.49 | 32.20 ± 0.50 | 26.83 ± 0.83 |
| MLP (2.2) | 13,643,786 | 53.49 ± 8.41 | 36.50 ± 0.61 | 29.67 ± 1.07 |
| **CNN (2.5)** | **1,175,786** | 93.53 ± 11.20 | **71.03 ± 3.67** | **67.70 ± 2.86** |

Per seed (test): linear 27.1, 27.5, 25.9; MLP 30.9, 29.1, 29.0; CNN 69.4, 64.4, 69.3.

- The CNN is **+38 points** better than the MLP on test with **11.6× fewer parameters** (and +41 over the linear model). This is by far the largest effect in Part 2; all MLP tuning in 2.2/2.3 moved test accuracy by only 1–3 points.
- Sanity check against leakage (because the jump is so large): no exact duplicates between train and val/test (MD5 of the raw images: 0/1000 val, 0/1000 test), and no near-duplicates (nearest train image of each val image in an 8×8 grayscale downsampling: min distance 88.5, median 247.7; a duplicate would be ≈ 0).
- **Variance across seeds is much larger than for the MLP** (val ± 3.67 vs ± 0.61). Reason, visible in `figures/q2_5/curves.pdf`: the val accuracy is very noisy in the first ~20 epochs (jumps of 10–20 points between epochs at the higher learning rates of the cosine schedule) and only stabilizes when the cosine schedule has reduced the lr. With patience 8, early stopping sometimes fires in this noisy phase: seed 1 stopped after epoch 22 (best epoch 14, val 66.8, train loss still 0.11), while seeds 0 and 2 trained to epoch 38/41 (best 30/33, val ≈ 73, train loss 0.007). The same pattern appears in all three top configurations: each has at least one seed with best epoch 5–14 and 7–14 points lower val than its best seed. **Early stopping with a small patience and a cosine schedule do not fit well together**; a fixed budget of 50 epochs gives higher and much more stable CNN numbers (see the check below). We kept patience 8 so the protocol is identical to the MLP's; the comparison is, if anything, slightly unfavourable to the CNN.
- **Check with a fixed budget** (requested follow-up; `results/q2_5/cnn_best_50ep/`, `figures/q2_5/curves_50ep.pdf`): the selected CNN re-trained with the same config and seeds but **50 epochs, no early stopping** (best val epoch still restored). Not part of the selection protocol; the main table keeps the patience-8 numbers so that the CNN and MLP protocols are identical.

| Selected CNN | Train | Val | Test | Best epochs |
|---|---|---|---|---|
| patience 8 (protocol) | 93.53 ± 11.20 | 71.03 ± 3.67 | 67.70 ± 2.86 | 30, 14, 33 |
| 50 epochs, no early stopping | 100.00 ± 0.00 | **73.63 ± 0.70** | **70.27 ± 1.00** | 41, 40, 33 |

  Per seed (val / test): patience 8: 73.4/69.4, 66.8/64.4, 72.9/69.3; fixed budget: 73.7/70.2, 74.3/71.3, 72.9/69.3.
  - Confirms the explanation: seed 1 improves from 66.8 to 74.3 val (64.4 → 71.3 test) once it is allowed to train through the noisy phase; its best epoch moves from 14 to 40. Seed 0 also gains a little (best epoch 30 → 41: patience 8 had stopped it at epoch 38). Seed 2 is identical (it was never cut short).
  - The seed std drops from 3.67 to **0.70** on val (2.86 → 1.00 on test), i.e. to the MLP's level (0.61). Most of the CNN's seed variance was caused by early stopping, not by the model.
  - Lesson for the report: with a cosine schedule the best epochs are late (when the lr is small), so early stopping with a small patience is a poor fit; a fixed budget (the schedule's T_max) with restoring the best val epoch is the natural protocol. With the fixed budget the CNN is **+40.6 points** over the MLP on test (70.27 vs 29.67).
- Overfitting: train accuracy ≈ 100 % and train loss ≈ 0.007 at the best epoch for seeds 0/2, yet val keeps improving until epoch ~30. The CNN generalizes much better than the MLP despite fitting the training set perfectly.

**Shift experiment with the CNN** (`figures/q2_5/shifts.pdf`, one row per model: linear, MLP, CNN; same protocol and controls as 2.4)

| Shift (px) | 0 | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| CNN, right | 67.70 ± 2.86 | 67.63 ± 2.95 | 67.00 ± 3.47 | 65.17 ± 4.43 | 63.07 ± 4.47 | 59.03 ± 5.93 | 56.03 ± 6.43 |
| CNN, right, lost strip | 67.70 | 67.53 ± 2.72 | 67.20 ± 2.69 | 67.50 ± 2.72 | 66.53 ± 1.70 | 65.33 ± 2.73 | 63.00 ± 1.80 |
| CNN, right, border only | 67.70 | 67.53 ± 2.71 | 66.97 ± 3.74 | 66.17 ± 3.95 | 63.73 ± 4.83 | 61.30 ± 6.67 | 58.10 ± 7.94 |
| CNN, down | 67.70 ± 2.86 | 67.43 ± 3.16 | 66.17 ± 3.43 | 66.30 ± 4.23 | 63.50 ± 2.10 | 60.57 ± 2.41 | 55.13 ± 2.67 |
| CNN, down, lost strip | 67.70 | 67.20 ± 3.12 | 66.87 ± 3.33 | 66.30 ± 3.51 | 64.07 ± 2.57 | 63.10 ± 2.04 | 59.07 ± 3.00 |
| CNN, down, border only | 67.70 | 67.33 ± 3.16 | 67.20 ± 3.35 | 66.50 ± 3.33 | 65.60 ± 2.62 | 65.00 ± 2.86 | 63.93 ± 2.12 |

Cost of the shift beyond losing the same content (shift − lost-strip control) at 16 px, absolute and relative to the accuracy above chance (acc. − 10 %):

| Model | right, points | right, rel. | down, points | down, rel. |
|---|---|---|---|---|
| Linear | −4.5 | −27 % | −7.0 | −42 % |
| MLP | −5.3 | −27 % | −7.5 | −38 % |
| **CNN** | −7.0 | **−12 %** | −3.9 | **−7 %** |

- **Small shifts (1–2 px) leave the CNN unchanged** (shifted = lost-strip control within 0.7 points), as expected from translation equivariance + global average pooling.
- **The CNN is much more robust than the MLP relative to what it knows**: at 16 px down the shift (beyond the content loss) costs the CNN 7 % of its above-chance accuracy, the MLP 38 % and the linear model 42 %; for right shifts 12 % vs 27 %. In absolute points the losses look similar (−4 to −7) only because the CNN starts 38 points higher.
- **But the CNN is not perfectly invariant.** Reasons: (i) **zero padding** at the image borders: every conv layer pads with 0, so features near the border differ from those in the interior, and the network can learn where the border is, i.e. absolute position (known effect: CNNs encode position through padding); in the shifted image the content touches a different border than in the control; (ii) **the blank strip is not equivalent to padding** after the first layer (BatchNorm/bias make the features of a constant region non-zero, while padding stays 0); (iii) **max pooling with stride 2** (4×) makes the network equivariant only to shifts that are multiples of 16 px, so 1–15 px shifts change the pooling grid; (iv) the final map is only 4×4, so a 16 px shift is a whole column/row of it. Q1.1(b) assumes circular padding and no striding, which makes equivariance exact; our CNN only approximates those assumptions.
- The "border only" control hurts the CNN more than the MLP for right shifts (−9.6 at 16 px): a blank strip on the left is itself a strong local pattern (an edge) that the CNN reacts to wherever it is, while the MLP's smooth templates barely react to it.

**Pixel-permutation experiment with the CNN** (same fixed permutation as in 2.4, seed 0; retrained from scratch with the selected config and the same protocol; `results/q2_5/cnn_perm/`)

| Model | Pixels | Train | Val | Test | Best epochs |
|---|---|---|---|---|---|
| MLP | original | 53.49 ± 8.41 | 36.50 ± 0.61 | 29.67 ± 1.07 | 12, 6, 8 |
| MLP | permuted | 51.62 ± 11.64 | 35.60 ± 0.80 | 30.20 ± 1.15 | 12, 8, 4 |
| CNN | original | 93.53 ± 11.20 | 71.03 ± 3.67 | 67.70 ± 2.86 | 30, 14, 33 |
| CNN | permuted | 31.87 ± 1.77 | 29.47 ± 1.27 | 28.57 ± 1.20 | 9, 10, 10 |

- **The CNN collapses from 67.7 to 28.6 on test (−39 points), the MLP is unchanged (+0.5).** On permuted pixels the CNN is no better than the MLP or the linear model.
- Why: a 3×3 kernel combines a pixel with its 8 neighbours, assuming that neighbouring pixels belong together (locality). After the permutation, the 9 pixels under a kernel are 9 random positions of the original image, so local patterns (edges, textures, parts) no longer exist; weight sharing then shares a filter for patterns that do not repeat. The architecture's prior is now wrong, and the CNN even underfits: train acc. 31.9 % (vs 93.5 %), train loss stays at 1.2–1.7.
- Why it still gets ~29 % (far above chance): global average pooling of per-pixel features is invariant to any permutation of positions, and colour statistics survive the permutation. So the permuted CNN can still use the colour distribution of the image, which is about what the MLP and the linear model use anyway (2.4 (a)). The MLP keeps exactly its accuracy because it never used locality (2.4 (c)).

**Confusion matrix of the best CNN** (`figures/q2_5/confusion_cnn.pdf` for `q2_5/cnn_best`, the protocol model; `figures/q2_5/confusion_cnn_50ep.pdf` for the fixed-budget check; test, summed over 3 seeds; numbers in `results/q2_5/summary.json`)

| Class | MLP (2.2) | CNN, patience 8 | CNN, 50 epochs |
|---|---|---|---|
| airplane | 64.0 ± 5.0 | 75.0 ± 7.9 | 81.0 ± 3.0 |
| bus | 31.7 ± 2.1 | 73.0 ± 3.6 | 74.7 ± 0.6 |
| cat | 33.0 ± 8.5 | 68.7 ± 14.0 | 76.7 ± 3.1 |
| dog | 16.7 ± 7.8 | 56.3 ± 19.6 | 48.3 ± 7.4 |
| elephant | 22.3 ± 6.8 | 67.3 ± 5.5 | 74.7 ± 4.0 |
| giraffe | 29.3 ± 4.9 | 78.3 ± 4.0 | 82.7 ± 2.1 |
| horse | 16.7 ± 11.0 | 47.0 ± 9.8 | 51.3 ± 2.3 |
| motorcycle | 44.7 ± 7.5 | 84.3 ± 7.5 | 86.7 ± 2.3 |
| train | 23.3 ± 8.7 | 73.3 ± 2.9 | 71.7 ± 3.5 |
| truck | 15.0 ± 8.2 | 53.7 ± 4.2 | 55.0 ± 1.7 |

- Most frequent confusions of the CNN (patience 8, count / 300): dog → cat 88, cat → dog 75, truck → bus 51, horse → dog 50, bus → truck 41, elephant → dog 37, train → truck 27, truck → train 25, horse → motorcycle 24, horse → elephant 20. (50 epochs: dog → cat 101, truck → bus 48, cat → dog 48, bus → truck 37, train → truck 30, horse → elephant 29.)
- **The CNN improves every class**, most for the classes defined by shape rather than scene: giraffe +49, bus +41, train +50, motorcycle +40, elephant +45 points. Airplane gains least (+11): the MLP could already recognize it from the sky.
- **Its errors are semantically plausible:** 78 % of the CNN's errors stay within animals/vehicles (MLP: 60 %, uniform: 44 %), and the top confusions are visually similar pairs: dog ↔ cat (furry pets, similar size and indoor scenes), truck ↔ bus ↔ train (large boxy vehicles), horse → dog/elephant (four-legged animals). The MLP's colour-based confusions (dog → giraffe, cat → motorcycle, train → airplane) are gone.
- Hardest classes for the CNN: horse (47–51 %), truck (54–55 %), dog (48–56 %). Each has a close neighbour class with similar shape and context; dog vs cat is the single largest error (29–34 % of dogs predicted as cat). With 64×64 images and 650 training images per class, fine-grained cues (ears, snout, cargo box) are hard to learn.
- The predicted class counts are much more balanced for the CNN (206–394 per 300 true; MLP 200–454), i.e. no strong "sink" class.
- The very large per-class stds of the patience-8 CNN (cat ± 14.0, dog ± 19.6) come from the early-stopped seed 1 (see above); with the fixed budget they shrink to ± 3–7.

**Explanation with Q1.1** (for the report)
- Q1.1(a): a convolution is a linear layer W with a fixed structure (each row is the kernel, shifted; circulant with circular padding), so it has k (here 3×3×C_in per output channel) free parameters instead of n². Q1.1(c) for 64×64×3 → 64×64×16: dense layer 12,288 × 65,536 + 65,536 = 805,371,904 parameters; 3×3 conv with 16 channels: 3·3·3·16 + 16 = 448. Our whole CNN (1.18M) has fewer parameters than the first layer of our MLP (12.6M).
- What the convolution gains: far fewer parameters (less overfitting with 6500 images), and the same detector at every position (a feature learned in one place works everywhere, i.e. it learns from all positions at once). Together with pooling this gives (approximate) translation invariance (Q1.1(b)): small shifts do nothing, large shifts cost much less than for the MLP.
- What it gives up: it cannot represent arbitrary linear maps of the image, in particular ones that combine distant pixels or depend on absolute position (Q1.1(b): equivariant ⇔ circulant). This is a restriction only if the data violate the locality/stationarity assumption, which is exactly what the permutation does: there the CNN loses its advantage completely, while the MLP, having no such prior, does not care.
- In short: the CNN's large gain on real images and its collapse on permuted images are two sides of the same inductive bias.

# Part 3: Image captioning

## Q3.1 Text preprocessing

Code: `text.py` (`tokenize`, `Vocabulary`); script `python -m tasks.q3_1.text_stats` (summary in `results/q3_1/summary.json`, vocabulary in `results/q3_1/vocab.json`, log in `results/q3_1/log.txt`, figure `figures/q3_1/caption_lengths.pdf`).

**Tokenizer** (own implementation, one regular expression)
- Lowercase; a token is either a word (letters/digits, optionally with one apostrophe part, so "man's", "don't" stay one token) or a single punctuation character; everything else (whitespace) separates tokens. Example: "A green semi-truck drives down a shaded street." → `a green semi - truck drives down a shaded street .`
- Separating punctuation matters for the class mention metric in 3.3(b): "semi-truck" yields the token "truck".

**Vocabulary** (built from the 32,500 training captions only)
- Special tokens: `<pad>` = 0, `<bos>` = 1, `<eos>` = 2, `<unk>` = 3; then the words by decreasing training frequency (ties alphabetically, so the vocabulary is deterministic).
- `encode(caption) = [<bos>] + ids + [<eos>]`, words not in the vocabulary → `<unk>`. This is the `encode` passed to `caption_loaders`, with `pad_id = 0`.
- **Frequency threshold: min_count = 5** (words seen fewer than 5 times in the training captions → `<unk>`).

| min_count | Vocabulary size (incl. 4 special) | Val `<unk>` tokens | Val captions with ≥ 1 `<unk>` |
|---|---|---|---|
| 1 | 6,625 | 0.88 % | 8.32 % |
| 2 | 3,881 | 1.37 % | 12.42 % |
| 3 | 2,955 | 1.77 % | 15.84 % |
| **5** | **2,193** | **2.36 %** | **20.52 %** |
| 10 | 1,451 | 3.57 % | 29.24 % |
| 20 | 950 | 5.40 % | 40.28 % |

- Rationale: 6,621 distinct training tokens, 4,432 of them (67 %) occur fewer than 5 times but together cover only 1.99 % of the training tokens (mostly typos, rare names, numbers: the captions contain the typos of real annotators, e.g. "tuck" for "truck"). With min_count = 5 the vocabulary shrinks by 3× (fewer output-layer parameters, no embeddings for words seen 1–4 times that cannot be learned well), while only 2.4 % of the validation tokens become `<unk>`. Even with min_count = 1, 0.9 % of the val tokens are unseen words.
- **Reported numbers: vocabulary size 2,193 (2,189 words + 4 special tokens); `<unk>` fraction in the validation captions 2.36 % of 55,155 tokens** (20.5 % of the validation captions contain at least one `<unk>`). Train 1.99 %, test 2.44 %.
- Most frequent words: a (50,179), . (24,491), the, on, in, of, is, with, and, to, an, then train, standing, cat, dog, bus, two, next, parked, truck: function words dominate, and the class names are among the most frequent content words.

**Caption lengths** (tokens incl. punctuation, without `<bos>`/`<eos>`)

| Split | Captions | Mean ± std | Min | Median | 95 % | 99 % | Max | ≤ 15 tokens |
|---|---|---|---|---|---|---|---|---|
| Train | 32,500 | 11.11 ± 2.42 | 7 | 11 | 15 | 20 | 53 | 95.7 % |
| Validation | 5,000 | 11.03 ± 2.20 | 7 | 11 | 15 | 18 | 27 | 96.2 % |
| Test | 5,000 | 11.02 ± 2.30 | 7 | 11 | 15 | 19 | 43 | 95.8 % |

- Matches the project statement (about 11 tokens, 95 % with at most 15). Short and narrowly distributed (mode 10, IQR 10–12), with a long right tail (a few captions of 40–53 tokens). The three splits have the same distribution. (Figure: `figures/q3_1/caption_lengths.pdf`.)
- Reference point for 3.3(b): the human reference captions mention the image's class (with `class_mention_accuracy` and our tokenizer) in 93.5 % (train), 95.2 % (val), 95.4 % (test) of the cases.

**How the provided loaders batch captions of different lengths, and how padding is ignored**
- `caption_loaders` builds a `CaptionDataset` with one item per (image, caption) pair: every image appears 5 times per epoch, so an epoch has 32,500 training items. `encode` is applied once, when the dataset is built.
- `PadCollate` stacks the images and right-pads the captions with `pad_id` to the length of the **longest caption in the mini-batch** (`pad_sequence(..., batch_first=True)`), giving `captions (B, T)` with T varying per batch. Example (batch size 64): the first three training batches have T = 19, 23, 26; a short caption becomes `<bos> red double decker bus on a city street <eos> <pad> × 9`. Over one epoch, 41.8 % of the caption tensor entries are padding (long outlier captions make whole batches long).
- **Ignoring padding in the loss:** with teacher forcing the decoder input is `captions[:, :-1]` and the target is `captions[:, 1:]`; the token-level cross-entropy is computed with `F.cross_entropy(..., ignore_index=pad_id)`, so positions whose target is `<pad>` contribute neither to the loss nor to the gradient, and the mean is taken over real target tokens only (each caption's `<eos>` is still a target, so the model learns when to stop). Check in the script: for random logits, appending 5 extra `<pad>` columns leaves the loss with `ignore_index` exactly unchanged (8.15563 → 8.15563), while counting pads as targets changes it (8.16371 → 8.15813). Perplexity = exp(mean loss over real tokens).
- The decoder's outputs at padded input positions are computed but ignored; since padding is on the right, a unidirectional RNN never reads a `<pad>` before a real token, so padding cannot influence the predictions of real tokens.

## Q3.2 An RNN captioner

Code: `models.Captioner`, `captioning.py` (training, losses, decoding, metrics; results saved like Part 2); script `python -m tasks.q3_2.captioner` (summary in `results/q3_2/summary.json`, log in `results/q3_2/log.txt`, figures `figures/q3_2/curves.pdf`, `figures/q3_2/examples.pdf`).

**Model** (`models.Captioner`)
- **Encoder:** the selected CNN of Q2.5 (4 blocks × 2 conv, 32–256 channels), **pretrained on classification** (the 50-epoch checkpoints `q2_5/cnn_best_50ep`, same seed as the captioner) and **frozen** (no gradients, always in eval mode so BatchNorm uses its running statistics). Image vector f = the 256-d globally average-pooled features (`CNN.features`), i.e. the input of the classification layer.
- **Feeding the image to the decoder:** h₀ = tanh(W_h f + b) initializes the GRU state; in the selected configuration (`concat_image`) a second projection W_x f (256-d) is also concatenated to the word embedding at **every** time step.
- **Decoder:** word embedding (2,193 × 256, `padding_idx` = `<pad>`), one-layer GRU (hidden 512), dropout 0.3 on embeddings and GRU outputs, linear layer to the 2,193-word vocabulary.
- Parameters: 4,635,515 in total, of which 3,459,729 trainable (decoder) and 1,175,786 in the frozen encoder.

**Training**
- Teacher forcing: input `<bos> w₁ … w_n`, target `w₁ … w_n <eos>`; token-level cross-entropy with `ignore_index = <pad>` (3.1); Adam, lr 1e-3, batch size 64 (image–caption pairs), gradient-norm clipping at 5, at most 20 epochs, early stopping on **validation loss** with patience 4, best epoch restored (provided `train` loop; `val_score_fn` = − validation loss).
- Greedy decoding: start with `<bos>`, take the arg-max word at every step, stop at `<eos>` or after 25 tokens (99 % of reference captions have ≤ 20 tokens; no generated caption reached the limit).
- BLEU-4: corpus-level (`metrics.bleu4`), one generated caption per image (1,000 val / 1,000 test images) against its 5 reference captions, all tokenized with our tokenizer (lowercase, punctuation as tokens).
- Hardware: Apple GPU (MPS); one epoch ≈ 55 s with the frozen CNN. Frozen-encoder runs on MPS were bit-reproducible.

**Decoder search** (seed 0, frozen CNN encoder; one change at a time w.r.t. the base; selection by validation BLEU-4)

| Trial | Val loss (ppl) | Val BLEU-4 | Test BLEU-4 | Test class mention | Best epoch |
|---|---|---|---|---|---|
| base (h₀ only, dropout 0.3, lr 1e-3, hidden 512) | 2.480 (11.9) | 13.83 | 12.02 | 66.3 % | 6 |
| **concat_image** (image also at every step) | 2.475 (11.9) | **15.01** | 14.95 | 69.2 % | 8 |
| dropout 0.5 | 2.498 (12.2) | 14.55 | 14.06 | 69.2 % | 14 |
| lr 3e-4 | **2.450 (11.6)** | 13.91 | 13.68 | 67.8 % | 18 |
| hidden 256 | 2.469 (11.8) | 13.14 | 14.11 | 68.9 % | 15 |

- All trials are within ~2 BLEU points with 1 seed, about the seed-to-seed spread (± 0.7–0.8, see below), so the choice is not strongly supported; concat_image is also the variant motivated by the architecture (the image information cannot fade over the sequence).
- **Validation loss and BLEU rank the trials differently** (lr 3e-4 has the best perplexity but only average BLEU). Lower perplexity = better probability for the reference words under teacher forcing; BLEU measures the single greedy caption. They are related but not the same objective.

**Selected captioner, 3 seeds** (`results/q3_2/final/`)

| Split | Loss (cross-entropy / token) | Perplexity | BLEU-4 | Class mention | Mean length |
|---|---|---|---|---|---|
| Train (eval mode) | 1.940 ± 0.091 | 6.98 | – | – | – |
| Validation | 2.480 ± 0.006 | 11.94 ± 0.07 | **14.91 ± 0.72** | 70.9 % | 10.95 |
| Test | 2.509 ± 0.008 | 12.29 ± 0.10 | **14.02 ± 0.83** | 68.6 % | 10.96 |

Per seed (val / test BLEU-4): 15.01 / 14.95, 14.15 / 13.37, 15.58 / 13.74; best epochs 8, 5, 7.

**Loss and perplexity curves** (`figures/q3_2/curves.pdf`; dashed = training loss averaged over the epoch with dropout on, solid = validation loss in eval mode)
- Validation loss drops quickly (2.70 → 2.50 in 4 epochs) and then flattens; its minimum is at epoch 5–8 (val ppl ≈ 11.9), after which it rises slowly while the training loss keeps falling (2.04 at epoch 12, ppl 7.7): **overfitting starts after ~6 epochs**. The train/val gap at the restored epoch is large (eval-mode train loss 1.94 vs val 2.48).
- In epoch 1 the training loss (3.17) is above the validation loss (2.71) only because it is averaged over the epoch while the model is still learning (and with dropout).
- The seeds behave almost identically (val loss ± 0.006).

**Generated captions** (`figures/q3_2/examples.pdf`: 12 test images, seed 0; 4 best and 4 worst by a smoothed sentence-level BLEU-2 used only for picking, 4 random; each with the generated caption and one reference; all 5 references in the log)
- Good: "a truck is parked on the side of the road ." (exactly a reference), "a train traveling down tracks next to a forest ." (exactly a reference), "a bus is parked on the side of the road ." (two images).
- Random: horse with man and draft horses → "a large white airplane sitting on top of a runway ." (wrong class); show jumping → "a man riding a horse in front of a building ." (right class, plausible but generic); man under chained elephant → "a group of elephants standing next to each other ." (right class, wrong count/scene); city bus → "a bus is parked on the side of the street ." (correct).
- Bad: airplane at dusk → "a black and white photo of a black and white photo of a man and a woman" (wrong class, repetition); airplane in the sky → "a black and white cat sitting on top of a grass field ." (wrong class); horses pulling a plow → "a man riding a horse drawn carriage on a city street ." (right class, wrong scene); two elephants in the dark → "a dog with a hat on its head".

**Typical errors** (test, seed 0, quantified)
- **Generic, template-like captions** (mode collapse to frequent training sentences): only 439 of 1,000 test captions are distinct (398 and 409 for the other seeds); the most frequent are "a bus is parked on the side of the road ." (36×), "a train traveling down tracks next to a forest ." (25×), "a man riding a motorcycle down a street ." (24×). 14–20 % of the generated captions are copies of a training caption. Greedy decoding picks the most probable continuation at each step, which favours the most common phrasing of each class.
- **Wrong class:** the caption mentions the correct class in 68.6 % of the test images (human references: 95.4 %). Per class (seed 0): motorcycle 90 %, cat 79 %, airplane 78 %, giraffe 78 %, bus 73 %, elephant 70 %, train 65 %, horse 60 %, dog 47 %, truck 46 %. The wrong class is usually a visually similar one, the same pairs as in the Q2.5 confusion matrix: dog → cat (36 %), truck → bus (17 %) / motorcycle (11 %) / train (8 %), cat → dog (14 %), train → truck (12 %). So the captioner's class errors largely inherit the encoder's classification errors.
- **Right class, wrong details:** counts ("a group of elephants" for one elephant), actions and scenes ("horse drawn carriage on a city street" for a plough in a field), colours; details like "chained", "jumping", "at dusk" are never produced (they are rare in the training captions and need fine visual information that a 256-d pooled vector of a 64×64 image does not keep).
- **Repetition:** 55/1,000 captions repeat a bigram ("a black and white photo of a black and white photo of …"), a typical failure of greedy RNN decoding once the state no longer tracks what was already said.
- **`<unk>`:** 0.43 % of the generated test tokens (seed 0) are `<unk>`; it can be generated because `<unk>` is a target in the training captions (2 % of the tokens).
- **Low diversity:** the 1,000 test captions (seed 0) use only 195 distinct words and 559 distinct bigrams (vocabulary: 2,189 words). Compared again in 3.4 (beam search).

## Q3.3 Encoders and class mention

Code: `tasks/q3_3/encoders.py` (3.3 a; log `results/q3_3/log.txt`, summary `results/q3_3/encoders.json`), `tasks/q3_3/class_mention.py` (3.3 b; log `results/q3_3/class_mention_log.txt`, `results/q3_3/class_mention.json`, figures `figures/q3_3/metrics_scatter.pdf`, `figures/q3_3/mention_per_class.pdf`).

### (a) MLP vs CNN encoder, frozen vs jointly trained

**Setup.** 2 encoders × 3 training modes × 3 seeds = 18 captioners, all with the decoder selected in Q3.2 (concat_image, GRU 512, dropout 0.3, Adam lr 1e-3, batch 64, ≤ 20 epochs, early stopping on val loss with patience 4). Same budget for all.
- Encoders: the selected CNN of Q2.5 (256-d image vector, 1.18 M params) and the selected MLP of Q2.2 (image vector = last hidden layer, 1024-d, 13.6 M params). Pretrained = the Part 2 classifier with the same seed.
- **frozen:** pretrained, not updated (frozen CNN = the Q3.2 runs, copied). **finetuned:** pretrained, trained jointly with the decoder, encoder lr 1e-4 (10× smaller than the decoder, to not destroy the pretrained features in the first steps). **scratch:** random init, trained jointly (lr 1e-3), only from the caption loss.
- Jointly trained runs on MPS are not bit-reproducible (small run-to-run differences); frozen runs are.

| Encoder | Mode | Params (trainable) | Val BLEU-4 | **Test BLEU-4** | Test class mention | Val ppl | Test ppl | Best epochs | s / epoch |
|---|---|---|---|---|---|---|---|---|---|
| CNN | frozen | 4.64 M (3.46 M) | 14.91 ± 0.72 | 14.02 ± 0.83 | **68.6 ± 0.6 %** | 11.94 ± 0.07 | 12.29 | 8, 5, 7 | 38 |
| CNN | finetuned | 4.64 M (4.64 M) | 15.29 ± 1.70 | 14.21 ± 1.37 | 60.9 ± 3.2 % | 12.64 ± 0.08 | 13.01 | 4, 4, 4 | 104 |
| CNN | scratch | 4.64 M (4.64 M) | 15.72 ± 1.56 | **15.51 ± 1.66** | 62.9 ± 3.6 % | **11.27 ± 0.12** | 11.51 | 6, 9, 4 | 232 |
| MLP | frozen | 17.69 M (4.05 M) | 10.39 ± 0.70 | 9.50 ± 0.66 | 30.4 ± 0.5 % | 13.59 ± 0.39 | 13.87 | 3, 4, 4 | 20 |
| MLP | finetuned | 17.69 M (17.69 M) | 9.90 ± 1.06 | 8.61 ± 0.92 | 26.0 ± 1.0 % | 12.46 ± 0.06 | 12.72 | 6, 7, 8 | 34 |
| MLP | scratch | 17.69 M (17.69 M) | 5.15 ± 2.15 | 5.34 ± 2.26 | 10.0 ± 0.0 % | 12.88 ± 0.02 | 13.05 | 7, 6, 7 | 126 |

(mean ± std over seeds 0, 1, 2; class mention with the original `CLASS_SYNONYMS`; classification test accuracy of the pretrained encoders: CNN 70.27 ± 1.00 %, MLP 29.67 ± 1.07 %.)

**Findings**
- **The encoder architecture matters most:** every CNN captioner beats every MLP captioner by ~5–10 BLEU points and ~30–50 points of class mention. Frozen CNN vs frozen MLP: +4.5 BLEU (2·SE ≈ 1.2), +38 points mention. The MLP has 4× more encoder parameters, so this is the inductive bias of Q2.4/2.5 (locality, weight sharing, pooling), not capacity.
- **CNN, frozen vs jointly trained: no significant BLEU difference.** Scratch has the highest mean (15.51) but the difference to frozen (+1.5) is below 2·SE ≈ 2.1; finetuned is equal to frozen (+0.2). Joint training makes runs much less stable (BLEU std 1.4–1.7 vs 0.8) and 3–6× slower per epoch.
- **But joint training lowers class mention** (frozen 68.6 % vs finetuned 60.9 % and scratch 62.9 %; differences 7.6 and 5.7 points, above 2·SE ≈ 3.7 / 4.2): the frozen encoder keeps the class information it was trained for; jointly trained encoders optimize the caption loss, which rewards generic wording ("a man riding …") as much as the class word.
- **Finetuning overfits fastest:** best epoch 4 in all seeds, val ppl 12.64 is *worse* than frozen (11.94). With all 4.6 M parameters trainable on 10 k images (50 k captions), the encoder adapts to the training captions; a smaller encoder lr or freezing the first blocks would be the next things to try.
- **The CNN trained from scratch learns a classifier from captions alone:** its linear-probe accuracy (69.5 ± 1.9 %, see b) equals the pretrained classifier (70.3 %). The captions name the class in 95 % of the cases, so the caption loss contains the class signal, plus extra signal (scene, actions). It also reaches the lowest perplexity (11.27) — but its captions are less diverse (203–388 distinct test captions vs 398–439 frozen).
- **MLP from scratch collapses completely:** all 1024 encoder features are 0 for every image (dead ReLUs: lr 1e-3 on a randomly initialized 13.6 M MLP), so the decoder ignores the image and generates **one fixed caption for all 1,000 test images** (seed 0: "a man riding a motorcycle with a woman on back of it .", seed 1: "a large elephant standing in a field …", seed 2: "a man is standing next to a motorcycle ."). Class mention exactly 10 % (= one class), BLEU depends on which sentence it picks (3.7–7.9). It is a pure language model, and still has test ppl 13.05 — **lower than the frozen MLP captioner (13.87)**: the frozen MLP features help BLEU (+4) but not perplexity, because the decoder overfits to the 1024-d features within 3–4 epochs.
- **MLP finetuned** has better perplexity than frozen (12.46 vs 13.59) but worse BLEU (8.6 vs 9.5) and the least diverse captions (95–146 distinct). Again: perplexity and BLEU disagree (see Q3.2).

### (b) Class mention accuracy and classification accuracy

**Classification accuracy of each encoder.** Only the pretrained encoders have a trained classification layer, so for all 18 encoders we additionally train a **linear probe** (softmax regression on the standardized, frozen image vectors of the training images; full-batch Adam lr 1e-2, wd 1e-4, 300 epochs, epoch selected by val accuracy; test accuracy reported). For the frozen encoders the probe matches the classifier (CNN 69.8 vs 70.3 %, MLP 29.3 vs 29.7 %), so it is a fair measure of "how much class information is in the image vector".

**Extended synonym list** (our extension, `metrics.py` unchanged): possessives ("dog's", one token in our tokenizer) and frequent class words in the training captions (biplane; retriever, husky, shepherd, collie, terrier; horseback; motorcyclist; subway, trolley, tram; firetruck, pickup). Human references: 95.4 → 96.3 % (test). **Generated captions: no change at all** (identical to the original list for all 18 runs) — the captioners never produce these rarer words, they always use the most frequent class word.

| Encoder | Mode | Test BLEU-4 | Class mention | Linear-probe acc | Classifier acc | Mention when classifier right / wrong |
|---|---|---|---|---|---|---|
| CNN | frozen | 14.02 ± 0.83 | 68.6 ± 0.6 % | 69.8 ± 1.1 % | 70.3 ± 1.0 % | 92.9 ± 1.3 % / 11.1 ± 4.4 % |
| CNN | finetuned | 14.21 ± 1.37 | 60.9 ± 3.2 % | 67.8 ± 1.0 % | – | – |
| CNN | scratch | 15.51 ± 1.66 | 62.9 ± 3.6 % | 69.5 ± 1.9 % | – | – |
| MLP | frozen | 9.50 ± 0.66 | 30.4 ± 0.5 % | 29.3 ± 0.8 % | 29.7 ± 1.1 % | 66.9 ± 1.7 % / 14.9 ± 2.2 % |
| MLP | finetuned | 8.61 ± 0.92 | 26.0 ± 1.0 % | 29.8 ± 1.6 % | – | – |
| MLP | scratch | 5.34 ± 2.26 | 10.0 ± 0.0 % | 10.0 ± 0.0 % | – | – |

**Does a better classifier make a better captioner?**
- **Across encoders: yes, clearly.** Over the 18 runs, Pearson r = 0.99 (probe acc vs class mention), 0.94 (probe acc vs BLEU-4), 0.95 (class mention vs BLEU-4); see `figures/q3_3/metrics_scatter.pdf`. For the frozen captioners, class mention ≈ classification accuracy (68.6 vs 70.3 %, 30.4 vs 29.7 %).
- **But this is a between-group effect.** Within the CNN captioners (9 runs, probe acc 66.7–71.2 %), the correlations are weak (probe–mention r = 0.65, probe–BLEU r = 0.39, mention–BLEU r = 0.30); within the MLP frozen/finetuned runs, probe–BLEU is even negative (−0.41). Once the encoder is "good enough" (~70 %), small differences in classification accuracy do not predict BLEU; the decoder, the training mode and seed noise matter as much. The jointly trained CNNs show this directly: same probe accuracy as frozen (68–70 %), lower mention, equal or higher BLEU.
- **Image level (frozen encoders): the caption follows the classifier.** When the CNN classifier is right, the caption names the right class in 92.9 % of the images; when it is wrong, only in 11.1 %. When it is wrong, the caption names the **classifier's (wrong) predicted class** in 73–80 % of the images (MLP: 43–55 %). So the frozen captioner essentially "reads out" the class from the image vector and writes a typical sentence for that class; its class errors are the classifier's errors (cf. the confusion pairs dog ↔ cat, truck → bus in Q3.2).
- **Per class** (`figures/q3_3/mention_per_class.pdf`): the CNN captioners are best on motorcycle, giraffe, cat, airplane (≥ 75 %) and worst on dog and truck (25–50 %), the classes the CNN classifier confuses most (Q2.5). Jointly trained CNNs lose most on truck (50 → 25/35 %), elephant (scratch), motorcycle and cat (finetuned). The MLP captioners name only motorcycle/cat/airplane reasonably often.

**What each metric captures**
- **Classification accuracy (classifier / linear probe):** how much *class* information the image vector contains, independent of language.
- **Class mention accuracy:** whether that information reaches the caption (one keyword); it ignores everything else in the sentence and gives no credit for synonyms outside the list (small effect here, see above).
- **BLEU-4:** n-gram overlap with 5 references, i.e. wording, scene and fluency; the class word is only one of ~11 tokens, so a caption with the right class but generic wording can still score low and a fluent caption of a frequent template can score well (the MLP-scratch "language model" gets BLEU 3.7–7.9 with 10 % mention).
- **Perplexity:** how probable the reference captions are under the model with teacher forcing; it measures the language model plus image conditioning, and can be good without image information (MLP scratch 13.05 < MLP frozen 13.87). None of these metrics alone says whether a caption is correct; they agree across very different encoders but disagree within a family.

## Q3.4 Decoding

Code: `Captioner.beam_search` in `models.py`; script `python -m tasks.q3_4.decoding` (log `results/q3_4/log.txt`, per-seed captions `results/q3_4/seed<k>.json`, summary + examples `results/q3_4/summary.json`, figure `figures/q3_4/tradeoffs.pdf`).

### (a) Beam search vs greedy

**Implementation** (batched over images, uses `Captioner.step`)
- Every image keeps K unfinished hypotheses with their summed log-probability log p(y₁…y_t | x). At each step all K·V extensions are scored, and the 2K best are taken (at least K of them do not end in `<eos>`).
- An extension that ends with `<eos>` and is among the K best goes to the image's list of **finished captions**. The beam is refilled with the best K extensions that do not end in `<eos>`.
- An image is done once it has K finished captions (fairseq-style stopping). Hypotheses still open after 25 tokens would also count as candidates, but this never happened.
- The returned caption is the finished one with the highest **log p(y | x)** or, with length normalization, the highest **log p(y | x) / L** (L = number of tokens including `<eos>`).
- All open hypotheses at step t have the same length t, so normalization cannot change the pruning inside the beam. It only changes **which finished caption wins** when finished captions have different lengths.
- Check: beam size 1 reproduces the stored greedy captions exactly (100 % identical on val and test, all seeds). For beam size 1 both rankings are identical (only one finished caption).
- Model: the selected captioner of Q3.2 (frozen CNN, concat_image), seeds 0, 1, 2; no retraining. Beam size 10 was added as an extra point.

**Results** (test, mean ± std over 3 seeds; validation in the log shows the same picture)

| Decoding | BLEU-4 | Mean length | Distinct words | Distinct bigrams | Distinct captions (of 1,000) | Class mention | Mean log p(y \| x) | Repeated bigram |
|---|---|---|---|---|---|---|---|---|
| greedy (beam 1) | 14.02 ± 0.83 | 10.96 | 187 ± 7 | 534 ± 23 | 415 | 68.6 % | −9.71 | 4.9 % |
| beam 3 | 15.46 ± 0.66 | 10.44 | 173 ± 6 | 470 ± 22 | 343 | 69.2 % | −8.00 | 3.3 % |
| beam 3, norm. | 15.36 ± 0.68 | 10.99 | 178 ± 9 | 489 ± 15 | 366 | 69.3 % | −8.23 | 4.6 % |
| beam 5 | **16.27 ± 0.41** | 10.27 | 165 ± 3 | 429 ± 22 | 287 | 69.5 % | −7.66 | 2.6 % |
| beam 5, norm. | 15.80 ± 0.29 | 11.01 | 174 ± 5 | 456 ± 32 | 317 | 69.2 % | −7.96 | 4.5 % |
| beam 10 (extra) | 16.79 ± 0.53 | 10.13 | 156 ± 6 | 383 ± 20 | 228 | 69.3 % | −7.38 | 2.6 % |
| beam 10, norm. (extra) | 16.24 ± 0.45 | 11.00 | 163 ± 5 | 408 ± 14 | 257 | 69.4 % | −7.69 | 4.8 % |

Validation BLEU-4: greedy 14.91, beam 3 16.89 / 16.64 (norm.), beam 5 17.15 / 16.81, beam 10 17.53 / 17.15.

**Findings and trade-offs**
- **Beam search finds more probable captions and gets higher BLEU.** Mean log p(y | x) rises from −9.71 (greedy) to −7.66 (beam 5) and −7.38 (beam 10). Test BLEU-4 rises by +2.25 with beam 5. The gain is the same for each seed (paired, per seed: +1.66, +2.45, +2.64 test; +1.86, +2.41, +2.45 val), so it is clearly not noise. Most of the gain comes from beam 3. Beam 5 → 10 adds only +0.5 on test, and on validation the per-seed differences are +0.98, −0.12 and +0.27, so the effect is inconsistent.
- **Greedy is often not the most probable caption:** with beam 3, only 20 % of the captions are identical to greedy, and with beam 10 only 12 %.
- **Without normalization, larger beams make captions shorter:** mean length 10.96 → 10.44 → 10.27 → 10.13 (beam 1 → 3 → 5 → 10). Captions of ≤ 8 tokens increase (seed 0: 1.2 % → 7.2 %). The final "." is dropped more often (seed 1: 3.7 % → 8.2 % of captions end without "."), e.g. "a large air plane on a run way" instead of "a large airplane sitting on top of an airport runway ." (beam 5 norm.). This is the bias toward short captions explained in (b).
- **Length normalization removes the length bias:** the mean length stays at 11.0, the same as greedy and close to the references (11.1). Its captions are slightly less probable (−7.96 vs −7.66 at beam 5) and score slightly lower BLEU (−0.1 to −0.6). The differences are small but have the same sign in 8 of 9 test (beam, seed) pairs. The shorter, safer unnormalized captions match the references a bit better in n-grams.
- **Diversity falls as the beam grows:** distinct bigrams drop 534 → 429 → 383 and distinct captions 415 → 287 → 228 (greedy → beam 5 → beam 10). Beam search moves every image toward the most probable, most generic phrasing of its class ("a motorcycle parked on the side of a road ." instead of "a man riding a motorcycle down a street next to a green tent ."). Length normalization keeps somewhat more diversity (+15–30 bigrams) because it allows longer, more specific endings.
- **So BLEU and diversity trade off.** Beam search optimizes the model probability, which rewards the frequent n-grams that BLEU also rewards. Specific details that are right in some images disappear, and wrong details are no longer produced as often. Repeated bigrams (the "a black and white photo of a black and white photo" failure) become less frequent without normalization (4.9 % → 2.6 %); with normalization the rate stays at ~4.6 %.
- **Class mention does not change (68.6–69.5 %):** the class is decided by the image vector (see 3.3 b), and decoding only changes the wording. Beam search cannot fix encoder errors (e.g. dog #8 is described as a "black and white cat" by every setting).
- **Cost:** K hypotheses per image, i.e. K times the decoder computation per step (here a few seconds per 1,000 images for all settings; our bookkeeping is in Python, so the time measurement is not meaningful).
- **Choice:** beam 5 with log-probability ranking. It has high validation BLEU, and beam 10 is not consistently better while losing more diversity. Length normalization is the better choice if caption length or diversity matter more than BLEU.

### (b) Theory

**(i) Greedy does not return the most probable caption.** Vocabulary {a, b, `<eos>`}.
- Step 1: p(a | x) = 0.6, p(b | x) = 0.4, p(`<eos>` | x) = 0.
- After "a": p(a | a) = 0.3, p(b | a) = 0.3, p(`<eos>` | a) = 0.4.
- After "b": p(`<eos>` | b) = 0.9, p(a | b) = p(b | b) = 0.05.
- Greedy picks a (0.6), then `<eos>` (0.4): p(a `<eos>` | x) = 0.6 · 0.4 = **0.24**.
- But p(b `<eos>` | x) = 0.4 · 0.9 = **0.36**.
- Every caption that starts with "a" has probability ≤ 0.6 · 0.4 = 0.24, so "b `<eos>`" is the most probable caption. Greedy commits to the locally best first word and never sees that "a" leads to a flat, uncertain continuation. A beam of size 2 would keep "b" and find it. Q3.4(a) shows the same thing in practice: greedy and beam 3 agree on only 20 % of the images.

**(ii) Prefix probabilities never increase.** By the chain rule,
p(y₁, …, y_{t+1} | x) = p(y₁, …, y_t | x) · p(y_{t+1} | y₁, …, y_t, x) ≤ p(y₁, …, y_t | x),
because the conditional probability is ≤ 1. Equivalently, log p(y₁…y_t | x) = Σ_{s≤t} log p(y_s | y_{<s}, x) is a sum of terms ≤ 0, so it can only decrease as t grows (strictly, unless the model is certain of the next token).

*Why exact maximization favors short captions:*
- A complete caption of length L is a prefix followed by `<eos>`, and every additional token multiplies its probability by another factor < 1.
- A long caption must therefore be "paid for" at every step. Ending early costs only the one factor p(`<eos>` | prefix).
- Unless the model gives `<eos>` a very small probability after short prefixes, a short caption ("a cat on a rug", or in the extreme an almost empty caption) has a higher total probability than a longer, more informative one, even if every word of the long one is likely (e.g. a 10-token caption whose tokens each have probability 0.8 has 0.8¹⁰ ≈ 0.107, less than a 6-token caption with 0.7 per token, 0.7⁶ ≈ 0.118, although every token of the long caption is more likely).
- The better the search approximates the arg max, the more short captions it finds. This is what (a) shows: mean length 10.96 → 10.13 from greedy to beam 10 without normalization, and the final "." gets dropped.

*Length normalization:*
- Ranking by log p(y | x) / L compares the **average log-probability per token**, i.e. the geometric mean of the per-token probabilities, (Π p(y_t | …))^{1/L}.
- The linear growth of the penalty with L is removed. A longer caption wins if its tokens are on average as likely as those of the shorter one.
- In the example above, the 10 × 0.8 caption (average log p = log 0.8) beats the 6 × 0.7 caption (log 0.7).
- In (a), normalization keeps the mean length at 11.0 for all beam sizes.
- It can over-correct, favoring long captions with many easy tokens. A softer variant divides by L^α (α ∈ [0, 1]) or by the GNMT penalty ((5 + L)/6)^α, which interpolates between no normalization (α = 0) and full normalization (α = 1).
