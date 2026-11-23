# Report Notes

Bullet points for the report, extended after each finished question. All numbers are accuracies in %, given as mean ± std (sample std, n = 3) over seeds 0, 1 and 2, unless stated otherwise.

## General (method, code)

- **Seeds:** 0, 1, 2. With `set_seed(seed)` from `train.py` the runs are deterministic: rerunning the same config with the same seed gives exactly the same numbers.
- **Model selection:** on validation only. Test is read only for the final numbers. At the end, `train()` restores the epoch with the best val accuracy.
- **Loss:** cross-entropy (`F.cross_entropy` on the logits, averaged over the batch). Rationale: it is the negative log-likelihood of the softmax model, i.e. maximum likelihood. The gradient w.r.t. the logits is `softmax(z) − onehot(y)`. In the linear case the loss is convex. MSE on the softmax output would have a vanishing gradient for confident wrong predictions.
- **Normalization:** per channel, with statistics from the training set only (`channel_stats`), applied unchanged to validation and test.
- **Skeleton code:** `data.py`, `train.py`, `metrics.py` and `load_data.py` are **unchanged**. Our own code is in `models.py`, `common.py` and `tasks/q2_*.py` (incl. the extra experiments `tasks/q2_1_lr_sweep.py` and `tasks/q2_2_activations.py`), `tasks/q2_3.py` and `tasks/q2_4.py`. Added to `common.py` for 2.4: `shift_images`, `mask_border`.
- **No numpy:** all statistics (mean, std, class counts) are computed with torch.
- **Reproduce:** from the project folder run `python -m tasks.q2_1`, `python -m tasks.q2_2`, etc. Results are organized per question: `results/q2_1/`, `results/q2_2/`, … (checkpoints likewise in `checkpoints/q2_x/`). Each run is saved as JSON in `results/q2_x/<run>/seed<k>.json` (config, per-epoch history, accuracies, parameter count).

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

**Learning-rate sweep** (`python -m tasks.q2_1_lr_sweep`, summary in `results/q2_1/lr_sweep.json`)
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
- The best epoch is in the middle of the run (14–30 of 50). After that the model overfits: train loss keeps falling (1.59 at epoch 50), val accuracy drops by 1–3 points (final val 29.3–30.5). Plot: `figures/q2_1_curves.pdf` (mean ± std over seeds).
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
- Script: `python -m tasks.q2_2_activations`; results in `results/q2_2/activations/<activation>/`, summary in `results/q2_2/activations.json`, log in `results/q2_2/activations_log.txt`. No checkpoints are saved for these runs.

**Files**
- Learning curves of the 3 seeds: `figures/q2_2_curves.pdf` (a dot marks the best epoch of each)
- Full search with ranking: `results/q2_2/summary.json`
- Individual runs: `results/q2_2/search/`, `results/q2_2/top/`, `results/q2_2/mlp_best/`
- Checkpoints for 2.3–2.5: `checkpoints/q2_2/mlp_best/seed{0,1,2}.pt`

## Q2.3 One-factor-at-a-time study

**Protocol** (`python -m tasks.q2_3`, summary in `results/q2_3/summary.json`, log in `results/q2_3/log.txt`)
- Base = the MLP selected in 2.2: [1024, 1024] ReLU, dropout 0.5, SGD lr 0.013 with momentum 0.9, weight decay 1e-4, batch size 128, no scheduler, normalized inputs, no augmentation.
- One factor changed at a time, everything else fixed. 7 factors (at least 3 required).
- **Same budget for every run: 50 epochs, no early stopping.** The best val epoch is restored at the end (as everywhere). The base run with this budget gives exactly the same numbers as `mlp_best` in 2.2 (all best epochs ≤ 12, so patience 8 never cut off a later improvement).
- 3 seeds per setting, 45 runs in total. Batch size was not varied: with a fixed number of epochs it changes the number of updates, which would break the "same budget" rule.
- Columns: **train** = train accuracy of the restored epoch (so it depends on where the best epoch is; this explains the large stds); **val** = best val accuracy within 50 epochs; **final val** = val accuracy after epoch 50 (best val − final val shows how much the model overfits when trained to the end); **final train loss** = mean train loss in epoch 50.
- **Meaningful difference rule:** Δval vs base, with standard error SE = sqrt(s_a²/3 + s_b²/3); marked * if |Δ| > 2·SE. With n = 3 this is a rough rule, not a proper test.
- Test is not used in the study; it is read only for the best model (per-class accuracy and confusion matrix below).
- Plots: `figures/q2_3_curves.pdf` (per factor: train loss (log) and val accuracy per epoch, mean ± std over seeds).

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

**Confusion matrix** (`figures/q2_3_confusion.pdf`, rows = true class, summed over 3 seeds, i.e. 300 test predictions per true class; raw counts in `results/q2_3/summary.json`)

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

Script: `python -m tasks.q2_4` (summary in `results/q2_4/summary.json`, log in `results/q2_4/log.txt`). Model: `mlp_best` (decision in 2.3). Helpers `shift_images` and `mask_border` are in `common.py`; `shift_accuracies` and `permutation_experiment` in `tasks/q2_4.py` are reused for the CNN in 2.5.

**(a) First-layer weights**
- Figures: `figures/q2_4_linear_templates.pdf` (10 class templates of the linear baseline, seed 0), `figures/q2_4_class_means.pdf` (mean training image per class), `figures/q2_4_mlp_weights_random.pdf` (8 random hidden units of `mlp_best`, seed 0), `figures/q2_4_mlp_weights_top.pdf` (the 8 units whose weights changed most). Each weight vector is shown as a 64×64 RGB image, grey = 0, scaled per image by its largest |w|. The weights act on normalized inputs, so colours are relative to the training mean colour.
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
- **Why the templates show no recognizable objects** (`figures/q2_4_class_means.pdf`): even the *mean training image* of each class shows no object, only a blurred colour/brightness layout (top row). Objects appear at every position, scale and pose, and the cluttered scene covers more of the image than the object, so averaging over 650 images per class leaves only the typical scene. Subtracting the mean image of all classes (bottom row, contrast-stretched) makes the differences visible: airplane = light-blue sky everywhere; bus/train/truck = blue sky at the top, dark road/rails at the bottom; elephant = dark central blob on green/sand; horse = brown blob on green/yellow; giraffe = green top (trees), yellow bottom (savanna); cat/dog = brown/beige indoor tones; motorcycle = dark central shape. A linear classifier has one weight per pixel position, so the best it can do is respond to these position-wise averages. (On centred, aligned data such as MNIST the templates do look like the class; here the data are not aligned.)
- **The learned templates match the class means quantitatively:** correlation between the template W − W_init of class i and the deviation image (class mean − overall mean, in normalized input space) of class j: mean 0.37 on the diagonal (0.25 train … 0.52 giraffe) vs −0.04 off the diagonal; **for all 10/10 templates the own class has the highest correlation.** So the linear model has essentially learned "class mean scene minus average scene", plus the fine-grained differences between classes. A template is still not a prototype: it is discriminative (positive where the class differs from the others, negative where it lacks something), which is why the correlation is moderate, not ~1.
- **MLP hidden units (W − W_init):** also smooth and blob-like (neighbour corr. 0.82), mostly global brightness/colour layouts over the whole image (top vs bottom, centre vs border, some colour-opponent blue vs brown units). No localized edge or part detectors as in a CNN: each unit sees all 12,288 pixels and nothing in the architecture prefers local patterns.
- Interpretation: the smoothness is learned from the data (natural images are smooth, so the gradient, which is a weighted sum of input images, is smooth), not built in. Part (c) confirms this.

**(b) Shifted test images** (no retraining; `figures/q2_4_shifts.pdf`)
- Test images shifted by k pixels to the right or down, the uncovered border filled with 0 (= training mean colour after normalization). Mean ± std over the 3 seeds.
- **Control:** shifting also removes a k-pixel strip of the image. To separate the effect of *misalignment* from the *information loss*, the same models were also evaluated on images where the same border is blanked but the image is **not** moved (`mask_border`).

| Shift (px) | 0 | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| MLP, right | 29.67 ± 1.07 | 29.10 ± 1.25 | 29.10 ± 0.82 | 29.07 ± 1.27 | 28.03 ± 0.90 | 26.10 ± 1.01 | 24.10 ± 0.85 |
| MLP, right, border only | 29.67 | 29.97 ± 0.72 | 29.67 ± 1.08 | 29.03 ± 0.46 | 28.97 ± 0.21 | 29.60 ± 0.26 | 28.97 ± 0.32 |
| MLP, down | 29.67 ± 1.07 | 29.67 ± 1.36 | 29.20 ± 0.75 | 28.20 ± 1.31 | 23.93 ± 1.12 | 21.20 ± 0.17 | 19.30 ± 1.01 |
| MLP, down, border only | 29.67 | 29.43 ± 1.70 | 29.13 ± 1.46 | 28.67 ± 1.01 | 28.80 ± 0.79 | 28.37 ± 1.33 | 27.60 ± 1.87 |
| Linear, right | 26.83 ± 0.83 | 26.47 ± 0.91 | 26.47 ± 1.06 | 25.20 ± 0.53 | 24.20 ± 1.51 | 23.67 ± 0.31 | 21.43 ± 0.61 |
| Linear, down | 26.83 ± 0.83 | 27.17 ± 0.67 | 27.43 ± 0.71 | 24.80 ± 0.95 | 22.17 ± 0.06 | 19.60 ± 1.44 | 18.43 ± 0.31 |

(Linear border-only control at 8–16 px: 25.1–25.9 for right, 24.1–25.4 for down; full numbers incl. 3 and 6 px in the JSON/log.)

- **Blanking the border alone costs almost nothing** (MLP: ≤ 1 point for right shifts, ≤ 2.1 points for down shifts, up to 16 px). So the drop under shifting is due to **misalignment**, not to the lost strip: at 16 px, shifting costs −4.9 (right) and −8.3 (down) points more than blanking the same strip.
- **Small shifts (1–4 px) barely hurt** (≤ 1.5 points, within the seed std). This is *not* a built-in invariance: it follows from the learned weights being smooth (neighbour corr. 0.82). A 1-pixel shift barely changes the dot product of a smooth template with the image. Accuracy degrades steadily with the shift size, as expected for a model whose weights are tied to absolute pixel positions.
- **Vertical shifts hurt about twice as much as horizontal ones.** Likely reason: the classes are recognized from the global vertical layout of the scene (sky at the top, ground/grass at the bottom, horizon height), which the templates in (a) show; horizontal position is more variable in photos (objects appear left or right), so the learned templates are more tolerant horizontally.
- The linear baseline behaves the same way (at 16 px: −5.4 right, −8.4 down vs. unshifted; −3.8 and −5.7 vs. its border-only control), so this is a property of "one weight per pixel position", not of the hidden layers.
- The MLP stays far above chance (10 %) even at 16 px: much of what it uses (global colour statistics) survives a shift.

**(c) Fixed random pixel permutation** (retrained from scratch)
- One fixed random permutation of the 64×64 pixel positions (`pixel_permutation(seed=0)`, the same for all 3 channels) applied to train, val and test. The `mlp_best` configuration retrained with the same protocol (3 seeds, at most 50 epochs, early stopping with patience 8 on val). Per-channel normalization statistics are unchanged by a permutation. Results in `results/q2_4/mlp_perm/`, checkpoints in `checkpoints/q2_4/mlp_perm/`.

| Pixels | Train | Val | Test | Best epochs |
|---|---|---|---|---|
| original (`mlp_best`) | 53.49 ± 8.41 | 36.50 ± 0.61 | 29.67 ± 1.07 | 12, 6, 8 |
| permuted (`mlp_perm`) | 51.62 ± 11.64 | 35.60 ± 0.80 | 30.20 ± 1.15 | 12, 8, 4 |

- **No meaningful difference** (val −0.90 ± 0.58, test +0.53 ± 0.89; both < 2·SE). The permuted images are unrecognizable for a human, but the MLP learns them just as well.
- **Why, exactly:** the first layer computes W x. For a permutation matrix P, W (P x) = (W P) x, so an MLP with weights W P on the original images computes the same function as an MLP with weights W on permuted images. The initialization is i.i.d. over the inputs (its distribution does not change if the columns of W are permuted), and SGD updates commute with the permutation. So training on permuted pixels is the same learning problem up to relabelling the input dimensions: the MLP is **invariant to the order of its inputs as a learning algorithm**. The small differences come only from different random draws (the init is not permuted along with the data).
- `figures/q2_4_perm_weights.pdf`: the learned change W − W_init of the permuted model looks like noise as learned (neighbour corr. 0.033), but after mapping it back to the original pixel layout it is smooth again (0.819, the same as the original MLP's 0.818) and shows the same kind of colour blobs. The model learned the same thing, just in a scrambled coordinate system.

**What this says about the MLP's assumptions (for the report)**
- An MLP makes **no assumption about the spatial structure of images**: it does not know which pixels are neighbours (c), and it has no weight sharing across positions, so a pattern learned at one position does not transfer to another (b). Every pixel position is an independent input feature.
- What looks like structure in its weights (smooth templates, robustness to 1–4 px shifts) is learned from the data, and it costs data: with 6500 images the MLP can only learn coarse global colour/layout templates (a), which explains the confusions between classes with similar backgrounds (2.3) and the strong overfitting (12.6M first-layer parameters, one per pixel and unit).
- A CNN builds in exactly the two missing assumptions, locality (small kernels) and translation equivariance (weight sharing). The prediction for 2.5: the CNN should handle shifts much better and should lose a lot of accuracy on permuted pixels, because the permutation destroys the local neighbourhoods its kernels rely on.
