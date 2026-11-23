# Report Notes

Bullet points for the report, extended after each finished question. All numbers are accuracies in %, given as mean ± std (sample std, n = 3) over seeds 0, 1 and 2, unless stated otherwise.

## General (method, code)

- **Seeds:** 0, 1, 2. With `set_seed(seed)` from `train.py` the runs are deterministic: rerunning the same config with the same seed gives exactly the same numbers.
- **Model selection:** on validation only. Test is read only for the final numbers. At the end, `train()` restores the epoch with the best val accuracy.
- **Loss:** cross-entropy (`F.cross_entropy` on the logits, averaged over the batch). Rationale: it is the negative log-likelihood of the softmax model, i.e. maximum likelihood. The gradient w.r.t. the logits is `softmax(z) − onehot(y)`. In the linear case the loss is convex. MSE on the softmax output would have a vanishing gradient for confident wrong predictions.
- **Normalization:** per channel, with statistics from the training set only (`channel_stats`), applied unchanged to validation and test.
- **Skeleton code:** `data.py`, `train.py`, `metrics.py` and `load_data.py` are **unchanged**. Our own code is in `models.py`, `common.py` and `tasks/q2_*.py` (incl. the extra experiments `tasks/q2_1_lr_sweep.py` and `tasks/q2_2_activations.py`).
- **No numpy:** all statistics (mean, std, class counts) are computed with torch.
- **Reproduce:** from the project folder run `python -m tasks.q2_1`, `python -m tasks.q2_2`, etc. Each run is saved as JSON in `results/<name>/seed<k>.json` (config, per-epoch history, accuracies, parameter count).

## Q2.1 Data and linear baseline

**Data** (source: `results/q2_1_data_stats.json`)

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

**Learning-rate sweep** (`python -m tasks.q2_1_lr_sweep`, summary in `results/q2_1_lr_sweep.json`)
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
- The first-guess runs with lr 0.001 are kept in `results/linear_lr/0.001/` (identical to the earlier numbers 26.70 ± 0.53 on test, since the runs are deterministic).
- The seed 0 weights are in `checkpoints/linear/seed0.pt` and are needed for the class templates in 2.4.

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
- Setup: the chosen configuration is kept fixed, only the activation is swapped, 3 seeds each. ReLU is the chosen model itself (`results/mlp_best/`).
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
- Script: `python -m tasks.q2_2_activations`; results in `results/mlp_act_<activation>/`, summary in `results/q2_2_activations.json`, log in `results/q2_2_activations_log.txt`. No checkpoints are saved for these runs.

**Files**
- Learning curves of the 3 seeds: `figures/q2_2_curves.pdf` (a dot marks the best epoch of each)
- Full search with ranking: `results/q2_2_summary.json`
- Individual runs: `results/mlp_search/`, `results/mlp_top/`, `results/mlp_best/`
- Checkpoints for 2.3–2.5: `checkpoints/mlp_best/seed{0,1,2}.pt`
