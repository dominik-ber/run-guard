# Model comparison: XGBoost vs MLP

Run on 2026-11-11 · code: `RunGuard/ml_logic/compare_models.py`, `RunGuard/ml_logic/dl_model.py` (branch `feature/dl_models`, based on master `9368986`)

## Question

Does a small neural network (Keras MLP) predict injuries better than the current XGBoost model? And what does `scale_pos_weight` do to XGBoost?

## Setup

- **Data:** `day_approach_maskedID_timeseries.csv`, the 28 contract features (7 days × 4 features).
- **Split:** `train_test_data_prep`, which splits by athlete. Train has 60 athletes (34,203 rows, 453 injuries) and test has 14 athletes (8,563 rows, 130 injuries, injury rate **1.52%**). Every model uses the same test athletes.
- **Metrics:** `evaluate_injury_risk` from `model.py` gives PR-AUC, ROC-AUC and recall at 10% false alarm rate. Two more columns show the quality of the probabilities themselves:
  - `log_loss`: lower is better. Predicting the train injury rate (1.32%) for every row gives **0.0788** on test.
  - `mean_prob`: the mean predicted probability. It should be close to the real 1.52%.
- **Seeds:** each model is trained with 5 seeds (0–4). The table shows mean ± std over the seeds.

| Model | Details |
|---|---|
| no-skill | 0.5 for every row (lower bound) |
| xgboost | `initialize_xgboost_model` as in `model.py`, `scale_pos_weight = neg/pos ≈ 74.5`; trained on all 60 train athletes |
| xgboost (no weight) | Same, with `scale_pos_weight = 1` |
| mlp | `Normalization → Dense(32) → Dropout(0.3) → Dense(16) → Dropout(0.3) → Dense(1, sigmoid)`, L2 = 1e-4, about 1.5k weights. Adam (lr 1e-3), batch 256, no class weights. Early stopping on validation PR-AUC (patience 15, best weights restored). The validation set is about 1/7 of the train athletes and is resampled for every seed, so the MLP trains on about 53 athletes. |

## Results (test set, mean ± std over 5 seeds)

| Model | PR-AUC | ROC-AUC | Recall @ 10% FAR | Log-loss | Mean prob. |
|---|---|---|---|---|---|
| no-skill | 0.0152 ± 0.0000 | 0.500 ± 0.000 | 0.000* | 0.693 | 50.0% |
| xgboost | **0.0227** ± 0.0014 | 0.583 ± 0.008 | **0.180** ± 0.021 | 0.314 ± 0.007 | 23.4% ± 0.5% |
| xgboost (no weight) | 0.0220 ± 0.0004 | **0.614** ± 0.012 | **0.180** ± 0.020 | **0.0800** ± 0.0005 | 1.34% ± 0.02% |
| mlp | 0.0209 ± 0.0029 | 0.599 ± 0.047 | 0.145 ± 0.035 | 0.0810 ± 0.0045 | 1.52% ± 0.73% |

\* A constant score has only two points on the ROC curve, so the recall at 10% FAR is 0 by construction.

ROC-AUC per seed:

| Seed | xgboost | xgboost (no weight) | mlp |
|---|---|---|---|
| 0 | 0.591 | 0.619 | 0.522 |
| 1 | 0.588 | 0.597 | 0.621 |
| 2 | 0.571 | 0.625 | 0.608 |
| 3 | 0.589 | 0.608 | 0.648 |
| 4 | 0.579 | 0.621 | 0.598 |

## Findings

1. **The signal is weak for every model.** The best PR-AUC is about 0.022, against 0.015 for random guessing (about 1.5× better), and ROC-AUC is 0.58–0.61.
2. **The MLP is no better than XGBoost, and it is less stable.** Its mean PR-AUC and recall are the lowest, and its ROC-AUC varies from 0.52 to 0.65 depending on the seed. Part of that comes from which 7 athletes land in validation and steer early stopping. At this data size the MLP gives no reason to replace XGBoost.
3. **`scale_pos_weight` hurts XGBoost more than it helps:**
   - **Probabilities:** with the weight the model predicts 23% on average when the real rate is 1.5%, and its log-loss is 0.31. Without the weight it predicts 1.3% and its log-loss is 0.080.
   - **Ranking:** without the weight ROC-AUC is higher (0.614 vs 0.583, consistent over all 5 seeds). PR-AUC (0.0220 vs 0.0227) and recall (0.180 vs 0.180) stay the same within noise.
4. **No model beats a constant on log-loss.** The best value is 0.0800, against 0.0788 for predicting 1.32% for everyone. The models rank athletes somewhat better than chance, but their probabilities for individual days are no more accurate than the average rate. The UI should not present the output as a precise probability without calibration.

## Suggested next steps

- **Set `scale_pos_weight = 1` in `model.py` (team decision).** That gives realistic probabilities and a better ROC-AUC with no clear loss on the other metrics. It also fixes the "12% injury risk" the UI currently shows for a typical user. For Dominik's Strava export the weighted model gives 11.75%. XGBoost without the weight gives 0.26–0.43% over 5 seeds, below the average daily rate of 1.5%.
- **Features before models.** Features normalized per athlete (load compared with the athlete's own usual load) and acute:chronic load ratios are more likely to help than a different model class.
- **More stable evaluation.** Use all 5 folds of `StratifiedGroupKFold` instead of a single test fold; 14 test athletes make one split noisy.
- **1D-CNN (Trello card): expectations are low** after this result. Run it with the same script (add a `run_cnn`) so it can be compared directly.
- **Calibration** of the chosen model before the UI shows percentages, or show relative risk instead ("2× your usual risk").

## Reproduce

```bash
# from the repo root, with raw_data/day_approach_maskedID_timeseries.csv in place
pip install tensorflow xgboost
python -m RunGuard.ml_logic.compare_models    # about 1.5 minutes on CPU
```

Versions used: Python 3.10.6, TensorFlow 2.21.0 (Keras 3), XGBoost 3.2.0, scikit-learn 1.7.2.
