# Context

這份檔案是 `codex_notes` 的唯一入口。它只保留：

- 開發原則
- 程式架構
- related works survey 入口
- 依開發日期排序的實驗結論
- 要查細節時應該 reference 哪個日期檔、哪個章節

實驗細節全部放在 `codex_notes/dev_log/YYYY-MM-DD.md`。
related works 與讀過的 paper 筆記放在 `codex_notes/related_work.md`；
原始 paper PDF 放在 `codex_notes/surveyed_papers/`。之後若使用者提供新 paper，
要在 `related_work.md` 補上論文核心解釋、與本研究相同點、以及本研究可主張的 novelty。

## Development Principles

- 優先保留原始 ACSIncome 實驗邏輯，不要為了 synthetic 驗證破壞主流程。
- 若需要擴充功能，優先整合進共用核心，而不是複製一套平行版本。
- synthetic 的用途是驗證 loss 設計，不是取代 ACSIncome 主實驗。
- benchmark 數值不必自動落成 JSON；由使用者自行整理即可。
- SHAP plots 要保留，因為它們是 feature utilization 的直接證據。
- `run_synthetic_ood.py` 的風格要和 ACS runner 一致：
  - 常數集中在檔案頂部
  - 直接呼叫 `main(...)`
  - 直接 `print` 結果
- synthetic 和 ACSIncome 共用 `src/main.py` / `src/trainer.py`。
- 若需要 dataset-specific 行為，應透過參數切換，而不是複製核心程式。

## Related Work Notes

- `codex_notes/related_work.md`
  - 用來記錄已 survey 的 related works。
  - 每篇 paper 至少包含：
    - 論文核心解釋
    - 與本研究的相同點
    - 本研究相對可主張的 novelty
    - 對後續實驗或寫作 positioning 的啟發
- `codex_notes/surveyed_papers/`
  - 存放使用者提供或已讀過的 paper PDF。
  - 已紀錄：
    - `Large Language Models as Attribution Regularizers for Efficient Model Training - 2502.20268v3.pdf`

## Architecture

### Entrypoints

- `run_acs_fitce.py`
  - ACS `FeatureImportanceTargetCELoss` runner
  - dataset metadata comes from `acs_tasks/config.py`
  - loss / ranking hyperparameters remain local runner settings
- `run_acs_mlp_ce.py`
  - ACS MLP + `CrossEntropyLoss` baseline runner
  - uses the same ACS task metadata, split/cache rules, seeds, and validation-best
    checkpoint selection as `run_acs_fitce.py`
- `run_acs_laat.py`
  - ACS LAAT-style LLM attribution regularizer baseline
  - uses MLP + gradient-attribution MSE alignment to LLM feature scores
  - intentionally does not use the `importance_scale="train_std"` scaling from
    `FeatureImportanceTargetCELoss`
- `run_acs_llm_select.py`
  - ACS `LLM-Select` feature-removal baseline
  - selects retained features from LLM ranking artifacts, then trains MLP CE
  - `score_threshold` is restricted to `score` / `score_all`; other ranking methods
    use top-p selection because ACS tasks have different feature counts
- `run_acs_llm_lasso.py`
  - ACS `LLM-Lasso` baseline
  - uses score-based ranking artifacts only
  - transforms LLM scores into weighted L1 logistic penalty factors and selects
    `eta` / `C` by validation accuracy
- `run_acs_linear.py`
  - ACS sklearn `LogisticRegression` baseline runner
  - uses the same ACS task metadata and split/cache rules
  - fits `StandardScaler` on train only and selects `C` by validation accuracy
- `run_synthetic_ood.py`
  - synthetic sanity-check / loss 設計驗證入口
- `archive/acsincome/run_acsincome_gpt_groups.py`
  - archived ACSIncome GPT coarse grouping 實驗 helper
  - 讀取 `archive/acsincome/gpt_acsincome.json`
  - 可用 `--skip-analysis` 避免 plotting workflow 阻塞 screening
- `archive/scripts/`
  - archived one-off ACS benchmarking / aggregation / train-state sweep utilities
  - kept only for historical reproduction of dev-log experiments

### Shared Core

- `src/main.py`
  - dataset loader dispatch
  - model dispatch
  - loss dispatch
  - repeat / seed control
  - returns each aggregate metric as `(mean, ci_range)`, where `ci_range` is the
    95% Student-t confidence-interval half-width over repeats
  - shared experiment knobs:
    - `MODEL_NAME`: `"mlp"` / `"linear"`
    - `LOSS_NAME`: `"cross_entropy"` / `"feature_grad_ce"` / `"first_layer_weight_ce"` / `"feature_importance_target_ce"`
    - `LOSS_KWARGS`
    - `MODEL_SEEDS`
    - `MAX_EPOCHS`
    - `DATASET_CONFIG`
    - `PLOT_TEST_SHAP`
- `src/trainer.py`
  - training loop
  - early stopping by validation accuracy
  - OOD testing
  - optional SHAP computation
- `src/loss.py`
  - `CrossEntropyCELoss`
  - `FeatureGradCELoss`
  - `FirstLayerWeightCELoss`
  - `LLMAttributionAlignedCELoss`
  - `FeatureImportanceTargetCELoss`
- `src/mlp.py`
  - `MLP`
  - `Linear`
- `src/utils.py`
  - seeding
  - training curves
  - SHAP plots
  - accuracy delta plots

### Datasets

- `acs_tasks/dataset.py`
  - shared ACS task loader
  - feature removal
  - train/val split
  - optional train-only balanced oversampling via `RESAMPLING`
  - optional train-stat standardization via `standardize`
- `acs_tasks/<dataset>/data/`
  - regenerated local cache for processed ACS train/val/test tensors
  - cache path is derived from data-changing factors:
    `rm_<removed-feature-indices>__rs<0/1>__std<0/1>`
  - example: `rm_9-10-14__rs0__std0`
  - `VAL_RATE` is intentionally omitted because ACS runs use the fixed `0.2`
    train/validation split ratio
  - old ad hoc cache directories were deleted on 2026-05-31
- `synthetic_ood/dataset.py`
  - synthetic OOD data generation
  - controllable spurious correlation shift

### Outputs

- `acsincome/plots/curves`
- `acsincome/plots/shap`
- `acsincome/plots/accdelta`
- `synthetic_ood/plots/curves`
- `synthetic_ood/plots/shap`
- Raw experiment logs under `acs_tasks/*logs*` are not retained once their detailed
  Accuracy/F1 tables have been recorded in `codex_notes/dev_log/`.
- Raw result artifacts under `acs_tasks/state_sweep/` and
  `acs_tasks/metric_benchmarks/` were also removed after their results were recorded
  in `codex_notes/dev_log/` and `README.md`.

## Dated Conclusions And References

### 2026-05-31

Conclusion:

- Archived the previous README `Dataset Conclusions` tables and
  `FeatureImportanceTargetCELoss` settings before replacing the public README section
  with a broader baseline template.
- README now has empty Accuracy/F1 templates for `MLP CrossEntropyLoss`,
  `FeatureImportanceTargetCELoss`, `LLM Attribution Regularizer`, `LLM-Select`,
  `LLM-Lasso`, `LogisticRegression`, `XGBoost`, `CatBoost`, `RandomForest`,
  `SVM`, and `TabPFN`.
- The `GPT Ranking Token Cost` table remains in README because the new ACS FITCE
  experiments still reuse those ranking artifacts.

Reference details:

- `codex_notes/dev_log/2026-05-31.md`
- section: `README Dataset Conclusions Archive`

### 2026-05-27

Conclusion:

- Completed `FeatureImportanceTargetCELoss` `TARGET_POWER` sweep with
  `score_all_direct`, `SUPPRESS_BOUND=0.0`, `LOSS_LAMBDA=16.0`,
  `LOSS_ALPHA=0.75`, `suppress_scale=1.0`, `REG_SCALE=1.0`, and `REPEAT=5`.
- `TARGET_POWER=1.0` remained the best target-shaping setting by the selection
  score `Acc OOD MEAN + Acc OOD WORST + F1 OOD MEAN + F1 OOD WORST`:
  - score: `2.5611`
  - Acc OOD MEAN: `0.7342`
  - Acc OOD WORST: `0.6802`
  - F1 OOD MEAN: `0.6270`
  - F1 OOD WORST: `0.5197`
- Completed `REG_SCALE` sweep plus high sweep. The combined best setting is
  `REG_SCALE=2.0`:
  - score: `2.5656`
  - Acc OOD MEAN: `0.7342`
  - Acc OOD WORST: `0.6790`
  - F1 OOD MEAN: `0.6284`
  - F1 OOD WORST: `0.5240`
- Higher `REG_SCALE` values did not beat `2.0`; the best high-sweep value was
  `REG_SCALE=2.5` with score `2.5523`.
- Runner defaults should keep `TARGET_POWER=1.0` and `REG_SCALE=2.0`.

Reference details:

- `codex_notes/dev_log/2026-05-27.md`
- sections:
  - `FeatureImportanceTargetCELoss TARGET_POWER Sweep`
  - `FeatureImportanceTargetCELoss REG_SCALE Sweep`
  - `FeatureImportanceTargetCELoss REG_SCALE High Sweep`

### 2026-05-24

Conclusion:

- Added `SUPPRESS_BOUND` support to ranking-to-feature-weight conversion so low
  `FEATURE_LOSS_WEIGHTS` can be set to `0.0` and activate
  `FeatureImportanceTargetCELoss` suppressed-feature penalty.
- Corrected the threshold grid to match `score_all_direct` score scale:
  `SUPPRESS_BOUND=0.0, 0.1, ..., 1.0`.
- In the suppress-only sweep, `SUPPRESS_BOUND=0.0` ranked first:
  - score: `2.5611`
  - Acc OOD MEAN: `0.7342`
  - Acc OOD WORST: `0.6802`
  - F1 OOD MEAN: `0.6270`
  - F1 OOD WORST: `0.5197`
- Directly removing zero-weight features did not improve over suppression. It ties at
  `SUPPRESS_BOUND=0.0` because no additional features are removed, then generally
  degrades as threshold increases; all-feature-removed cases are recorded as
  `0.0000` ACC/F1.
- Best setting after this round remains no thresholding: `SUPPRESS_BOUND=0.0`.

Reference details:

- `codex_notes/dev_log/2026-05-24.md`
- sections:
  - `FeatureImportanceTargetCELoss Suppress Bound Sweep`
  - `Suppress Bound Sweep Results`
  - `Suppress vs Direct Remove Sweep Results`

### 2026-05-20

Conclusion:

- Completed `FeatureImportanceTargetCELoss` `(LOSS_LAMBDA, LOSS_ALPHA)` grid search
  over all 8 ACS tasks with `score_all_direct`, `REG_SCALE=1.0`, `target_power=1.0`,
  `importance_scale="train_std"`, and `REPEAT=5`.
- The runner maps:
  - `grad_scale = LOSS_LAMBDA * LOSS_ALPHA`
  - `weight_scale = LOSS_LAMBDA * (1.0 - LOSS_ALPHA)`
- Selection score is
  `Acc OOD MEAN + Acc OOD WORST + F1 OOD MEAN + F1 OOD WORST`.
- Best grid setting:
  - `LOSS_LAMBDA=16`
  - `LOSS_ALPHA=0.75`
  - `grad_scale=12.0`
  - `weight_scale=4.0`
  - score: `2.5611`
  - Acc OOD MEAN: `0.7342`
  - Acc OOD WORST: `0.6802`
  - F1 OOD MEAN: `0.6270`
  - F1 OOD WORST: `0.5197`
- This setting became the default for later suppress-bound, target-power, and
  REG_SCALE sweeps.

Reference details:

- `codex_notes/dev_log/2026-05-20.md`
- section: `FeatureImportanceTargetCELoss Lambda/Alpha Grid Search`

### 2026-05-15

Conclusion:

- Added shared ranking-to-`FEATURE_LOSS_WEIGHTS` helper in `src/ranking.py`.
- The helper supports `rank`, `score`, `score_all`, and `seq` ranking artifacts.
- Weight mapping is rank-position based: for `n` features, rank 1 gets `n` and the
  last feature gets `1`.
- The shared ACS runner uses the helper with `FEATURE_RANKING_METHOD = "score_all"`,
  `LOSS_NAME = "feature_importance_target_ce"`, `REG_SCALE = 1.0`, and the shared
  FeatureImportanceTargetCELoss kwargs.

Reference details:

- `codex_notes/dev_log/2026-05-15.md`
- implementation reference:
  - `src/ranking.py`
  - `run_acs_fitce.py`
  - `acs_tasks/config.py`

### 2026-05-14

Conclusion:

- Added GPT ranking method `score_all`.
- `score_all` asks GPT to score all task features in each call, repeats 5 times, and
  ranks features by mean score.
- Generated `gpt-5.5_score_all_feature_ranking.json` for all 8 ACS tasks.
- README `GPT Ranking Token Cost` table now includes `score_all`; each ACS task used
  5 calls and substantially fewer total tokens than per-feature `score`.

Reference details:

- `codex_notes/dev_log/2026-05-14.md`
- implementation reference:
  - `rank_features.py`
  - `acs_tasks/<dataset>/rankings/gpt-5.5_score_all_feature_ranking.json`
  - `README.md`

### 2026-05-11

Conclusion:

- ACS metric tables now use the new Accuracy/F1 format with `ID`, `OOD MEAN`,
  `OOD WORST`, and `OOD STD`.
- `CrossEntropyLoss fix dataset issue` has been rerun for all 8 ACS tasks on GPU.
- Benchmark JSON files now record CUDA device metadata; partial GPU runs preserve a
  `devices` list after aggregation.
- README Dataset Conclusions removed the old ACC-only HTML table and now reports the
  new grouped Accuracy and F1 tables.

Reference details:

- `codex_notes/dev_log/2026-05-11.md`
- implementation reference:
  - `archive/scripts/run_acs_metric_benchmark.py`
  - `archive/scripts/aggregate_acs_metric_partials.py`
  - `README.md`

### 2026-05-10

Conclusion:

- Automated GPT feature ranking for `FeatureImportanceTargetCELoss` across all ACS task
  runners.
- `README.md` and the ACS runner can intentionally contain different ACSIncome GPT
  rankings: the runner ranking came from a later manual web-ChatGPT query and is not a
  constraint on the automation target.
- `rank_features.py` now imports ACS dataset/feature metadata from `acs_tasks/config.py`,
  builds a strict JSON ranking prompt, calls `gpt-5.5`, validates the response, and writes
  reusable ranking JSON files under `acs_tasks/<dataset>/rankings/`.
- The ranking JSON stores only the ranking result and call metadata. It intentionally does
  not store derived feature weights or repeated train/test split metadata; ranking-to-weight
  conversion remains a separate design question.
- After reading LLM-Select, `rank_features.py` supports `rank`, `score`, and `seq`
  methods. ACSIncome has been tested with all three, and JSON outputs now include
  input/output token counts for cost comparison.
- Metrics were expanded to track both Accuracy and binary F1 with sub-metrics `ID`,
  `OOD MEAN`, `OOD WORST`, and `OOD STD`. Synthetic verification preserved the previous
  OOD accuracy benchmark and added initial Accuracy/F1 metric tables to `README.md`.

Reference details:

- `codex_notes/dev_log/2026-05-10.md`
- implementation reference:
  - `rank_features.py`
  - `run_acs_fitce.py`
  - `acs_tasks/config.py`
### 2026-05-04

Conclusion:

- Added ACS Folktables task templates:
  - `ACSIncome`
  - `ACSEmployment`
  - `ACSHealthInsurance`
  - `ACSPublicCoverage`
  - `ACSTravelTime`
  - `ACSMobility`
  - `ACSEmploymentFiltered`
  - `ACSIncomePovertyRatio`
- Added shared loader wiring in `acs_tasks/dataset.py` and registered new dataset keys in
  `src/main.py`.
- Merged `ACSIncome` into `acs_tasks`, removed the old `acsincome/dataset.py` loader, and
  fixed raw data loading to `acs_tasks/raw/data/2018/1-Year`.
- ACS cache and plot artifacts now route to:
  - `acs_tasks/{task_name}/data/{train,val,tests}.pkl`
  - `acs_tasks/{task_name}/plots/`
- Added `run_*.py` templates using `cross_entropy`, class reweighting, PR train/validation,
  and the existing ACSIncome held-out state list as placeholders.
- No experiments were run.

Reference details:

- `codex_notes/dev_log/2026-05-04.md`

Additional conclusion:

- Completed ACS train/validation state selection under MLP without BatchNorm + plain CE
  with no balancing.
- Confirmed fixed states:
  - `acsemployment`: `SD`
  - `acsemploymentfiltered`: `SD`
  - `acshealthinsurance`: `MN`
  - `acsincome`: `PR`
  - `acsincomepovertyratio`: `HI`
  - `acsmobility`: `AK`
  - `acspubliccoverage`: `CA`
  - `acstraveltime`: `AZ`
- Full result summary is recorded in `codex_notes/dev_log/2026-05-04.md`; the
  former raw output directory `acs_tasks/state_sweep/` was deleted on 2026-05-31.

Additional conclusion:

- Diagnosed the very low OOD WORST values on several binary ACS tasks by adding
  `BatchNorm1d` back to the MLP.
- 5-seed OOD WORST before/after BatchNorm for completed tasks:
  - `acsemployment`: 0.7061 -> 0.7649
  - `acsemploymentfiltered`: 0.6474 -> 0.7062
  - `acshealthinsurance`: 0.2214 -> 0.7829
  - `acsincomepovertyratio`: 0.3964 -> 0.6330
  - `acsmobility`: 0.6385 -> 0.6624
- Short screening also showed large BN gains on the two slow tasks:
  - `acspubliccoverage`: 0.1339 -> 0.6288
  - `acstraveltime`: 0.2255 -> 0.4635
- Conclusion: `BatchNorm1d` is now part of the default MLP architecture. The no-BN MLP is
  only a diagnostic baseline.
- Next ACS task baseline question: with BN fixed as default, decide per task whether
  class reweighting and removal of raw state-specific code features should become default
  future settings.
- Detailed results: `codex_notes/dev_log/2026-05-04.md`, section
  `ACS Plain CE Baseline: BatchNorm Diagnosis`.

Additional conclusion:

- Completed 5-seed same-budget ACS baseline ablations for reweighting and removal of raw
  state-specific code features under default BN MLP:
  - `REPEAT=5`
  - `MAX_EPOCHS=800`
  - `PATIENCE=100`
- Reweighting before/after OOD WORST:
  - `acsemployment`: 0.7643 -> 0.7623; do not default reweighting
  - `acsemploymentfiltered`: 0.7080 -> 0.6699; do not default reweighting
  - `acshealthinsurance`: 0.7739 -> 0.7035; do not default reweighting
  - `acsincomepovertyratio`: 0.6216 -> 0.6223; default reweighting, but weak effect
  - `acsmobility`: 0.6489 -> 0.6071; do not default reweighting
  - `acspubliccoverage`: 0.5915 -> 0.4959; do not default reweighting
  - `acstraveltime`: 0.4658 -> 0.4623; do not default reweighting
- Remove state-specific raw code features before/after OOD WORST:
  - `acshealthinsurance`, remove `ST`: 0.7739 -> 0.7776; default remove
  - `acspubliccoverage`, remove `ST`: 0.5915 -> 0.6648; default remove
  - `acstraveltime`, remove `PUMA`, `ST`, `POWPUMA`: 0.4658 -> 0.4764; default remove
- Runner defaults reflected these decisions at the time; current dataset-specific
  removed-feature defaults live in `acs_tasks/config.py`.
- Detailed results: `codex_notes/dev_log/2026-05-04.md`, section
  `ACS Plain CE Baseline: Reweighting And State-Feature Defaults`.

Additional conclusion:

- Completed full fixed `CrossEntropyLoss` baselines for all non-income ACS tasks using the
  same training setting as ACSIncome:
  - `TRAIN_BATCH=256`
  - `EVAL_BATCH=2048`
  - `LR=1e-4`
  - `PATIENCE=500`
  - `REPEAT=5`
  - `MAX_EPOCHS=5000`
  - seeds `[9803, 38224, 8113, 4854, 98825]`
- Final full fixed baseline results:
  - `acsemployment`: `ID = 0.8436`, `OOD MEAN = 0.8052`, `OOD WORST = 0.7649`
  - `acsemploymentfiltered`: `ID = 0.7897`, `OOD MEAN = 0.7558`, `OOD WORST = 0.7062`
  - `acshealthinsurance`: `ID = 0.8278`, `OOD MEAN = 0.8277`, `OOD WORST = 0.7816`
  - `acsincomepovertyratio`: `ID = 0.7290`, `OOD MEAN = 0.7011`, `OOD WORST = 0.6254`
  - `acsmobility`: `ID = 0.7236`, `OOD MEAN = 0.7243`, `OOD WORST = 0.6624`
  - `acspubliccoverage`: `ID = 0.7198`, `OOD MEAN = 0.7753`, `OOD WORST = 0.6715`
  - `acstraveltime`: `ID = 0.5998`, `OOD MEAN = 0.5982`, `OOD WORST = 0.4820`
- `README.md` benchmark now uses these full-setting values for the non-income ACS
  `CrossEntropyLoss fix dataset issue` column.
- Detailed results: `codex_notes/dev_log/2026-05-04.md`, section
  `ACS Full Fixed Baseline Results`.

### 2026-04-29

Conclusion:

- Completed ACSIncome scale-aware ranking / assignment / hyperparameter sweep using:
  - `reweighting=True`
  - raw inputs, no dataset standardization
  - `importance_scale="train_std"`
  - fixed seeds `[9803, 38224, 8113]` for screening and
    `[9803, 38224, 8113, 4854, 98825]` for confirm.
- Weight-assignment screening:
  - best 3-seed OOD WORST came from moderate `OCCP` reduction:
    - `structured_occp6`: `ID = 0.8502`, `OOD MEAN = 0.7608`, `OOD WORST = 0.7201`
  - more aggressive `OCCP=4`, GPT 3-group / 4-group, old GPT raw scores, and old direct GPT
    ranking did not improve worst-state accuracy.
- Hyperparameter screening:
  - best 3-seed hyperparameter move was increasing `REG_SCALE` to `1.0`:
    - `ID = 0.8563`, `OOD MEAN = 0.7618`, `OOD WORST = 0.7194`
  - `target_power=0.5` and reducing either grad or weight target pressure raised ID but hurt
    OOD WORST.
  - `suppress_scale` had no effect for all-positive structured score weights because no feature
    entered the suppressed mask.
- 5-seed confirm:
  - `structured_occp6`, `REG_SCALE=0.5`:
    - `ID = 0.8533`
    - `OOD MEAN = 0.7614`
    - `OOD WORST = 0.7191`
  - structured score, `REG_SCALE=1.0`:
    - `ID = 0.8553`
    - `OOD MEAN = 0.7624`
    - `OOD WORST = 0.7212`
- Current best feature-prior setting is structured GPT score weights with `REG_SCALE=1.0`.
  Compared with CE reweighting baseline (`0.8490 / 0.7580 / 0.7232`), it wins on ID and
  OOD MEAN but still loses on OOD WORST.
- Main conclusion:
  - `importance_scale="train_std"` plus tuning can make the GPT-prior loss beat CE reweighting
    on ID and OOD MEAN.
  - Worst-state robustness remains unsolved; CE reweighting still has the best OOD WORST.

Reference details:

- `codex_notes/dev_log/2026-04-29.md`
- implementation / output reference:
  - `archive/acsincome/run_acsincome_scale_sweep.py`
  - `archive/acsincome/results/scale_sweep_screening.jsonl`
  - `archive/acsincome/results/scale_sweep_hparam_structured.jsonl`
  - `archive/acsincome/results/scale_sweep_confirm.jsonl`

### 2026-04-26

Conclusion:

- `importance_scale="train_std"` works as a unit conversion, not as a new causal assumption.
- Original raw gradient importance measured `d logit / d x_j`, which depends on the raw unit
  of feature `x_j`. For large-scale features such as `OCCP`, one raw unit is a tiny movement,
  so the model can have very small raw gradient while still changing output a lot over the
  observed feature distribution.
- Train-std scaling instead regularizes `d logit / d z_j`, where
  `x_j = mean_j + std_j * z_j`:
  - `scaled_grad_j = raw_grad_j * train_std_j`
  - `scaled_weight_abs_j = abs(first_layer_weight_j) * train_std_j`
- This aligns the loss more closely with SHAP, because both are now closer to measuring
  output sensitivity over realistic feature variation rather than one arbitrary raw unit.
- Empirical status from the 3-seed plotting run:
  - small `Feature Grad L2` is fixed: plotted per-feature grad L2 is around `1e-3`, not
    the previous `1e-7`-level collapse.
  - SHAP is improved but not fully solved: repeat 1/2 no longer show `OCCP` dominance, but
    repeat 3 still has `OCCP = 0.1656` versus `WKHP/SCHL = 0.0776`.
- Therefore std scaling fixes the raw-unit mismatch, but does not guarantee stable OOD
  feature usage. Remaining instability likely comes from target-prior mass on `OCCP`, CE
  pressure, nonlinear training dynamics, and state-specific distribution shift.

Reference details:

- `codex_notes/dev_log/2026-04-26.md`

### 2026-04-25

Conclusion:

- 不需要把 ACSIncome dataset 本身 standardize；更小干預是保留 raw inputs，
  但在 feature-importance regularizer 裡用 train-set std 換算 importance。
- 已新增 `importance_scale="train_std"`：
  - gradient importance 用 `grad_j * train_std_j`
  - first-layer weight importance 用 `abs(weight_j) * train_std_j`
- 這讓 regularizer 控制的是每一個 feature 變動一個 train std 時的 output sensitivity，
  與 SHAP 的 feature-variation notion 更一致。
- 5-seed ACSIncome screening with 2026-04-14 GPT ranking + reweighting:
  - CE reweighting baseline from 2026-04-19:
    - `ID = 0.8490`
    - `OOD MEAN = 0.7580`
    - `OOD WORST = 0.7232`
  - `FeatureImportanceTargetCELoss + importance_scale="train_std"`:
    - `ID = 0.8591`
    - `OOD MEAN = 0.7602`
    - `OOD WORST = 0.7170`
- 結論：
  - scale-aware importance 解決 standardize dataset 造成的 ID collapse；
  - ID 與 OOD MEAN 已超過 CE reweighting；
  - OOD WORST 仍低於 CE reweighting，下一步要降低 GPT ranking 中 `OCCP` 的 target mass。

Reference details:

- `codex_notes/dev_log/2026-04-25.md`
- implementation reference:
  - `src/loss.py`
  - `src/main.py`
  - `run_acs_fitce.py`

### 2026-04-23

Conclusion:

- ACSIncome 上 `FeatureImportanceTargetCELoss` 讓 `Feature Grad L2` 變成 `1e-7` 且 SHAP 被 `OCCP` 主導，主要原因是 ACSIncome raw features 沒有標準化。
- `synthetic_ood` 會用 train stats 標準化；ACSIncome 原本直接餵原始 Folktables 編碼。
- PR train split 的尺度差異非常大：
  - `OCCP std = 2405.278`，約是 `AGEP` 的 `176.20x`
  - `POBP std = 56.874`，約是 `AGEP` 的 `4.17x`
- `FeatureImportanceTargetCELoss` 控制的是 raw input unit 上的 `d logit / d x_j` 與 first-layer weight distribution；SHAP 反映的是 feature 值變動造成的輸出變動。因此大尺度 raw feature 可以用很小 gradient 產生很大的 SHAP。
- 同一個 initial batch 診斷：
  - no standardization: `total_grad_l2 = 3.1759e-7`
  - train-stat standardization: `total_grad_l2 = 2.9516e-2`
- 已新增 ACSIncome train-stat standardization config：
  - `DATASET_CONFIG={"resampling": False, "standardize": True}`
  - train/val/test 都使用 PR train split 的 mean/std。

Reference details:

- `codex_notes/dev_log/2026-04-23.md`
- implementation reference:
  - `acsincome/dataset.py`
  - `src/main.py`
- `run_acs_fitce.py`

### 2026-04-19

Conclusion:

- pure CE balancing 2x2 experiment 已完成，固定 `LOSS_NAME="cross_entropy"`，比較 `{resampling=False/True} x {reweighting=False/True}`。
- `CrossEntropyCELoss(reweighting=True)` 使用完整 batch forward，再按 present ground-truth classes 平均 per-sample CE；這保留 `BatchNorm1d` 行為，避免 class subset size 為 1 時報錯。
- 5-seed averages:
  - no balancing:
    - `ID = 0.9140`
    - `OOD MEAN = 0.7063`
    - `OOD WORST = 0.6166`
  - resampling only:
    - `ID = 0.8470`
    - `OOD MEAN = 0.7594`
    - `OOD WORST = 0.7203`
  - reweighting only:
    - `ID = 0.8490`
    - `OOD MEAN = 0.7580`
    - `OOD WORST = 0.7232`
  - resampling + reweighting:
    - `ID = 0.8551`
    - `OOD MEAN = 0.7583`
    - `OOD WORST = 0.7154`
- 在 pure CE 下，resampling 與 reweighting 都能把 OOD WORST 從約 `0.6166` 拉到 `0.72` 左右，但 ID 會降到約 `0.85`。
- `reweighting` only 的 OOD WORST 最高；`resampling` only 的 OOD MEAN 最高；兩者一起用沒有疊加效果。
- 後續 feature-prior loss 實驗必須把 CE balancing baselines 作為必要對照。

Reference details:

- `codex_notes/dev_log/2026-04-19.md`
- implementation reference:
  - `src/loss.py`
  - `src/main.py`
  - `acsincome/dataset.py`

### 2026-04-18

Conclusion:

- ACSIncome train label imbalance 是 OOD WORST 的獨立瓶頸。
- PR train split label distribution:
  - before resampling: `0: 6483 (0.8933), 1: 774 (0.1067)`
  - after balanced oversampling: `0: 6483 (0.5000), 1: 6483 (0.5000)`
- `MLP + CrossEntropyLoss` 5-seed comparison:
  - no-resampling fixed-seed baseline:
    - `ID = 0.9140`
    - `OOD MEAN = 0.7063`
    - `OOD WORST = 0.6166`
  - train-only balanced oversampling:
    - `ID = 0.8470`
    - `OOD MEAN = 0.7594`
    - `OOD WORST = 0.7203`
- Resampling 大幅提高 OOD，尤其 OOD WORST，但 ID 大幅下降。
- 目前它更像 OOD upper-bound / tradeoff 訊號，不是可直接取代 fixed-seed baseline 的 final setting。
- 下一步是比較 pure CE 下的 loss reweighting，確認是否能保留 OOD gain 並降低 ID drop。
- 後續 loss + GPT-ranking 實驗應納入 resampling 或 class-weighted CE 作為比較軸，
  否則 ranking effects 可能被 label imbalance 掩蓋。

Reference details:

- `codex_notes/dev_log/2026-04-18.md`
- implementation reference:
  - `acsincome/dataset.py`
  - `src/main.py`
  - `src/loss.py`

### 2026-04-14

Conclusion:

- Synthetic loss 設計可以轉移到 ACSIncome，但前提是 ranking 足夠接近可遷移結構。
- ACSIncome fixed-seed baseline:
  - `ID = 0.9140`
  - `OOD MEAN = 0.7063`
  - `OOD WORST = 0.6166`
- leave-one-out ranking 診斷：
  - `leave_one_out_top7` 5 seeds:
    - `ID = 0.9157`
    - `OOD MEAN = 0.7129`
    - `OOD WORST = 0.6240`
  - 代表在 ranking 夠準時，`FeatureImportanceTargetCELoss` 確實可提升 ACSIncome OOD。
- GPT ranking / grouping 目前沒有成功超過 baseline：
  - `gpt_3group_321` 5 seeds:
    - `ID = 0.9126`
    - `OOD MEAN = 0.6972`
    - `OOD WORST = 0.5999`
  - `gpt_4group_4321` seed1 幾乎與 `gpt_3group_321` seed1 打平，沒有顯示更細 grouping 的明顯優勢。
- 當時的主結論：
  - loss 不是沒用
  - ACSIncome 的瓶頸更像 GPT ranking / grouping quality

Reference details:

- `codex_notes/dev_log/2026-04-14.md`
- sections:
  - `ACSIncome Transfer Experiment 1`
  - `ACSIncome Transfer Experiment 1: First-Round Main Result`
  - `Strategy Pivot After User Feedback`
  - `ACSIncome GPT Grouping Experiment 2: Coarse Grouping Sweep`
- implementation reference:
  - `run_acs_fitce.py`
  - `archive/acsincome/run_acsincome_gpt_groups.py`
  - `archive/acsincome/gpt_acsincome.json`

### 2026-04-12

Conclusion:

- `FeatureImportanceTargetCELoss` 優於單純 pairwise ranking。
- 在 synthetic perfect ranking 下，target distribution 對 gradient-based 與
  first-layer-weight-based variants 都比 pairwise ranking 更好。
- 在 synthetic perfect ranking 下，target distribution + suppressed-mass penalty 能大幅降低 spurious usage。
- term ablation 的重要性排序是：
  1. `suppress_scale`
  2. `weight_scale`
  3. `grad_scale`
- 最佳 synthetic reference setting:
  - `MLP`
  - `FeatureImportanceTargetCELoss`
  - perfect ranking: `causal_main > causal_aux > noise > spurious_main`
  - `REG_SCALE = 0.5`
  - `grad_scale = 2.0`
  - `weight_scale = 3.0`
  - `suppress_scale = 6.0`
  - `target_power = 1.0`
  - result: `ID = 0.9473`, `OOD MEAN = 0.8758`, `OOD WORST = 0.8758`
- Full factorial term ablation is now part of the 2026-04-12 dev log.
- 當時的下一個核心問題是：real dataset 能不能提供足夠好的 feature ranking。

Reference details:

- `codex_notes/dev_log/2026-04-12.md`
- sections:
  - `Synthetic OOD Experiment 1`
  - `Synthetic OOD Experiment 2: Variance Analysis`
  - `Synthetic OOD Experiment 3: Stronger Suppression`
  - `Synthetic OOD Experiment 4: Perfect Ranking Sensitivity`
  - `Synthetic OOD Experiment 5: Term Ablation`
  - `Synthetic OOD Experiment 6: Pairwise Ranking vs Target Distribution`
  - `Full Term Ablation Table`
- implementation reference:
  - `src/loss.py`
  - `run_synthetic_ood.py`

### 2026-04-02

Conclusion:

- synthetic pipeline 已整合回 shared core。
- ACSIncome 與 synthetic 共用 `src/main.py` / `src/trainer.py`。
- `MAX_EPOCHS`、`DATASET_CONFIG`、`PLOT_TEST_SHAP` 成為 shared knobs。
- 之後不應再維護一套 synthetic-only trainer / main。

Reference details:

- `codex_notes/dev_log/2026-04-02.md`
- implementation reference:
  - `src/main.py`
  - `src/trainer.py`
  - `run_synthetic_ood.py`

### 2026-04-01

Conclusion:

- 建立 synthetic OOD dataset 後，`FeatureGradCELoss` 的方向性 sanity check 成立：
  - favor causal features 提升 OOD
  - favor spurious feature 降低 OOD
- 移除 spurious / noise features 會給出最高 OOD upper-bound 訊號。
- 因此 loss 本身大概率沒有根本性設計錯誤。
- 當時的關鍵判讀是：第一個問題不是 gradient signal 本身壞掉，loss idea 值得繼續推進。

Reference details:

- `codex_notes/dev_log/2026-04-01.md`
- section: `Initial FeatureGradCELoss Sanity Benchmark`
