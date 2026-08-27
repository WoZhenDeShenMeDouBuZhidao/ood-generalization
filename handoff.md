# Project Handoff

Last updated: 2026-08-01.

## Immediate Task

After moving this project to the remote machine, reproduce the experiment in
`codex_notes/dev_log/2026-07-30.md` under **Real-World Reweighting Policy and
Balanced-Metric Results**. The goals are:

1. verify that the copied environment, data caches, GPT ranking artifacts, and
   CUDA setup work without unexpected errors;
2. rerun the 18 real-world datasets with CE, FITCE, and LAAT under explicit
   reweighting `off` and `on`;
3. compare the new JSON results with the 2026-07-30 tables and confirm that the
   overall conclusions remain the same.

Do not regenerate GPT rankings for this task. The existing GPT-5.4 `rank`
artifacts are the inputs used by the original experiment.

Read these files first:

- `codex_notes/context.md`: current protocol, active conclusions, architecture,
  and experiment history.
- `README.md`: CLI options, baseline settings, dataset settings, and environment
  setup.
- `codex_notes/dev_log/2026-07-30.md`: exact reference tables and interpretation.

## Architecture

There is one active experiment runner and one ranking-generation entrypoint:

| Path | Responsibility |
|---|---|
| `run_experiment.py` | Runs every dataset/method combination and writes JSON results. |
| `rank_features.py` | Generates real-world GPT ranking/scoring artifacts; not needed for the reproduction. |
| `src/benchmark_config.py` | Builds unified `DatasetSpec` objects, dataset caps, split roles, reweighting defaults, cache paths, feature indices, and prior mappings. |
| `src/main.py` | PyTorch repeats, fixed seeds, checkpoint selection, evaluation, JSON output, and plotting dispatch. |
| `src/trainer.py` | Training loop, early stopping, test evaluation, curves, SHAP, and attribution diagnostics. |
| `src/loss.py` | CE, gradient-regularized CE, FITCE, LAAT, and shared attribution helpers. |
| `src/metrics.py` | Binary Balanced Accuracy and Macro-F1. |
| `src/utils.py` | Confidence intervals, ID/OOD aggregation, selection score, JSON, curve, and SHAP helpers. |
| `src/ranking.py` | Loads ranking artifacts and converts semantic rank/score values into model-feature weights. |
| `src/semantic_features.py` | Expands one semantic categorical-feature prior across its one-hot model columns. |
| `src/feature_cards.py`, `src/ranking_prompts.py`, `src/ranking_generation.py` | Feature descriptions, prompts, and OpenAI API generation. |
| `src/data_cache.py`, `src/paths.py` | Canonical cache and output paths. |
| `data/<benchmark>/` | Benchmark loaders/configs and per-dataset `raw/`, `cache/`, and `rankings/`. |
| `output/<method>/<benchmark>/<dataset>/result.json` | Canonical experiment result. |
| `archive/` | Historical runners and implementations; active code must not import them. |

Active PyTorch methods are `mlp_ce`, `gradient_regularized_ce`, `fitce`,
`laat`, and `llm_select`. The `linear` and `llm_lasso` methods use sklearn.

The 18 real-world datasets used here are:

- ACS: `acsincome`, `acsemployment`, `acsemploymentfiltered`,
  `acshealthinsurance`, `acsincomepovertyratio`, `acsmobility`,
  `acspubliccoverage`, `acstraveltime`.
- WHYSHIFT: `taxi`, `accident`.
- TableShift: `college_scorecard`, `diabetes_readmission`, `nhanes_lead`,
  `nhanes_cholesterol`, `anes`, `acsfoodstamps`, `brfss_diabetes`,
  `brfss_blood_pressure`.

The five `synthetic_ood` datasets are not part of this reproduction.

## Local Artifacts That Must Be Moved

`.gitignore` excludes all `raw/`, `cache/`, `output/`, and `.env` paths. A Git
clone alone is therefore insufficient. Copy these local artifacts separately:

- all required `data/**/raw/` directories;
- the current default `data/**/cache/` directories;
- `data/**/rankings/`, especially every
  `gpt-5.4_rank_feature_ranking.json`;
- a newly created remote `.env` containing thread limits and any API
  credentials. Do not commit credentials.

Before transfer, this machine had all 18 default real-world caches and all 18
GPT-5.4 rank artifacts. The default cache configurations are:

- ACS: `max_train_val_size=10000`, uncapped tests, no resampling, no
  standardization.
- WHYSHIFT: preprocessed, `max_train_val_size=8000`,
  `max_per_test_size=8000`.
- TableShift: `max_train_val_size=10000`, uncapped `ood_test`.

Keep the copied relative paths unchanged. Cache directory names encode these
settings and removed feature indices; changing a cap or preprocessing flag
builds a different cache.

The workspace may contain intentional uncommitted research changes. Inspect
`git status` and do not revert unrelated files simply to obtain a clean tree.

## Environment Recreation

The reference environment is Linux x86_64 with Python `3.10.19`, pip
`26.0.1`, PyTorch `2.10.0+cu128`, and CUDA 12.8 runtime wheels. The remote
NVIDIA driver must support CUDA 12.8.

Run from the project root:

```bash
conda create -n ood -y \
  python=3.10.19 pip=26.0.1 setuptools=80.10.2 wheel=0.46.3
conda activate ood
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  git+https://github.com/mlfoundations/tableshift.git@fca9429814703a07e3902d005d46563a207b7f0a
```

TableShift must be installed last with `--no-deps`. Its metadata expects
`numpy==1.23.5`, `rtdl`, and `tab-transformer-pytorch`, while this project only
uses its dataset API with the newer NumPy/PyTorch stack. These three
TableShift-only messages from `pip check` are expected; other dependency errors
are not.

Recommended `.env` thread limits:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMBA_NUM_THREADS=1
```

An OpenAI API key is not required for this reproduction because it reads the
saved ranking artifacts.

## Environment And Data Checks

Run these before a formal experiment:

```bash
python -m compileall -q src data run_experiment.py rank_features.py
python run_experiment.py --help
python rank_features.py --help
python -m pip check
```

Verify CUDA and important imports:

```bash
python - <<'PY'
import numpy, pandas, shap, sklearn, tableshift, torch

print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU count", torch.cuda.device_count())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("shap", shap.__version__)
PY
```

Then run a short cache-backed smoke test with a unique result namespace:

```bash
python run_experiment.py \
  --benchmark acs \
  --dataset acsincome \
  --method mlp_ce \
  --device cuda:0 \
  --reweighting on \
  --repeat 1 \
  --patience 2 \
  --max-epochs 3 \
  --result-method handoff_smoke_mlp_ce
```

Confirm its JSON reports `"status": "ok"`, `"data_cache_hit": true`, the
expected CUDA device, and finite Balanced Accuracy/Macro-F1 values. Delete or
ignore only this uniquely named smoke output afterward.

## 2026-07-30 Reproduction Protocol

The current parser defaults are not the complete 07-30 controlled-screen
settings. Pass every setting below explicitly.

- Datasets: all 18 real-world datasets; exclude `synthetic_ood`.
- Methods: `mlp_ce`, `fitce`, `laat`.
- Reweighting: independently run `off` and `on`.
- Shared training: `repeat=3`, `patience=200`, `max_epochs=5000`, `lr=1e-4`,
  train batch 256, eval batch 2048.
- Seeds: first three entries of `[9803, 38224, 8113, 4854, 98825]`.
- Checkpoint score: validation Balanced Accuracy + validation Macro-F1.
- FITCE: GPT-5.4 semantic `rank`, rank-to-weight conversion, gradient term
  only, `reg_scale=0.1`, `loss_lambda=1`, `loss_alpha=1`, `tau=2`, no
  importance scaling, no warmup.
- LAAT: the same GPT-5.4 semantic `rank` prior and `reg_scale=0.1`.
- TableShift ID is official `validation`; its only reported OOD group is
  official `ood_test`.
- OOD selection score is BAcc OOD Mean + BAcc OOD Worst + Macro-F1 OOD Mean +
  Macro-F1 OOD Worst.

Use unique `--result-method` names so existing results remain untouched. Run
each command once for every `benchmark` in `acs`, `whyshift`, and `tableshift`.

CE template:

```bash
python run_experiment.py \
  --benchmark <benchmark> \
  --method mlp_ce \
  --result-method repro_0730_mlp_ce_<off-or-on> \
  --device <cuda-device> \
  --reweighting <off-or-on> \
  --repeat 3 --patience 200 --max-epochs 5000 --lr 1e-4 \
  --train-batch 256 --eval-batch 2048
```

FITCE template:

```bash
python run_experiment.py \
  --benchmark <benchmark> \
  --method fitce \
  --result-method repro_0730_fitce_<off-or-on> \
  --device <cuda-device> \
  --reweighting <off-or-on> \
  --repeat 3 --patience 200 --max-epochs 5000 --lr 1e-4 \
  --train-batch 256 --eval-batch 2048 \
  --ranking-method rank \
  --feature-weight-mode rank \
  --ranking-feature-space semantic \
  --ranking-model gpt-5.4 \
  --reg-scale 0.1 \
  --reg-warmup-epochs 0 \
  --loss-lambda 1 \
  --loss-alpha 1 \
  --grad-prob-temperature 2 \
  --importance-scale none
```

LAAT template:

```bash
python run_experiment.py \
  --benchmark <benchmark> \
  --method laat \
  --result-method repro_0730_laat_<off-or-on> \
  --device <cuda-device> \
  --reweighting <off-or-on> \
  --repeat 3 --patience 200 --max-epochs 5000 --lr 1e-4 \
  --train-batch 256 --eval-batch 2048 \
  --ranking-method rank \
  --feature-weight-mode rank \
  --ranking-feature-space semantic \
  --ranking-model gpt-5.4 \
  --reg-scale 0.1
```

The original run also produced curves and validation SHAP. They are not needed
to reproduce the reweighting policy or balanced-metric tables. Omit them for
the first environment check; add `--plot-curve --plot-shap
--shap-sample-size 500` only if diagnostic artifacts must also be reproduced.

On a two-GPU machine, start with two to four concurrent processes and increase
only if CPU load and GPU utilization permit. FITCE and LAAT are more expensive
than CE. Avoid launching all 18 benchmark-level commands simultaneously.

## Reference Results And Acceptance Criteria

The old result namespaces are:

- `bacc_reweight_r3_mlp_ce_<off|on>`
- `bacc_reweight_r3_fitce_rank_reg0p1_tau2_<off|on>`
- `bacc_reweight_r3_laat_rank_reg0p1_<off|on>`

Each `result.json` contains aggregate values in `selection_metrics` and
`metrics`, plus seed-level values in `repeats`. Compare matched seeds between
`off` and `on`; do not compare unrelated random runs.

Result schema v2 keeps `metadata` limited to the experiment method/baseline and
the necessary prior or selection semantics. For PyTorch methods,
`--record-best-grad-l2` adds `best_grad_l2` to each repeat. The vector follows
`features.names` and is measured at the fixed best checkpoint on the training
split using raw logit-margin input gradients. It is class-balanced when the
run's resolved reweighting is on and sample-averaged when reweighting is off;
FITCE's TrainStd scaling is not applied.

The original ID reweighting trends were:

| Method | Datasets with higher mean ID sum | Macro mean ID-sum delta, on minus off |
|---|---:|---:|
| CE | 16/18 | +0.0490 |
| FITCE | 16/18 | +0.0722 |
| LAAT | 13/18 | +0.0416 |

All observed negative mean changes were smaller than 1% of the corresponding
off score, and every paired 95% CI for a negative mean included zero. This led
to the frozen policy: class reweighting is enabled for every real-world dataset
and compatible method, based only on ID validation metrics.

With reweighting enabled, the old macro results were:

| Method | ID BAcc | ID Macro-F1 | OOD selection |
|---|---:|---:|---:|
| CE | 0.7212 +/- 0.0015 | 0.6870 +/- 0.0040 | 2.6360 +/- 0.0114 |
| FITCE | 0.7043 +/- 0.0120 | 0.6679 +/- 0.0165 | 2.5851 +/- 0.0296 |
| LAAT | 0.7185 +/- 0.0133 | 0.6836 +/- 0.0121 | 2.6261 +/- 0.0438 |

Exact equality is not required across machines because CUDA kernels and driver
versions can introduce small numerical differences. Treat the reproduction as
successful when all runs finish with valid JSON and the aggregate conclusions
remain stable:

- reweighting improves mean ID Balanced Accuracy + Macro-F1 for a clear
  majority of datasets for all three methods;
- negative dataset-level changes remain small rather than systematic;
- CE and LAAT remain close overall, while this controlled FITCE setting is
  weaker on the aggregate OOD selection score;
- per-dataset deviations are generally compatible with the old seed-level
  uncertainty, with no benchmark-wide reversal.

If a trend changes materially, first check `data_cache_hit`, dataset caps,
TableShift split roles, ranking metadata, the three model seeds, and all
explicit FITCE arguments. Record the remote GPU, driver, package versions, old
versus new macro tables, per-dataset outliers, and any errors in a new dated
`codex_notes/dev_log/YYYY-MM-DD.md`. Do not rewrite the 2026-07-30 historical
record.

## Current Research Status

Balanced Accuracy/Macro-F1 and the shared real-world reweighting policy are the
current protocol. Results before 2026-07-30 mostly use the superseded Accuracy /
positive-class F1 protocol and should not be mixed into the current leaderboard.
FITCE and LAAT still require full re-optimization under the expanded balanced
protocol; the 07-30 experiment is a controlled reweighting screen, not the
final optimized baseline comparison.
