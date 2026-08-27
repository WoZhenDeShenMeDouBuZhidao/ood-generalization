# OOD Generalization Experiments

這份 README 只保留公開 benchmark 摘要。研究脈絡、完整實驗表格、
implementation notes、後續 reference 都放在本地私有筆記 `codex_notes/`。

## Performance Comparison

Formal benchmark results use Balanced Accuracy and Macro-F1, reported as
`mean +/- 95% CI range`. The tables will be populated after the new
reweighting experiments are complete.

### Balanced Accuracy Metrics

Pending formal reruns.

### Macro-F1 Metrics

Pending formal reruns.

## Running Experiments

Use `run_experiment.py` for every benchmark and method. The synthetic OOD
datasets are selected with `--benchmark synthetic_ood`; omit `--dataset` to run
all five variants.

For a quick plotted synthetic OOD diagnostic run, use a short repeat/budget and
write to a throwaway result method so formal JSON results are not overwritten:

```bash
python run_experiment.py \
  --method fitce \
  --benchmark synthetic_ood \
  --dataset simple \
  --device cuda:0 \
  --repeat 1 \
  --max-epochs 500 \
  --patience 100 \
  --plot-curve \
  --plot-shap \
  --plot-test-shap \
  --show-progress \
  --result-method fitce_synthetic_plot
```

The command writes metrics to
`output/fitce_synthetic_plot/synthetic_ood/simple/result.json`, curve plots to
`output/fitce_synthetic_plot/synthetic_ood/simple/curve/`, and SHAP plots to
`output/fitce_synthetic_plot/synthetic_ood/simple/shap/`. `--plot-curve` includes the
feature-gradient and first-layer-weight panels when the selected loss reports
those terms; they are most informative for `fitce` and `laat`.
`--plot-shap` computes validation SHAP for the first 3 repeats, and
`--plot-test-shap` also computes held-out OOD SHAP.

For a formal synthetic OOD FITCE run after choosing hyperparameters, use the
normal repeat/budget:

```bash
python run_experiment.py \
  --method fitce \
  --benchmark synthetic_ood \
  --device cuda:0 \
  --repeat 5 \
  --max-epochs 5000 \
  --patience 200 \
  --result-method fitce
```

Experiment settings should be passed explicitly as CLI arguments. Environment
variables are reserved for OpenAI API configuration such as `API_KEY`,
`API_URL`, and `OPENAI_TIMEOUT`.

### Simple Method Commands

Each example runs one dataset; omit `--dataset` to run every dataset in the
selected benchmark.

```bash
python run_experiment.py \
  --method mlp_ce \
  --benchmark acs \
  --dataset acsincome
```

```bash
python run_experiment.py \
  --method gradient_regularized_ce \
  --benchmark acs \
  --dataset acsincome \
  --reg-scale 1.0 \
  --importance-scale none
```

```bash
python run_experiment.py \
  --method fitce \
  --benchmark acs \
  --dataset acsincome \
  --ranking-method score_all \
  --feature-weight-mode score \
  --ranking-model gpt-5.4 \
  --loss-lambda 16.0 \
  --loss-alpha 0.75 \
  --reg-scale 2.0 \
  --importance-scale train_std
```

```bash
python run_experiment.py \
  --method laat \
  --benchmark acs \
  --dataset acsincome \
  --ranking-method score_all \
  --feature-weight-mode score \
  --ranking-model gpt-5.4 \
  --reg-scale 1.0
```

```bash
python run_experiment.py \
  --method llm_select \
  --benchmark acs \
  --dataset acsincome \
  --ranking-method score_all \
  --ranking-model gpt-5.4 \
  --selection-mode top_p \
  --top-p 1.0
```

```bash
python run_experiment.py \
  --method linear \
  --benchmark acs \
  --dataset acsincome \
  --c 1.0 \
  --max-iter 1000
```

```bash
python run_experiment.py \
  --method llm_lasso \
  --benchmark acs \
  --dataset acsincome \
  --ranking-method score_all \
  --ranking-model gpt-5.4 \
  --eta 0.0 \
  --c 1.0 \
  --max-iter 5000 \
  --penalty-floor 0.1 \
  --class-weight none
```

### Shared Dataset Arguments

| option | runner | default | notes |
|---|---|---|---|
| `--method {mlp_ce,gradient_regularized_ce,fitce,laat,llm_select,linear,llm_lasso}` | `run_experiment.py` | `mlp_ce` | Selects the method. |
| `--benchmark {acs,whyshift,tableshift,synthetic_ood}` | `run_experiment.py` | `acs` | Selects the benchmark family. |
| `--dataset DATASET` | `run_experiment.py` | all datasets in benchmark | For synthetic OOD, use one of the five configured variants or omit this option. |
| `--max-train-val-size N` | `run_experiment.py` | benchmark-specific | `0` disables the cap. |
| `--max-per-test-size N` | `run_experiment.py` | benchmark-specific | `0` disables the cap. |
| `--no-preprocess` | `run_experiment.py` | off | Only affects WHYSHIFT preprocessing. |
| `--categorical-encoding {auto,integer,one_hot}` | `run_experiment.py` | `auto` | Preserves each dataset's native representation or forces categorical predictors to integer/one-hot encoding; encoded feature columns are mapped back to semantic GPT-ranking features. |
| `--semantic-expansion-policy {shared,split}` | `run_experiment.py` | `shared` | For semantic priors expanded over one-hot columns, either copies the full feature weight to every category or divides it equally across categories. |
| `--grad-alignment-space {model,semantic}` | `run_experiment.py` | `model` | FITCE only. `semantic` sums train-std-scaled Grad-L2 across one-hot columns before temperature, normalization, and alignment with the unexpanded semantic prior. |
| `--result-method NAME` | `run_experiment.py` | selected method | Overrides the method namespace in `output/<method>/<benchmark>/<dataset>/`; use a unique name for manual smoke runs. |

### Shared Neural Runner Arguments

These options apply to PyTorch methods: `mlp_ce`, `gradient_regularized_ce`,
`fitce`, `laat`, and `llm_select`.

| option | default | notes |
|---|---|---|
| `--device DEVICE` | `cuda:0` if available, else `cpu` | Use `cuda:0`, `cuda:1`, or `cpu`. |
| `--train-batch N` | `256` | Training batch size. |
| `--eval-batch N` | `2048` | Validation/test batch size. |
| `--lr X` | `1e-4` | Adam learning rate. |
| `--patience N` | `200` | Early stopping patience by validation Balanced Accuracy + Macro-F1. |
| `--repeat N` | `5` | Uses fixed model seeds in order. |
| `--max-epochs N` | `5000` | Maximum epochs per repeat. |
| `--plot-curve` | off | Saves training/validation and attribution curves for the first 3 repeats. |
| `--plot-shap` | off | Saves validation SHAP plots for the first 3 repeats. |
| `--plot-test-shap` | off | Also saves SHAP plots for held-out test groups; validation SHAP is enabled too. |
| `--shap-sample-size N` | `500` | Number of samples used per SHAP plot. |
| `--show-progress` | off | Shows tqdm training progress bars. |
| `--record-best-grad-l2` | off | Records raw logit-margin Grad-L2 on the training split at each repeat's best checkpoint. Aggregation is class-balanced when resolved reweighting is on and a sample mean when it is off. |
| `--reweighting {auto,on,off}` | `auto` | Class reweighting for neural losses; `auto` uses `DatasetSpec.reweighting`. |

`run_experiment.py` reads `.env` before importing `numpy`, `torch`, or
`sklearn`, so local CPU thread limits can be set once there. Recommended values
for SHAP-heavy runs:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMBA_NUM_THREADS=1
```

### Method-Specific Arguments

For `synthetic_ood`, FITCE, LAAT, LLM-Select, and LLM-Lasso always use the
dataset's built-in oracle feature importance. Ranking method, model, feature
space, and feature-weight conversion arguments apply only to real-world data.

| method | option | default | notes |
|---|---|---|---|
| `gradient_regularized_ce` | `--reg-scale X` | `0.0` | Input-gradient L2 regularization strength; use `1.0` in the baseline command. |
| `gradient_regularized_ce` | `--importance-scale {none,train_std}` | `none` | Optional training-standard-deviation scaling. |
| `fitce` | `--ranking-method {rank,score,score_all,seq}` | `score_all` | Ranking artifact method for real-world datasets; synthetic OOD always uses its oracle weights. |
| `fitce` | `--ranking-feature-space {semantic,model}` | `semantic` | `semantic` groups one-hot model columns before prompting; `model` prompts with every model input feature. |
| `fitce` | `--feature-weight-mode {rank,score}` | `score` | `score` is valid only for `score` / `score_all` on LLM artifacts. |
| `fitce` | `--ranking-model MODEL` | `gpt-5.4` | Ranking artifact model name for real-world datasets. |
| `fitce` | `--reg-scale X` | `0.0` | Multiplier on the FITCE regularization term; use `2.0` in the current FITCE command. |
| `fitce` | `--reg-warmup-epochs N` | `0` | Linearly increases `REG_SCALE` from `0` to its configured value over the first `N` epochs. |
| `fitce` | `--loss-lambda X` | `16.0` | Total FITCE attribution strength before alpha split. |
| `fitce` | `--loss-alpha X` | `0.75` | Split: `grad_scale=lambda*alpha`, `weight_scale=lambda*(1-alpha)`. |
| `fitce` | `--grad-prob-temperature X` | `1.0` | Temperature for `softmax(log(gradient_signal) / X)`; `1.0` matches sum normalization. |
| `fitce` | `--grad-alignment-top-p X` | `1.0` | Fraction of ranked semantic features included in the gradient-alignment loss. Selection happens before one-hot expansion; masked features remain available to the model and CE loss. With `--plot-shap`, RankDist is computed only over the selected semantic groups after summing signed category SHAP values within each sample. |
| `fitce` | `--reweighting-scope {all,ce}` | `all` | With class reweighting enabled, `all` class-balances CE and FITCE regularizer terms; `ce` class-balances CE only. The current first-layer-weight target is global, so class averaging is explicit under `all` but algebraically leaves that term unchanged. |
| `fitce` | `--importance-scale {none,train_std}` | `none` | Scales gradient/weight signals by training standard deviation; use `train_std` in the current ACS FITCE command. |
| `laat` | `--ranking-method {rank,score,score_all,seq}` | `score_all` | Ranking artifact method. |
| `laat` | `--ranking-feature-space {semantic,model}` | `semantic` | Ranking artifact feature space. |
| `laat` | `--feature-weight-mode {rank,score}` | `score` | Feature weight conversion mode. |
| `laat` | `--ranking-model MODEL` | `gpt-5.4` | Ranking artifact model name. |
| `laat` | `--reg-scale X` | `0.0` | LAAT regularization strength; default implementation uses nonnegative GPT prior weights, matching this project's ranking/score artifacts. |
| `llm_select` | `--ranking-method {rank,score,score_all,seq}` | `score_all` | Ranking artifact method. |
| `llm_select` | `--ranking-feature-space {semantic,model}` | `semantic` | Ranking artifact feature space. |
| `llm_select` | `--ranking-model MODEL` | `gpt-5.4` | Ranking artifact model name. |
| `llm_select` | `--selection-mode {top_p,score_threshold}` | `top_p` | `score_threshold` only supports `score` / `score_all`. |
| `llm_select` | `--top-p X` | `1.0` | Retained-feature fraction for `--selection-mode top_p`. |
| `llm_select` | `--score-threshold X` | `0.0` | Score threshold for `--selection-mode score_threshold`. |
| `linear` | `--c X` | `1.0` | Inverse L2 strength. |
| `linear` | `--max-iter N` | `1000` | LogisticRegression iteration cap. |
| `llm_lasso` | `--ranking-method {score,score_all}` | `score_all` | LLM-Lasso requires score-based artifacts. |
| `llm_lasso` | `--ranking-feature-space {semantic,model}` | `semantic` | Ranking artifact feature space. |
| `llm_lasso` | `--ranking-model MODEL` | `gpt-5.4` | Ranking artifact model name. |
| `llm_lasso` | `--eta X` | `0.0` | LLM penalty exponent; `0.0` recovers plain L1 logistic. |
| `llm_lasso` | `--c X` | `1.0` | Inverse regularization strength. |
| `llm_lasso` | `--max-iter N` | `1000` | LogisticRegression iteration cap; use `5000` in the current LLM-Lasso command. |
| `llm_lasso` | `--penalty-floor X` | `0.1` | Minimum transformed penalty before exponentiation. |
| `llm_lasso` | `--class-weight {none,balanced}` | `none` | Optional sklearn class weighting. |

`fitce` previously exposed `--target-power` to reshape the LLM-prior target
distribution. It was removed after adding `--grad-prob-temperature`, because
both parameters control distribution sharpness. Keeping the LLM prior unchanged
and calibrating the gradient-to-probability mapping is the cleaner loss design.
Historical dev logs may still mention `target_power=1.0`.

## Benchmark and Method Settings

### Common Protocol

- All datasets are treated as binary classification tasks and report Balanced
  Accuracy and binary Macro-F1 as `mean +/- 95% CI range` for `ID`,
  `OOD MEAN`, `OOD WORST`, and `OOD STD`.
- Hyperparameter selection and final baseline comparison use the same selection
  score:
  `Balanced Acc OOD MEAN + Balanced Acc OOD WORST + Macro-F1 OOD MEAN +
  Macro-F1 OOD WORST`. The score uses metric means; confidence interval ranges
  are reported but not included in the score.
- Neural runners select checkpoints by validation Balanced Accuracy + Macro-F1.
  Hyperparameter searches are run as separate commands and compared from their
  result JSON.
- Default neural-runner settings are MLP with hidden size 64, `--lr 1e-4`,
  `--train-batch 256`, `--eval-batch 2048`, `--patience 200`,
  `--repeat 5`, `--max-epochs 5000`, validation split 0.2, plotting off, and
  progress bars off.
- Neural model initialization seeds are fixed to
  `[9803, 38224, 8113, 4854, 98825]`.
- Dataset caps are controlled by `--max-train-val-size` and
  `--max-per-test-size`. Omitted values use benchmark defaults; setting `0`
  disables that cap. The first
  cap applies to the source training data before validation construction for
  ACS/WHYSHIFT/synthetic_ood, and to the official `train` split for TableShift.
  The second cap applies independently to each held-out test group.
- Class reweighting is enabled for every real-world dataset. A controlled
  `--reweighting on` versus `--reweighting off` comparison across CE, FITCE,
  and LAAT found that every observed ID Balanced Accuracy + Macro-F1 decrease
  was below 1% of the corresponding ID score. Synthetic datasets remain
  unweighted.
- Raw data and caches are not pushed. Runners write ignored, indented JSON
  results to `output/<method>/<benchmark>/<dataset>/result.json`; README tables
  should be aggregated from these JSON files.
- LLM-prior methods use `DEFAULT_RANKING_MODEL=gpt-5.4` and ranking/scoring
  artifacts under `data/<benchmark>/<dataset>/rankings/`. Supported artifact
  methods are `rank`, `score`, `score_all`, and `seq`; supported feature weight
  modes are `rank` and `score`, with `score` valid only for `score` /
  `score_all`.
- GPT prompts default to compact semantic feature cards. One-hot encoded model
  columns are grouped into semantic features for prompting. During training,
  `--semantic-expansion-policy shared` copies the derived weight to every
  one-hot column, while `split` divides it equally across the columns in that
  semantic group.
  Use `--ranking-feature-space model` to prompt with every model input feature
  directly. If semantic and model feature names are identical, the code resolves
  back to the semantic artifact to avoid duplicate API calls and files.

### Benchmark-Specific Protocol

| benchmark | datasets | train/validation source | reported ID | reported OOD | default caps |
|---|---|---|---|---|---|
| `acs` | 8 ACS tasks | one configured state per task, split with validation ratio 0.2 | validation split from the configured state | every `ACS_BASE_TEST_STATES` state except the train/validation state | `--max-train-val-size 10000`, `--max-per-test-size 0` |
| `whyshift` | `taxi`, `accident` | one configured source domain, split with validation ratio 0.2 | validation split from the source domain | configured held-out domains | `--max-train-val-size 8000`, `--max-per-test-size 8000` |
| `tableshift` | 8 official TableShift tasks | official `train` split for fitting and official `validation` split for checkpoint selection | official `validation` | official `ood_test`; `OOD MEAN == OOD WORST` and `OOD STD=0` | `--max-train-val-size 10000`, `--max-per-test-size 0` |
| `synthetic_ood` | `simple`, `range`, `categorical_integer`, `categorical_onehot`, `multi_spurious` | generated `train_env` and validation environment | generated `id_test` | `ood_weak`, `ood_independent`, `ood_heterogeneous`, `ood_reverse` | `--max-train-val-size 0`, `--max-per-test-size 0` |

### Dataset-Specific Settings

ACS metadata comes from `data/acs/config.py`. Feature counts below are the
configured ACS feature index before removed-feature filtering.

| dataset | train/validation state | OOD states | features | removed features | reweighting |
|---|---|---|---:|---|---|
| `acsincome` | `PR` | all ACS base states except `PR` | 10 | none | pending |
| `acsemployment` | `SD` | all ACS base states except `SD` | 16 | none | pending |
| `acsemploymentfiltered` | `SD` | all ACS base states except `SD` | 17 | none | pending |
| `acshealthinsurance` | `MN` | all ACS base states except `MN` | 25 | `23: ST` | pending |
| `acsincomepovertyratio` | `HI` | all ACS base states except `HI` | 20 | none | pending |
| `acsmobility` | `AK` | all ACS base states except `AK` | 21 | none | pending |
| `acspubliccoverage` | `CA` | all ACS base states except `CA` | 19 | `16: ST` | pending |
| `acstraveltime` | `AZ` | all ACS base states except `AZ` | 16 | `9: PUMA`, `10: ST`, `14: POWPUMA` | pending |

WHYSHIFT metadata comes from `data/whyshift/config.py`. Taxi domains are read
from raw CSV filenames; the current raw files expose `bog`, `uio`, and `mex`, so
the configured `bog` source reports `uio` and `mex` as OOD.

| dataset | source domain | OOD domains | model features | semantic features | preprocessing | reweighting |
|---|---|---|---:|---:|---|---|
| `taxi` | `bog` | available Taxi domains except `bog` | 7 | 7 | StandardScaler per domain | pending |
| `accident` | `CA` | `TX`, `FL`, `OR`, `MN`, `VA`, `SC`, `NY`, `PA`, `NC`, `TN`, `MI`, `MO` | 45 | 28 | WHYSHIFT-style cleaning, one-hot encoding, scaling | pending |

TableShift metadata comes from `data/tableshift/config.py` and the official
TableShift preprocessing pipeline. Feature counts are the current post-cache
model columns and semantic prompt features.

| dataset | train split | validation split | ID split | OOD split | model features | semantic features | reweighting |
|---|---|---|---|---|---:|---:|---|
| `college_scorecard` | `train` | `validation` | `validation` | `ood_test` | 118 | 118 | pending |
| `diabetes_readmission` | `train` | `validation` | `validation` | `ood_test` | 183 | 46 | pending |
| `nhanes_lead` | `train` | `validation` | `validation` | `ood_test` | 17 | 7 | pending |
| `nhanes_cholesterol` | `train` | `validation` | `validation` | `ood_test` | 53 | 13 | pending |
| `anes` | `train` | `validation` | `validation` | `ood_test` | 375 | 54 | pending |
| `acsfoodstamps` | `train` | `validation` | `validation` | `ood_test` | 239 | 28 | pending |
| `brfss_diabetes` | `train` | `validation` | `validation` | `ood_test` | 142 | 25 | pending |
| `brfss_blood_pressure` | `train` | `validation` | `validation` | `ood_test` | 100 | 18 | pending |

Synthetic OOD uses `data/synthetic_ood/dataset.py` with `train_size=6000`,
`val_size=2000`, and 4,000 rows per ID/OOD test group. All variants are exactly
class-balanced before optional subsampling and use built-in oracle weights, so
ranking CLI arguments do not invoke an LLM.

| dataset | model features | controlled change | preprocessing | oracle |
|---|---:|---|---|---|
| `simple` | 6 | original two-causal, one-spurious setup | train-stat standardization | causal `2,1`; spurious/noise `0` |
| `range` | 6 | same latent setup with feature scales from `0.01` to `100` | none | same as `simple` |
| `categorical_integer` | 8 | removes both continuous causal features, leaving categorical causal signal plus continuous shortcut/noise | train-stat standardization | categorical semantic group weights; spurious/noise `0` |
| `categorical_onehot` | 18 | same categorical-causal setup as integer, expanded one-hot | train-stat standardization | every column shares its semantic feature's oracle weight; spurious/noise `0` |
| `multi_spurious` | 9 | four continuous shortcuts with different train correlations | train-stat standardization | causal `2,1`; all shortcuts/noise `0` |

Each dataset reports one `id_test` plus four OOD shifts. `ood_weak` reduces all
shortcut correlations to `0.3`, `ood_independent` sets them to zero,
`ood_heterogeneous` mixes retained/zero/reversed shortcuts (or uses `-0.5` for
single-shortcut variants), and `ood_reverse` reverses every train correlation.

### Baseline Settings

| method | runner | model/package | selection rule | notes |
|---|---|---|---|---|
| `MLP CrossEntropyLoss` | `run_experiment.py --method mlp_ce` | PyTorch MLP | best validation Balanced Accuracy + Macro-F1 across epochs | Fixed baseline, not broadly optimized. Uses the common neural settings, no regularization term, `cross_entropy`, and class reweighting on every real-world dataset. If optimizing, search `--lr`, hidden size, and training budget. |
| `Gradient Regularized CE` | `run_experiment.py --method gradient_regularized_ce` | PyTorch MLP | best validation Balanced Accuracy + Macro-F1 across epochs | Plain input-gradient L2 regularization baseline with no LLM prior. Current command value: `--reg-scale 1.0`, `--importance-scale none`. Search regularization strength. |
| `FeatureImportanceTargetCELoss` | `run_experiment.py --method fitce` | PyTorch MLP | best validation Balanced Accuracy + Macro-F1 across epochs | Current command value from ACS optimization: `--ranking-model gpt-5.4`, `--ranking-method score_all`, `--feature-weight-mode score`, `--loss-lambda 16.0`, `--loss-alpha 0.75` (`grad_scale=12.0`, `weight_scale=4.0`), `--reg-scale 2.0`, `--grad-prob-temperature 1.0`, and `--importance-scale train_std`. Re-optimize on the expanded benchmark suite. |
| `LLM Attribution Regularizer` | `run_experiment.py --method laat` | PyTorch MLP | best validation Balanced Accuracy + Macro-F1 across epochs | Current default: `--ranking-model gpt-5.4`, `--ranking-method score_all`, `--feature-weight-mode score`, `--reg-scale 1.0`, common neural settings, and class reweighting on every real-world dataset; no `train_std` scaling. Search ranking method, feature weight mode, and regularization strength. |
| `LLM-Select` | `run_experiment.py --method llm_select` | PyTorch MLP | fixed subset setting per run | Not optimized yet. Current command value: `--ranking-model gpt-5.4`, `--ranking-method score_all`, `--selection-mode top_p`, `--top-p 1.0`. Search ranking method plus `--top-p` candidates such as `1.0,0.75,0.5,0.25`; for `score` / `score_all`, also search `--selection-mode score_threshold` with `--score-threshold` candidates such as `0.0,0.25,0.5,0.75`. |
| `LLM-Lasso` | `run_experiment.py --method llm_lasso` | scikit-learn L1 logistic regression | fixed `eta` and `C` per run | Not optimized yet. Requires score-based artifacts (`score` or `score_all`). Current command value: `--ranking-model gpt-5.4`, `--ranking-method score_all`, `--eta 0.0`, `--c 1.0`, `--penalty-floor 0.1`, `--class-weight none`, `--max-iter 5000`, train-only `StandardScaler`; `eta=0` recovers plain L1 logistic. Search `--eta` candidates such as `0,1,2,3,4`, `--c` candidates such as `0.01,0.1,1,10,100`, `--penalty-floor`, and `--class-weight`. Official LLM-Lasso uses penalty factors with a weighted Lasso solver; this path applies the same weighted-L1 idea through feature rescaling in sklearn. |
| `LogisticRegression` | `run_experiment.py --method linear` | scikit-learn `LogisticRegression` | fixed `C` per run | Current command value: train-only `StandardScaler`, `--c 1.0`, `--max-iter 1000`, L2 `lbfgs`; runner uses `class_weight=balanced` only when `DatasetSpec.reweighting=True`. If optimizing, search `--c` candidates such as `0.01,0.1,1,10,100`, class weighting, solver/penalty variants, and `--max-iter`. |
| `XGBoost` | planned | `xgboost` | OOD objective selects hyperparameters | use the shared split/cache/report protocol |
| `CatBoost` | planned | `catboost` | OOD objective selects hyperparameters | use the shared split/cache/report protocol |
| `RandomForest` | planned | scikit-learn `RandomForestClassifier` | OOD objective selects hyperparameters | use the shared split/cache/report protocol |
| `SVM` | planned | scikit-learn SVM | OOD objective selects hyperparameters | use the shared split/cache/report protocol |
| `TabPFN` | planned | `tabpfn` | OOD objective selects supported settings | use the shared split/cache/report protocol |

## GPT Ranking Token Cost

Token counts are recorded from formal `gpt-5.4` ranking JSON files for the
real-world benchmarks under `data/<benchmark>/<dataset>/rankings/`. Synthetic
OOD is excluded because every synthetic method now uses oracle feature
importance. Cost uses actual input/output token
counts with OpenAI GPT-5.4 standard short-context pricing checked on
2026-06-22: `$2.50 / 1M` input tokens and `$15.00 / 1M` output tokens
([OpenAI API pricing](https://openai.com/api/pricing/)). Cached input, Batch,
Flex, Priority, regional-processing, and tool-call discounts or surcharges are
not included. Each matrix cell is `total tokens / cost (calls)`.

Default semantic feature-space artifacts:

Method totals:

| method | calls | input tokens | output tokens | total tokens | cost |
|---|---:|---:|---:|---:|---:|
| `rank` | 18 | 34,629 | 19,821 | 54,450 | $0.38 |
| `score` | 2,440 | 724,710 | 260,005 | 984,715 | $5.71 |
| `score_all` | 90 | 178,635 | 97,495 | 276,130 | $1.91 |
| `seq` | 488 | 1,027,909 | 39,093 | 1,067,002 | $3.16 |
| `total` | 3,036 | 1,965,883 | 416,414 | 2,382,297 | $11.16 |

| benchmark | rank | score | score_all | seq | total |
|---|---:|---:|---:|---:|---:|
| `acs` | 16,745 / $0.13 (8) | 290,528 / $1.70 (720) | 88,096 / $0.66 (40) | 145,178 / $0.51 (144) | 540,547 / $3.00 (912) |
| `whyshift` | 3,905 / $0.03 (2) | 67,818 / $0.39 (175) | 20,018 / $0.15 (10) | 40,203 / $0.13 (35) | 131,944 / $0.70 (222) |
| `tableshift` | 33,800 / $0.23 (8) | 626,369 / $3.62 (1,545) | 168,016 / $1.10 (40) | 881,621 / $2.51 (309) | 1,709,806 / $7.46 (1,902) |
| `total` | 54,450 / $0.38 (18) | 984,715 / $5.71 (2,440) | 276,130 / $1.91 (90) | 1,067,002 / $3.16 (488) | 2,382,297 / $11.16 (3,036) |

Grand total: 2,382,297 tokens across 3,036 LLM calls, costing `$11.16`.

Additional model feature-space artifacts:

These artifacts were generated for the semantic-vs-model feature-space probe.
Only datasets whose encoded model features differ from their semantic feature
cards are included. `tableshift/anes` model-space `seq` is pending because API
generation stopped on `billing_not_active` with the first key and
`insufficient_quota` with the replacement key.

Method totals:

| method | calls | input tokens | output tokens | total tokens | cost |
|---|---:|---:|---:|---:|---:|
| `rank` | 9 | 143,445 | 29,730 | 173,175 | $0.80 |
| `score` | 6,360 | 2,189,749 | 689,403 | 2,879,152 | $15.82 |
| `score_all` | 45 | 719,980 | 155,964 | 875,944 | $4.14 |
| `seq` | 897 | 8,881,669 | 71,936 | 8,953,605 | $23.28 |
| `total` | 7,311 | 11,934,843 | 947,033 | 12,881,876 | $44.04 |

| benchmark | rank | score | score_all | seq | total |
|---|---:|---:|---:|---:|---:|
| `whyshift` | 5,556 / $0.04 (1) | 92,634 / $0.52 (225) | 25,850 / $0.15 (5) | 116,316 / $0.33 (45) | 240,356 / $1.05 (276) |
| `tableshift` | 167,619 / $0.77 (8) | 2,786,518 / $15.29 (6,135) | 850,094 / $3.98 (40) | 8,837,289 / $22.95 (852) | 12,641,520 / $42.99 (7,035) |
| `total` | 173,175 / $0.80 (9) | 2,879,152 / $15.82 (6,360) | 875,944 / $4.14 (45) | 8,953,605 / $23.28 (897) | 12,881,876 / $44.04 (7,311) |

Combined GPT-5.4 ranking artifacts currently total 15,282,672 tokens across
10,389 LLM calls, costing `$55.30`.

## Appendix

### Environment Setup
```bash
conda create -n ood python=3.10.19
conda activate ood
pip install -r requirements.txt
```

TableShift is optional and only required for `--benchmark tableshift`. Install
the base environment first, then install TableShift without dependency
resolution so its upstream metadata does not downgrade the CUDA 12 / torch 2.10
stack:

```bash
python -m pip install --no-deps \
  git+https://github.com/mlfoundations/tableshift.git@fca9429814703a07e3902d005d46563a207b7f0a
```
