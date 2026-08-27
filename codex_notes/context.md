# Context

最後更新：2026-08-01。

這份檔案是 `codex_notes` 的工作入口，目標是讓新機器或新對話能快速掌握：

- 現行程式架構與實驗 protocol；
- 目前仍有效的結論、設定與未解問題；
- 舊實驗應該查哪一份 `codex_notes/dev_log/YYYY-MM-DD.md`。

完整數字、per-dataset tables 與被取代的中間設定留在 dev logs，不在此重複。
Related-work survey 位於 `codex_notes/related_work.md`，原始 PDF 位於
`codex_notes/surveyed_papers/`。

## Development Principles

- 新增或修改程式時，以最簡單、直接、可讀的方式為優先。
- 優先擴充共用核心，不複製 dataset-specific 或 method-specific runner。
- 不為使用者已保證正確的 CLI 組合加入額外防衛性程式；真實錯誤應直接暴露。
- Active experiment entrypoint 只有 `run_experiment.py`；一次性舊 runner 與 scripts
  放在 `archive/`。
- Dataset metadata 與資料處理留在 `data/` / `src/benchmark_config.py`；loss 與訓練
  設定由 runner 控制。
- 正式結果寫成 indented JSON；不要依賴 stdout/tqdm logs 整理 benchmark。
- Curves、validation SHAP 與 attribution distance 是分析 feature utilization 的證據，
  正式診斷實驗應保留。
- 舊實驗可以保留作歷史證據，但 context 與 README 必須清楚標示是否已被新 metric、
  dataset split 或 protocol 取代。
- Raw data、cache、outputs 都是 local ignored artifacts；搬移或清理前先確認現行
  default paths 仍可命中。

## Current Snapshot

### Scope

- 23 個 binary-classification datasets：
  - ACS：8
  - WHYSHIFT：2
  - TableShift：8
  - synthetic OOD：5
- 18 個 real-world datasets 用於 baseline optimization 與 comparison。
- 5 個 synthetic datasets 使用 built-in oracle feature importance，主要用來隔離 prior
  quality、feature scale、categorical encoding 與 spurious correlation 等因素。
- Active methods：
  `mlp_ce`, `gradient_regularized_ce`, `fitce`, `laat`, `llm_select`, `linear`,
  `llm_lasso`。
- Planned but not implemented：XGBoost、CatBoost、Random Forest、SVM、TabPFN。

### Current Evaluation Protocol

- Report metrics：Balanced Accuracy 與 binary Macro-F1。
- Neural checkpoint / early stopping：validation
  `Balanced Accuracy + Macro-F1`。
- Hyperparameter selection 與 baseline comparison：
  `BAcc OOD Mean + BAcc OOD Worst + Macro-F1 OOD Mean + Macro-F1 OOD Worst`。
- 每個 aggregate metric 寫成 `mean +/- 95% Student-t CI half-width`。
- TableShift official `validation` 是 project ID；official `ood_test` 是 OOD。
  TableShift 只有一個 OOD group，因此 `OOD Mean == OOD Worst`、`OOD Std == 0`。
- ACS 與 WHYSHIFT 的 ID 是 source-domain validation split；held-out states/domains 是 OOD。
- 所有 real-world datasets 統一開啟 class reweighting；synthetic 全部關閉。
- Current common neural defaults：hidden size 64、LR `1e-4`、train batch 256、eval batch
  2048、patience 200、repeat 5、max epochs 5000、validation ratio 0.2。
- Model seeds：`[9803, 38224, 8113, 4854, 98825]`。
- `.env` 會由 `run_experiment.py` 自動讀取，主要用來限制 OpenMP/MKL/OpenBLAS 等
  CPU threads 以及提供 API credentials；實驗設定以 CLI 為主。

### Metric Transition Warning

2026-07-30 才正式把舊 Accuracy / positive-class F1 protocol 換成 Balanced Accuracy /
Macro-F1。因此：

- 07-30 的 balanced-metric results 是目前 protocol 的基準。
- 07-29 以前的 selection numbers 仍可用來理解 loss behavior、relative trends 與
  diagnostics，但不能直接和目前 leaderboard 數字比較。
- FITCE / LAAT 尚未在新 balanced protocol 下完成完整 re-optimization；07-30 results
  是 controlled reweighting screen，不是最終 optimized leaderboard。

## Architecture

### Entrypoints

- `run_experiment.py`
  - 唯一 active experiment runner。
  - `--benchmark {acs,whyshift,tableshift,synthetic_ood}` 選 benchmark。
  - `--dataset` 可跑單一 dataset；省略則跑該 benchmark 全部 datasets。
  - `--method` 選 model/loss path。
  - PyTorch path：`mlp_ce`, `gradient_regularized_ce`, `fitce`, `laat`,
    `llm_select`。
  - sklearn path：`linear`, `llm_lasso`。
  - `--result-method` 只控制 artifact namespace，避免不同候選互相覆蓋。
- `rank_features.py`
  - 為 real-world datasets 產生 GPT ranking/scoring artifacts。
  - Default model 是 `gpt-5.4`；API URL/key 從 environment 讀取。
  - Synthetic datasets 不走 GPT API。
- `archive/`
  - 保存舊 runners、one-off scripts、早期 loss implementation 與歷史重現工具。
  - Active code 不應再 import archived modules。

### Shared Core

- `src/benchmark_config.py`
  - 建立 `DatasetSpec`，集中 benchmark order、split roles、feature indices、預設移除
    features、dataset caps、reweighting、cache/result paths 與 prior mapping。
  - Real-world `DatasetSpec.reweighting=True`；synthetic 為 `False`。
- `src/main.py`
  - PyTorch experiment orchestration、repeat/seeds、checkpoint selection、ID/OOD reporting、
    JSON serialization 與 plotting dispatch。
- `src/trainer.py`
  - Training loop、validation Balanced Accuracy/Macro-F1、early stopping、test evaluation、
    curves 與 SHAP collection。
- `src/loss.py`
  - `CrossEntropyCELoss`
  - `GradientRegularizedCELoss`
  - `LLMAttributionAlignedCELoss` (LAAT)
  - `FeatureImportanceTargetCELoss` (FITCE)
  - Shared logit-margin gradient、first-layer weight 與 diagnostic helpers。
- `src/metrics.py`
  - Binary Balanced Accuracy 與 Macro-F1。
- `src/utils.py`
  - Confidence intervals、ID/OOD aggregation、selection score、JSON helpers、curves/SHAP plots。
- `src/data_cache.py` / `src/paths.py`
  - Dataset cache naming、loading/building 與 canonical paths。
- `src/ranking.py`
  - Ranking artifact loading/validation、rank/score-to-weight conversion、LLM-Select feature
    selection。
- `src/feature_cards.py`, `src/ranking_prompts.py`, `src/ranking_generation.py`
  - Compact feature cards、四種 prompt templates、API generation/checkpointing。
- `src/semantic_features.py`
  - Semantic feature與one-hot/model-column之間的 mapping；共享 semantic prior 可展開回
    每一個 model column。

## Dataset Protocol

| Benchmark | Datasets | ID | OOD | Default cap |
|---|---:|---|---|---|
| ACS | 8 | source-state validation | held-out states | train/val 10000, tests full |
| WHYSHIFT | 2 | source-domain validation | held-out domains/states | train/val 8000, each test 8000 |
| TableShift | 8 | official `validation` | official `ood_test` | train 10000, OOD full |
| synthetic OOD | 5 | `train_env` validation | four controlled shifts | full |

### ACS

- Loader：`data/acs/dataset.py`；metadata：`data/acs/config.py`。
- Tasks：`acsincome`, `acsemployment`, `acsemploymentfiltered`,
  `acshealthinsurance`, `acsincomepovertyratio`, `acsmobility`,
  `acspubliccoverage`, `acstraveltime`。
- Feature removal、source split、state tests、optional resampling/standardization 都由共用
  loader 處理。

### WHYSHIFT

- Loader/config：`data/whyshift/dataset.py`, `data/whyshift/config.py`。
- Taxi：7 semantic features；目前 source domain 是 `bog`，domain 由 CSV filename 決定。
- Accident：source state `CA`，其他支援 states 作 OOD；preprocessed/one-hot feature names
  有明確 reference，可供 GPT feature cards 使用。
- 目前是 WHYSHIFT-style multi-target shift；不是只跑 paper 中單一 source-target pair。

### TableShift

- Loader/config：`data/tableshift/dataset.py`, `data/tableshift/config.py`。
- Datasets：`college_scorecard`, `diabetes_readmission`, `nhanes_lead`,
  `nhanes_cholesterol`, `anes`, `acsfoodstamps`, `brfss_diabetes`,
  `brfss_blood_pressure`。
- 使用 official TableShift preprocessing 與 split；model columns 排序後再轉 tensor，
  保持 ranking index 穩定。

### Synthetic OOD

- Loader：`data/synthetic_ood/dataset.py`。
- Variants：`simple`, `range`, `categorical_integer`, `categorical_onehot`,
  `multi_spurious`。
- 每個 dataset 都有 oracle prior、1 個 ID environment 與 4 個 OOD environments。
- Feature names 保留 `causal` / `spurious` / `noise` 前綴，強度寫在名稱尾端。
- `categorical_integer` 與 `categorical_onehot` 會移除兩個 continuous causal features，
  讓穩定 causal signal 只來自 categorical features；continuous shortcut 與 noise 仍保留。
- Categorical one-hot columns 共享同一 semantic feature weight，與 real-world mapping 一致。
- `range` 刻意保留 heterogeneous feature ranges，用來測試 raw-gradient alignment 是否能
  轉移成 contribution/SHAP alignment。

## Ranking And Prior Artifacts

- Real-world artifacts：`data/<benchmark>/<dataset>/rankings/`。
- Supported methods：`rank`, `score`, `score_all`, `seq`。
- Supported feature spaces：`semantic`（default）與 `model`。
- Supported weight modes：
  - `rank`：N 個 features 依名次轉成 `N, N-1, ..., 1`。
  - `score`：直接使用 `score_mean`，只適用 `score` / `score_all` artifacts。
- Prompt `.txt` 以 ranking method 為單位，不綁特定 LLM model。
- Compact feature card 包含 semantic name、自然語言描述、continuous range、categorical
  levels 與 boolean meaning。
- Semantic mode 先合併 one-hot columns，生成 prior 後再把相同 weight 展開到所有相關
  model columns；這是目前 default。
- Model-space probe 沒有穩定優於 semantic，但 API 成本明顯較高，因此不繼續擴大。
- Formal semantic GPT-5.4 artifacts 已涵蓋 18 real-world datasets 與四種 methods。
- README 保存 GPT token/cost tables；model-space `tableshift/anes seq` 因 quota 未完成，
  不應誤當成完整 comparison。

## Method Details

### MLP CE

- MLP + class-reweighted cross entropy on every real-world dataset。
- 不使用 LLM prior；real-world CE results 不計算 attribution distance。
- Synthetic CE 可載入 oracle weights 只用於 diagnostics，不改變 CE training loss。

### Gradient-Regularized CE

- CE 加上所有 input logit-margin gradients 的 L2 penalty，不使用 prior。
- 尚未正式 optimize。

### FITCE

- 以 binary logit margin `logit_1 - logit_0` 對 inputs 取 gradient。
- Gradient signal 是 batch/class-balanced mean squared gradient；以
  `softmax(log(signal) / tau)` 轉成 feature distribution。
- 對 normalized nonnegative prior 做 cross entropy alignment。
- 可同時 regularize gradient distribution 與 first-layer-weight distribution：
  `grad_scale=lambda*alpha`, `weight_scale=lambda*(1-alpha)`。
- `importance_scale` 目前只支援 `none` / `train_std`；`train_range` 實驗後已從 active
  code 刪除。
- `target_power` 已刪除；prior sharpness 保持不變，model-side sharpness 只由 `tau` 控制。
- `REG_SCALE` 可用 `reg_warmup_epochs` 線性 warm up。

### LAAT

- Current variant 使用 nonnegative prior，負值 clamp 到 0，符合本專案 GPT artifact
  semantics。
- Attribution 是 `input * logit-margin gradient`，逐 sample L2 normalize 後與
  `prior * input` 的 normalized target 做 MSE。
- 不使用 FITCE 的 `train_std` scaling。

### LLM-Select / LLM-Lasso

- LLM-Select：先依 `top_p` 或 score threshold 選 features，再跑 MLP CE。
  Score threshold 只適用 `score` / `score_all`。
- LLM-Lasso：sklearn L1 logistic regression，透過 feature rescaling 實作 LLM-weighted
  penalty factors；概念對齊 paper/official weighted-Lasso 方法，但不是完整複製 solver。
- 兩者 runner 已可跑所有 datasets，尚未完成 expanded-suite optimization。

### Parser Defaults Versus Recent Study Settings

Parser defaults 是通用可執行值，不代表已在新 balanced protocol 下 optimized。

| Method | Parser default | 2026-07-30 controlled screen |
|---|---|---|
| FITCE | `score_all/score`, lambda 16, alpha 0.75, reg 0, tau 1, no scaling | `rank/rank`, lambda 1, alpha 1, reg 0.1, tau 2, no scaling, warmup 0 |
| LAAT | `score_all/score`, reg 0 | `rank/rank`, reg 0.1, nonnegative variant |

## Outputs And Caches

- Results：`output/<method>/<benchmark>/<dataset>/result.json`。
- Plots：`output/<method>/<benchmark>/<dataset>/{curve,shap,accdelta}/`。
- Result schema v2 將 dataset、feature、training 與 loss 設定分開；`metadata` 只保留
  experiment method/baseline 與必要的 prior/selection 語意。`--record-best-grad-l2`
  可在每個 repeat 額外記錄 best checkpoint 的 raw logit-margin Grad-L2；固定使用
  training split，不套 FITCE 的 TrainStd scaling，class aggregation 跟隨 resolved
  reweighting，因此不重複記錄 signal/split/reweighting metadata。
- `output/` 是唯一標準 runtime-output root；不保留獨立 stdout log directory。
- Raw/cache directories 都被 git ignore：
  - `data/<benchmark>/<dataset>/raw/`
  - `data/<benchmark>/<dataset>/cache/`
- Cache name 只包含會改變 tensor data 的因素：removed features、resampling、
  standardization/preprocessing、train cap、test cap，以及 TableShift `oodtest` split token。
- 2026-08-01 搬移前清除 20 個舊 cache variants，共 `989.3 MiB`；23/23 現行 default
  caches 都保留。當時 cache 總量約 `1.9 GiB`，raw data 約 `19 GiB`，project 約
  `22 GiB`。

## Current Experimental State

### Balanced-Metric Reweighting And Baselines (2026-07-30)

這是目前唯一完全符合 Balanced Accuracy / Macro-F1 protocol 的 CE/FITCE/LAAT
real-world comparison。設定是 18 datasets、R3、P200、curves + validation SHAP。

| Method | ID BAcc | ID Macro-F1 | ID Sum | OOD Selection |
|---|---:|---:|---:|---:|
| CE | 0.7212 | 0.6870 | 1.4082 | **2.6360** |
| FITCE | 0.7043 | 0.6679 | 1.3722 | 2.5851 |
| LAAT | 0.7185 | 0.6836 | 1.4021 | 2.6261 |

- Reweighting on 相對 off 的 macro ID-sum changes：CE `+0.0490`、FITCE `+0.0722`、
  LAAT `+0.0416`。
- On 在 dataset means 上勝出 CE 16/18、FITCE 16/18、LAAT 13/18。
- 最大 ID drop 是 LAAT / `brfss_blood_pressure` 的 `-0.0082`，只占 off score
  `0.64%`；所有負向 paired R3 CI 都包含 0。
- 因此 frozen policy 是所有 real-world datasets 統一 reweighting on；只用 ID 決策，
  不用 OOD test information。
- 完整 overall/per-dataset tables：`dev_log/2026-07-30.md` 的
  `Real-World Reweighting Policy and Balanced-Metric Results`。

### FITCE Loss Findings That Remain Active

1. **Use logit margin.** 07-09 synthetic check 顯示 `logit_1-logit_0` 比只看 positive
   logit 更符合 binary decision，且大幅改善高 regularization FITCE。
2. **Use per-class balancing when reweighted.** CE 與 attribution terms 都先在 present
   classes 內平均，再跨 class 平均。
3. **Tau is the model-side sharpness knob.** `tau=1` 等價舊 sum normalization；synthetic
   simple 在 `tau=2` 最佳並接近直接移除 spurious feature。`target_power` 因功能重疊而刪除。
4. **Gradient term is目前最有力的 synthetic mechanism.** Simple dataset grad-only
   `reg=10,tau=2` 幾乎達到 remove-spurious oracle；weight-only 即使擴到 reg 200 仍較差。
5. **Scale matters.** FITCE 在四個 standard-scale synthetic variants 接近 oracle，但
   `range` 顯示 raw gradient alignment 不保證 SHAP/contribution alignment。
6. **`train_std` 能修 synthetic range，但沒有穩定改善 real-world。** `train_range` 也能
   修 range stress test，但 real-world test 沒有價值，功能已刪除；dev-log 結果保留。
7. **Real-world optimum favors weaker regularization.** 舊 metric 下 reg 由 10 降到 0.1
   是最大 improvement；tau 2 優於較低 tau；warmup 只有小幅 gain，且多數 selected
   checkpoints 尚未到 full reg。
8. **Exact CE reduction verified.** FITCE `REG_SCALE=0` 在兩個 priority datasets 上與 CE
   aggregate/per-seed values 完全一致，排除 runner-path discrepancy。

### FITCE Versus LAAT And Distance Findings

- 07-02 fair sweep：FITCE 的 Grad RankDist/WeightDist 明顯低於 LAAT，SHAP RankDist 也
  較低；LAAT 的 SHAP WeightDist 與 downstream selection 較好。
- 這表示 FITCE 可以成功 fit local gradient prior，但較低 distance 不必然帶來更高
  predictive performance。
- 07-30 priority-dataset diagnostics 再次顯示 selection deficit 與 gradient distance 沒有
  一致關係；`acspubliccoverage` 是最大 performance failure，卻有很好的 gradient
  alignment。
- SHAP WeightDist 對 selection deficit 有部分訊號，但關係主要被最大 failures 驅動，
  不是穩定 monotonic proxy。
- Current interpretation：real-world FITCE gap 可能同時來自 prior quality、local-to-global
  attribution transfer 與 optimization interaction，而不是單純「loss 沒有 fit prior」。

### Synthetic Suite Findings

- 08-26 用 Balanced Accuracy/Macro-F1 重跑 CE/FITCE R3；categorical integer/one-hot
  移除兩個 continuous causal features，只保留 categorical causal signal、continuous
  shortcut 與 noise。Five-dataset macro OOD selection 是 Remove Spurious `3.6296`、
  FITCE `3.5349`、CE `2.2231`；FITCE-minus-Optimal 是 `-0.0947`。
- FITCE 對五個 datasets 都是 ID 輸 CE、OOD mean/worst 贏 CE；macro ID selection gap
  `-0.1533`，OOD selection gap `+1.3118`，支持強 controlled shift 下 oracle prior 有效。
- Categorical one-hot FITCE OOD selection 穩定在約 `3.62`；integer encoding 是
  `2.9913 +/- 1.6408`，有明顯 seed instability，仍是 representation/optimization concern。
- `range` 是關鍵 counterexample：未 scaling 時 FITCE gradient distance 很低，但 SHAP
  distance 與 OOD performance 明顯較差，證明 coordinate scale 會破壞 gradient-to-use
  transfer。
- Synthetic results 使用 oracle prior，因此能排除 GPT prior quality；FITCE real-world
  failure 不能只歸因於 core loss 完全無效。

### Current Open Questions

1. 在 frozen balanced-metric + all-reweighted protocol 下重新 optimize FITCE / LAAT，
   synthetic 不參與 hyperparameter selection。
2. 優先診斷 FITCE performance gap 最大且 variance 高的 datasets，尤其
   `acspubliccoverage`, `acshealthinsurance`, `acstraveltime`, `acsmobility`。
3. 分離 prior quality 與 optimization：比較不同 rank/score artifacts，同時檢查 Grad/SHAP
   RankDist 與 WeightDist，而不是只看 selection score。
4. 評估是否需要更直接的 contribution-aware target；任何 synthetic-specific zero-mass
   penalty 都不能直接當成 general solution。
5. 依序 optimize 尚未完成的 simple baselines：gradient-regularized CE、LLM-Select、
   LLM-Lasso，再實作 tree/SVM/TabPFN baselines。
6. 最終 README leaderboard 只能使用 frozen protocol 下重新跑出的 R5 results；目前
   07-30 R3 table 是 protocol screen。

## Recent Timeline

### 2026-08-26

- 依 CE 的 `ID (BAcc + Macro-F1) - OOD (BAcc + Macro-F1)` 排 real-world OOD severity，
  並比較 FITCE ID/OOD gaps。
- 重跑五個 synthetic datasets 的 CE/FITCE；categorical variants 移除 continuous causal
  features，確認 FITCE 在更強 categorical shortcut shift 下由 ID 落後轉成 OOD 領先。

### 2026-08-01

- 整理 `context.md`，以 current protocol 和近期結論為主。
- 刪除不會再被現行 path 命中的舊 data caches，釋放 `989.3 MiB`；保留全部 23 個
  default caches、小型 smoke caches 與 synthetic remove-feature variants。

### 2026-07-30

- 換成 Balanced Accuracy / Macro-F1，TableShift ID 改用 official validation。
- CE/FITCE/LAAT 完成 real-world reweighting on/off screen、curves、validation SHAP。
- 只根據 ID 決定所有 real-world datasets 統一 reweighting on。
- 驗證 FITCE reg=0 與 CE 完全等價，修正 diagnostics 不應更新 BatchNorm state。
- 分析 FITCE selection deficit 與 Grad/SHAP distances；distance 不是可靠 performance proxy。
- 詳細紀錄：`dev_log/2026-07-30.md`。

### 2026-07-29

- 在舊 Accuracy/F1 protocol 下重測 CE/FITCE reweighting；得到 dataset/method-specific
  choices，但已被 07-30 的 balanced ID-only shared policy 取代。
- Class imbalance 不是 FITCE deficit 的唯一解釋。

### 2026-07-27

- Real-world FITCE sweep 顯示較小 `REG_SCALE=0.1`、`tau=2` 最好；warmup 200 有小幅
  improvement，但仍低於 CE，且常在 warmup 完成前選 checkpoint。
- 這些數字使用舊 metric，保留作 loss-behavior evidence，不是 current final setting。

### 2026-07-23

- `train_std` / `train_range` 都能修 synthetic range stress test。
- `train_range` 未穩定優於 `train_std`，之後 real-world test 不佳而從 active code 刪除。

### 2026-07-22

- 建立五個 balanced synthetic variants 的 CE/LAAT/FITCE/remove-spurious comparison。
- FITCE 在四個 variants 接近 oracle；`range` 揭露 raw gradient 與 SHAP contribution
  mismatch。

### 2026-07-20

- Loss fixes 後 patience 50/100/200 screen；200 是三種 neural methods 的最佳 tested
  lower-patience setting，因此後續固定 P200 省時間。
- 六種 rank/score configurations 的 real-world gate 顯示 prior choice 會影響方法排名，
  FITCE 仍多數低於 CE/LAAT。
- FITCE rank diagnostics 顯示部分 failure 不是因 gradient target 完全沒 fit。

### 2026-07-19

- FITCE first-layer-weight-only sweep 擴到 reg 200；雖優於 CE/LAAT synthetic baseline，
  仍不及 grad-only 與 remove-spurious oracle。

### 2026-07-09

- 改用 binary logit-margin gradients。
- LAAT 改成 nonnegative-prior variant。
- 加入 gradient softmax temperature；`tau=2` synthetic 最佳。
- 刪除 `target_power`，保留 tau 作唯一 distribution-sharpness control。

### 2026-07-02

- Fair FITCE-vs-LAAT alignment sweep：FITCE 贏 gradient alignment，LAAT 贏 SHAP
  magnitude alignment 與 downstream performance。
- `train_std` 改善部分 FITCE behavior，但沒有建立 universal real-world win。
- Synthetic oracle comparison開始把 performance 與 distance 同時記錄。

### 2026-06-30

- Small-LR x patience sweep：降低 LR 沒有改善 FITCE relative position；長 patience 在當時
  protocol 下較好，但後續為 runtime 與新 loss fixes 固定 P200。

### 2026-06-29

- Semantic vs model feature-space probe：performance 大致相近，model-space API cost
  明顯更高，因此 semantic 維持 default。
- Model-space completed artifacts 約 `$44.04`；semantic formal artifacts 約 `$11.16`。

## Earlier History Summary

### June 2026

- 06-22：loss/runner 簡化成目前核心 methods，README 整理 GPT token/cost。
- 06-10：18 real-world dataset suite 的早期 FITCE/LAAT optimization；使用舊 metric，
  只保留作 search-history reference。
- 06-08：TableShift 8 datasets、compact feature cards、semantic-to-model mapping、四種 GPT
  prompt methods 完成整合與 smoke tests。
- 06-04：WHYSHIFT Taxi/Accident 加入；確認 8000/8000 caps 與 cross-benchmark runner。
- 06-02：TEST_STATES 修正後重跑早期 ACS FITCE optimization。
- 06-01：早期 ACS LAAT/ranking/reg-scale comparison。
- 05-31：舊 README dataset conclusions 與 early FITCE settings 備份。

### May 2026

- 完成四種 ranking methods、direct-score modes、GPT ranking cost、ACS task runners 與
  early CE/FITCE benchmark tables。
- 依序搜尋 FITCE lambda/alpha、suppress/remove threshold、target power、REG_SCALE；這些
  結果使用舊 TEST_STATES、舊 loss terms 或舊 metrics，不能作 current optimum。
- Consolidate ACS runners/dataset loaders/config、建立 cache naming、JSON results、CI metrics，
  並把 one-off scripts/logs 移入 archive 或刪除。
- 詳細索引：`2026-05-04`, `05-10`, `05-11`, `05-14`, `05-15`, `05-20`,
  `05-24`, `05-27`, `05-31`。

### April 2026

- 建立最早期 synthetic OOD、ACS data/state sweeps、gradient/weight regularization 與 GPT
  ranking/scoring prototypes。
- 這些實驗的 dataset split、loss implementation、metric 與 runner 架構都已多次更改，
  目前只作 provenance；不要直接引用數字作 final comparison。
- 詳細索引：`2026-04-01`, `04-02`, `04-12`, `04-14`, `04-18`, `04-19`,
  `04-23`, `04-25`, `04-26`, `04-29`。

## Dev-Log Index

| Date | Main topic | Current status |
|---|---|---|
| 2026-08-26 | Real-world OOD severity; categorical-causal synthetic rerun | Current case-study evidence |
| 2026-07-30 | Balanced metrics, shared reweighting, FITCE diagnostics | **Current protocol** |
| 2026-07-29 | Old-metric CE/FITCE reweight screen | Superseded by 07-30 policy |
| 2026-07-27 | FITCE reg/tau/warmup | Mechanism reference; old metrics |
| 2026-07-23 | train-range scaling | Negative/ablation evidence; feature removed |
| 2026-07-22 | Original five synthetic datasets | Historical pre-categorical-ablation evidence |
| 2026-07-20 | P200 gate, priors, real-world diagnostics | Current P200 rationale; old metrics |
| 2026-07-19 | FITCE weight-only | Current ablation evidence |
| 2026-07-09 | Margin gradients, nonnegative LAAT, tau, target-power removal | Current loss-design basis |
| 2026-07-02 | FITCE/LAAT distance comparison | Current diagnostic basis; old metrics |
| 2026-06-30 | Small-LR/patience sweep | Historical search |
| 2026-06-29 | Semantic/model feature-space comparison | Current semantic-default rationale |
| 2026-06-22 | Runner/loss cleanup and GPT cost | Architecture/cost reference |
| 2026-06-10 | Expanded-suite early optimization | Historical search |
| 2026-06-08 | TableShift/ranking integration | Dataset/ranking provenance |
| 2026-06-04 | WHYSHIFT integration | Dataset provenance |
| 2026-06-02 | ACS rerun after TEST_STATES fix | Historical search |
| 2026-06-01 | ACS FITCE/LAAT optimization | Historical search |
| 2026-05-31 and earlier | Early ACS/synthetic development | Brief provenance only |

## Related Work

`codex_notes/related_work.md` 是 canonical survey notes，包含 paper core idea、與本研究的
相同點/差異、可主張 novelty 與 implementation implications。目前三篇：

- *Large Language Models as Attribution Regularizers for Efficient Model Training* (LAAT)
- *LLM-Select: Feature Selection with Large Language Models*
- *LLM-Lasso: A Robust Framework for Domain-Informed Feature Selection and Regularization*

原始 PDFs 位於 `codex_notes/surveyed_papers/`。新增 paper 時，優先更新
`related_work.md`，context 只保留研究方向與 active implementation 的摘要。
