## OOD Generalization Experiments

這份 README 只保留公開 benchmark 摘要。研究脈絡、完整實驗表格、
implementation notes、後續 reference 都放在本地私有筆記 `codex_notes/`。

## Dataset Conclusions

同一個 method 在不同 dataset 的參數設定不同，因此 benchmark 以
method setting 為 row、dataset metric group 為 columns。`--` 表示該設定未在該
dataset 上評估或不適用。

<table>
  <thead>
    <tr>
      <th rowspan="2">method / setting</th>
      <th colspan="3">synthetic_ood</th>
      <th colspan="3">ACSIncome</th>
    </tr>
    <tr>
      <th>ID</th>
      <th>OOD MEAN</th>
      <th>OOD WORST</th>
      <th>ID</th>
      <th>OOD MEAN</th>
      <th>OOD WORST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CrossEntropyLoss</code>, no balancing</td>
      <td align="right">0.9693</td>
      <td align="right">0.6828</td>
      <td align="right">0.6828</td>
      <td align="right">0.9140</td>
      <td align="right">0.7063</td>
      <td align="right">0.6166</td>
    </tr>
    <tr>
      <td><code>CrossEntropyLoss</code>, ACSIncome class reweighting</td>
      <td align="center">--</td>
      <td align="center">--</td>
      <td align="center">--</td>
      <td align="right">0.8490</td>
      <td align="right">0.7580</td>
      <td align="right"><strong>0.7232</strong></td>
    </tr>
    <tr>
      <td><code>FeatureImportanceTargetCELoss</code>, synthetic perfect ranking</td>
      <td align="right">0.9473</td>
      <td align="right"><strong>0.8758</strong></td>
      <td align="right"><strong>0.8758</strong></td>
      <td align="center">--</td>
      <td align="center">--</td>
      <td align="center">--</td>
    </tr>
    <tr>
      <td><code>FeatureImportanceTargetCELoss</code>, ACSIncome scale-aware GPT prior</td>
      <td align="center">--</td>
      <td align="center">--</td>
      <td align="center">--</td>
      <td align="right"><strong>0.8553</strong></td>
      <td align="right"><strong>0.7624</strong></td>
      <td align="right">0.7212</td>
    </tr>
  </tbody>
</table>

Notes:

- `synthetic_ood` has one OOD test environment, so `OOD MEAN` and `OOD WORST`
  are identical for the reported averages.
- On `synthetic_ood`, `FeatureImportanceTargetCELoss` clearly improves OOD accuracy over
  CE when the feature ranking is correct, while sacrificing some ID accuracy.
- On `ACSIncome`, the best confirmed scale-aware GPT prior beats the class-reweighted CE
  baseline on ID and OOD MEAN, but CE still has the best OOD WORST.

### FeatureImportanceTargetCELoss Settings

- `synthetic perfect ranking`:
  - dataset: `synthetic_ood`
  - model: `MLP`
  - seeds: `[9803, 38224, 8113, 4854, 98825]`
  - feature weights: `causal_main=3`, `causal_aux=2`,
    `noise_1/2/3=1`, `spurious_main=0`
  - `REG_SCALE=0.5`
  - `grad_scale=2.0`, `weight_scale=3.0`, `suppress_scale=6.0`,
    `target_power=1.0`
- `ACSIncome scale-aware GPT prior`:
  - dataset: `ACSIncome`
  - model: `MLP`
  - train/validation state: `PR`
  - OOD tests: 46 held-out states
  - seeds: `[9803, 38224, 8113, 4854, 98825]`
  - raw inputs with `DATASET_CONFIG={"resampling": False, "standardize": False}`
  - class-balanced CE term with `reweighting=True`
  - structured GPT score weights:
    `SCHL=9`, `WKHP=8`, `OCCP=8`, `AGEP=7`, `COW=6`,
    `MAR=5`, `RELP=4`, `POBP=3`, `SEX=2`, `RAC1P=1`
  - `REG_SCALE=1.0`
  - `grad_scale=2.0`, `weight_scale=3.0`, `suppress_scale=6.0`,
    `target_power=1.0`, `importance_scale="train_std"`

## Appendix

### Environment Setup
```bash
conda create -n ood python=3.10.19
conda activate ood
pip install -r requirements.txt
```

### Run Experiments
- Main Experiment
```bash
python run_{dataset_name}.py
```

- ACSIncome: Check distributional shift & Causal features.
```python
# use all features
ID_base, OOD_MEAN_base, OOD_WORST_base = main(
    DATASET, TRAIN_VAL_STATE, TEST_STATES, FEATURE_INDEX, [], FEATURE_LOSS_WEIGHTS,
    TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
    PATIENCE=PATIENCE, REPEAT=REPEAT, device=device,
)
print(f"### Use all features:\n- ID: {ID_base:.4f}\n- OOD MEAN: {OOD_MEAN_base:.4f}\n- OOD WORST: {OOD_WORST_base:.4f}")

# leave one feature out
rmfeature_accdelta = {}
for feat_i, feat in FEATURE_INDEX.items():
    REMOVED_FEATURE_INDICES = [feat_i]
    ID_rm, OOD_MEAN_rm, OOD_WORST_rm = main(
        DATASET, TRAIN_VAL_STATE, TEST_STATES, FEATURE_INDEX, REMOVED_FEATURE_INDICES, FEATURE_LOSS_WEIGHTS,
        TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
        PATIENCE=PATIENCE, REPEAT=REPEAT, device=device,
    )
    print(f"### Remove {feat}:\n- ID: {ID_rm:.4f}\n- OOD MEAN: {OOD_MEAN_rm:.4f}\n- OOD WORST: {OOD_WORST_rm:.4f}")

    rmfeature_accdelta[feat] = {
        "ID": (ID_rm - ID_base),
        "OOD MEAN": (OOD_MEAN_rm - OOD_MEAN_base),
        "OOD WORST": (OOD_WORST_rm - OOD_WORST_base)
    }

# check distributional shift, causal/non-causal features
plot_accdelta_bars(rmfeature_accdelta, ID_base, OOD_MEAN_base, OOD_WORST_base)
```
