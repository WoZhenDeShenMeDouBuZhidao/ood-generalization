## Notes

### Files
- `src/*`: 訓練核心邏輯，視覺化結果
- `synthetic_ood/*`: synthetic OOD dataset 與實驗結果
- `{dataset name}/plots/`: accuracy, gradient l2 curves, SHAP values
- `{dataset name}/data/`: save dataset，節省每次下載/生成 data 的時間
- `{dataset name}/dataset.py`: load data 的 functions
- `run_{dataset name}.py`: 在指定 dataset 上跑實驗

### Experimental Setups
- `REPEAT` 參數一律設成 $5$，**ID** / **OOD MEAN** / **OOD WORST** accuracy 皆是 $5$ 次實驗的平均值
- 為了確保比較公平性，以下超參數也不可更改：
  - `TRAIN_BATCH` = $256$
  - `EVAL_BATCH` = $2048$
  - `LR` = $1*10^{-4}$
  - `PATIENCE` = $500$
  - `REMOVED_FEATURE_INDICES` = []（不直接丟掉 feature）

### Goal
- *Baseline* 是單純用 CE loss 訓練出來的結果
- 目標是透過更改 loss function 調整每個 feature 對 model 的 improtance，達成 OOD generalization
  - 新增 loss term
  - 改 loss term 的 `REG_SCALE`



## Benckmark Results

### ACSIncome
ACSIncome 沒有像 **Synthetic OOD** 那樣可直接使用的 perfect ranking，因此這部分的核心問題不是 loss 能不能工作，而是「LLM 給出的 ranking 是否足夠接近資料中的可遷移結構」。

這裡的報告流程分成三步：
- 先建立 baseline
- 再用 leave-one-out ranking 當診斷工具，確認新 loss 在 ranking 夠準時是否真的有幫助
- 最後回到實際可用的 GPT ranking，檢查 end-to-end pipeline 是否仍然成立

#### 1. Baseline

原始 benchmark baseline:

| Name | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ------ | ------------ | ------------- |
| *Baseline* | 0.9132 | 0.7038 | 0.6145 |

為了和後續 transfer experiment 做公平比較，我們另外固定 model seeds `[9803, 38224, 8113, 4854, 98825]` 重跑 baseline：

| Name | Seeds | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ----- | ------ | ------------ | ------------- |
| `fixed-seed baseline` | 5 | 0.9140 | 0.7063 | 0.6166 |

後續 ACSIncome 的比較，主要都以這組 fixed-seed baseline 為準。

#### 2. Leave-One-Out Ranking: 診斷新 loss 是否有用

先用 MLP + 純 CE loss 做 leave-one-out 實驗，把一個 feature 拿掉看 ID / OOD MEAN / OOD WORST 的變化，結果畫在 `acsincome/plots/accdelta/accdelta_bars.png`。

這個 ranking 不是正式可部署方案，但它可以回答一個重要問題：
- 如果 ranking 夠接近資料中的可遷移結構，`FeatureImportanceTargetCELoss` 能不能在 ACSIncome 上帶來收益？

leave-one-out ranking:
- `SCHL > WKHP > SEX > AGEP > POBP > MAR > RELP > OCCP > COW > RAC1P`

固定設定：
- `MLP`
- `FeatureImportanceTargetCELoss`
- `REG_SCALE = 0.5`
- `grad_scale = 2.0`
- `weight_scale = 3.0`
- `suppress_scale = 6.0`
- `target_power = 1.0`

結果：

| Setting | Ranking Source | Weighting | Seeds | **ID** | **OOD MEAN** | **OOD WORST** |
| ------- | -------------- | --------- | ----- | ------ | ------------ | ------------- |
| `fixed-seed baseline` | none | CE only | 5 | 0.9140 | 0.7063 | 0.6166 |
| `leave_one_out_all_positive` | leave-one-out | `10 > 9 > ... > 1` | 3 | 0.9142 | 0.7015 | 0.6087 |
| `leave_one_out_top5` | leave-one-out | top-5 positive, rest 0 | 3 | 0.9131 | 0.7059 | 0.6117 |
| `leave_one_out_top7` | leave-one-out | top-7 positive, bottom-3 suppressed | 3 | 0.9157 | 0.7155 | 0.6287 |
| `leave_one_out_top7` | leave-one-out | top-7 positive, bottom-3 suppressed | 5 | 0.9157 | 0.7129 | 0.6240 |

重點觀察：
- `leave_one_out_top7` 明確高於 fixed-seed baseline，表示 synthetic 上找到的 loss 超參數可以轉移到 ACSIncome。
- 只把 ranking 換對但不 suppress (`leave_one_out_all_positive`) 還是不夠。
- suppress 太多也不行；`top5` 幾乎回到 baseline。
- 目前 ACS 上最好的平衡點是：
  - ranking 夠接近資料結構
  - 同時只 suppress 最尾端幾個 feature

這一步的結論很重要：
- 新 loss 本身不是沒用
- 它在 ACSIncome 上是有潛力的
- 但前提是 ranking 必須夠準

#### 3. GPT Ranking V1: Score-Based Ranking / Threshold Suppression

專案真正的目標不是依賴 leave-one-out，而是依賴 LLM / GPT 對 dataset features 的理解來產生 ranking。

最先使用的 GPT 訊號有兩種：
- 直接生成 ranking:
  - `SCHL > OCCP > WKHP > AGEP > COW > MAR > POBP > SEX > RAC1P > RELP`
- per-feature score 後再排序：
  - `SCHL 7.4 > OCCP 7.1 > WKHP 6.8 > AGEP 6.2 > COW 5.9 > SEX 4.4 > MAR 4.3 > POBP 3.9 > RELP 2.9 > RAC1P 2.9`

我們先用 score-based ranking 做三種 setting：
- `score_raw_all_positive`
- `score_threshold_4p0`
- `score_threshold_4p5`

結果：

| Setting | Idea | Seeds | **ID** | **OOD MEAN** | **OOD WORST** |
| ------- | ---- | ----- | ------ | ------------ | ------------- |
| `fixed-seed baseline` | CE only | 5 | 0.9140 | 0.7063 | 0.6166 |
| `score_raw_all_positive` | 全部 feature 給正 target | 3 | 0.9120 | 0.6998 | 0.6063 |
| `score_threshold_4p0` | suppress `POBP` / `RELP` / `RAC1P` | 3 | 0.9125 | 0.6996 | 0.6038 |
| `score_threshold_4p5` | 再額外 suppress `SEX` / `MAR` | 3 | 0.9096 | 0.6829 | 0.5779 |

重點觀察：
- 這三組都沒有超過 baseline。
- 把尾端 features 壓太多只會更差。
- 單靠「GPT score ranking + threshold suppression」還不足以讓新 loss 發揮。

#### 4. GPT Ranking V2: Structured Ranking + Coarse Grouping

後續改用 `gpt_acsincome.json` 的 structured output。這版 prompt 除了 ranking 外，還要求模型輸出：
- `spurious_start_rank`
- high-confidence front features
- high-risk spurious features
- coarse grouping 建議

本輪 GPT ranking:
- `SCHL > WKHP > OCCP > AGEP > COW > MAR > RELP > POBP > SEX > RAC1P`

對應 3-group:
- `group_1 = [SCHL, WKHP, OCCP]`
- `group_2 = [AGEP, COW, MAR]`
- `group_3 = [RELP, POBP, SEX, RAC1P]`

先做 coarse grouping sweep：

| Setting | Weights | Seeds | **ID** | **OOD MEAN** | **OOD WORST** | Note |
| ------- | ------- | ----- | ------ | ------------ | ------------- | ---- |
| `gpt_2group_110` | rank `1-6 = 1`, rank `7-10 = 0` | 1 | 0.9090 | 0.6818 | 0.5766 | early stop |
| `gpt_3group_320` | `group_1 = 3`, `group_2 = 2`, `group_3 = 0` | 3 | 0.9092 | 0.6865 | 0.5822 | completed |
| `gpt_3group_310` | `group_1 = 3`, `group_2 = 1`, `group_3 = 0` | 1 | 0.9107 | 0.6894 | 0.5863 | early stop |
| `gpt_3group_321` | `group_1 = 3`, `group_2 = 2`, `group_3 = 1` | 3 | 0.9123 | 0.6990 | 0.6033 | completed |
| `gpt_3group_321` | `group_1 = 3`, `group_2 = 2`, `group_3 = 1` | 5 | 0.9126 | 0.6972 | 0.5999 | confirmed |
| `gpt_4group_4321` | ranks `[1,2]=4`, `[3,4]=3`, `[5,6,7]=2`, `[8,9,10]=1` | 1 | 0.9123 | 0.6989 | 0.6021 | early stop |

重點觀察：
- `gpt_3group_321 > gpt_3group_310 > gpt_3group_320`。
- 這代表 GPT ranking 的尾端 features 不能被全部當成 pure-suppressed features 直接壓成 0。
- 但即使改成比較溫和的 `group_3 = 1`，5-seed 結果仍低於 fixed-seed baseline：
  - baseline: `0.7063 / 0.6166`
  - `gpt_3group_321`: `0.6972 / 0.5999`
- `gpt_4group_4321` 的第一個 seed 幾乎和 `gpt_3group_321` 打平，沒有顯示更細 bucket 的明顯優勢。

#### 5. ACSIncome Conclusion

這一輪 ACSIncome 的結論可以整理成一句話：

- `FeatureImportanceTargetCELoss` 在 ACSIncome 上不是沒用，但它目前無法和現有 GPT ranking 組合成有效的 end-to-end pipeline。

更具體地說：
- 若 ranking 接近資料中的可遷移結構，新的 loss 確實有幫助：
  - `leave_one_out_top7` 可從 `0.7063 / 0.6166` 提升到 `0.7129 / 0.6240`
- 但一旦回到實際可用的 GPT ranking，不論是 score-based thresholding 還是 structured coarse grouping，都沒有超過 baseline。
- 因此目前 ACSIncome 的主要瓶頸不是 loss 超參數，而是 GPT ranking / grouping 的品質。

目前最合理的研究結論：
- synthetic 上的 loss 設計是成立的
- ACSIncome 上也有跡象顯示它在「ranking 夠準」時能工作
- 但以目前的 GPT ranking quality，這條 end-to-end 路線還沒有被驗證成功

所以下一步若要繼續推進，優先順序應該是：
- 改善 prompt 與 ranking 產生流程
- 改善 feature grouping definition
- 而不是繼續微調 `2-group / 3-group / 4-group` 或 loss 超參數

### Synthetic OOD
Train/validation 使用穩定 causal features 與高度相關 spurious feature；OOD test 將 `spurious_main` 的相關性從 `+0.95` 翻成 `-0.95`。以下結果皆為 `python run_synthetic_ood.py` 跑 `5` 次的平均值。

Feature weight ranking 搭配 `feature_grad_ce`:
- *Favor spurious*: `spurious_main` > `noise` > `causal_main` > `causal_aux`
- *Baseline*: all equal rank, `reg_scale = 0` 等同 CE loss
- *Favor causal*: `causal_main` > `causal_aux` > `noise` > `spurious_main`
- *Optimal*: remove `spurious_main`, `noise`, all equal rank

| Name | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ------ | ------------ | ------------- |
| *Favor spurious* | 0.9691 | 0.6390 | 0.6390 |
| *Baseline* | 0.9693 | 0.6828 | 0.6828 |
| *Favor causal (reg=1)* | 0.9694 | 0.7090 | 0.7090 |
| *Optimal* | 0.9214 | 0.9323 | 0.9323 |

#### 1. Ranking Constraint vs. Target Distribution

公平比較設定：
- perfect ranking target: `causal_main = 3`, `causal_aux = 2`, `noise = 1`, `spurious_main = 0`
- model: `MLP`
- `REG_SCALE = 0.5`
- fixed model seeds: `[9803, 38224, 8113, 4854, 98825]`

gradient-based 對照：

| Name | Constraint | **ID** | **OOD MEAN** | **OOD STD** |
| ---- | ---------- | ------ | ------------ | ----------- |
| `FeatureGradCELoss` | pairwise ranking | 0.9691 | 0.6956 | 0.0057 |
| `FeatureImportanceTargetCELoss (grad-only)` | target distribution | 0.9698 | 0.7108 | 0.0116 |

weight-based 對照：

| Name | Constraint | **ID** | **OOD MEAN** | **OOD STD** |
| ---- | ---------- | ------ | ------------ | ----------- |
| `FirstLayerWeightCELoss` | pairwise ranking | 0.9684 | 0.7031 | 0.0161 |
| `FeatureImportanceTargetCELoss (weight-only)` | target distribution | 0.9682 | 0.7147 | 0.0257 |

重點觀察：
- 在固定其他參數後，無論是 gradient-based 還是 weight-based，`target distribution` 都比 `pairwise ranking` 有更高 OOD。
- gradient 路線的差距更乾淨：固定 seed 的 OOD SHAP 中，`FeatureGradCELoss` 仍把 `spurious_main` 排在第 2，但 `FeatureImportanceTargetCELoss (grad-only)` 已改成 `causal_main > causal_aux > noise`。
- weight 路線的差距較小，但平均 OOD 仍是 distribution 版本較高；代表單純要求「排序正確」不如直接指定「importance mass 怎麼分配」。

#### 2. Add `suppress_scale` and `target_power`

在接受「target distribution 比 pairwise ranking 更好」後，下一步就是把 `FeatureImportanceTargetCELoss` 補成完整版本：
- `grad_scale`
- `weight_scale`
- `suppress_scale`
- `target_power`

核心動機：
- 單靠 target distribution，仍可能讓 suppressed features 保留過多 mass
- 因此需要 `suppress_scale` 直接懲罰 suppressed features 的 total mass
- `target_power` 則控制 target distribution 的尖銳程度

#### 3. Perfect Ranking Sensitivity: `FEATURE_LOSS_WEIGHTS` Sweep (`MLP` + `FeatureImportanceTargetCELoss`)

固定使用：
- ranking: `causal_main > causal_aux > noise > spurious_main`
- dataset: `synthetic_ood`
- fixed model seeds: `[9803, 38224, 8113, 4854, 98825]`
- common `LOSS_KWARGS` baseline:
  - `grad_scale = 1.0`
  - `weight_scale = 3.0`
  - `suppress_scale = 6.0`
  - `target_power = 1.0`

`FEATURE_LOSS_WEIGHTS` sweep (`REG_SCALE = 2.0`):

| Name | Weights | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ------- | ------ | ------------ | ------------- |
| `weights_balanced_3210` | `3, 2, 1, 0` | 0.9395 | 0.8717 | 0.8717 |
| `weights_scale_equiv_3210x10` | `30, 20, 10, 0` | 0.9395 | 0.8717 | 0.8717 |
| `weights_causal_heavier_6310` | `6, 3, 1, 0` | 0.9410 | 0.8702 | 0.8702 |
| `weights_noise_light_631p50` | `6, 3, 0.5, 0` | 0.9417 | 0.8614 | 0.8614 |

#### 4. `REG_SCALE` Sweep

`REG_SCALE` sweep (`FEATURE_LOSS_WEIGHTS = 3, 2, 1, 0`):

| Name | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ------ | ------------ | ------------- |
| `reg_0.5` | 0.9480 | 0.8751 | 0.8751 |
| `reg_1.0` | 0.9427 | 0.8565 | 0.8565 |
| `reg_2.0` | 0.9395 | 0.8717 | 0.8717 |
| `reg_4.0` | 0.9363 | 0.8711 | 0.8711 |

#### 5. `LOSS_KWARGS` Sweep

`LOSS_KWARGS` sweep (`FEATURE_LOSS_WEIGHTS = 3, 2, 1, 0`, `REG_SCALE = 2.0`):

| Name | `grad_scale` | `weight_scale` | `suppress_scale` | `target_power` | **ID** | **OOD MEAN** | **OOD WORST** |
| ---- | ------------ | -------------- | ---------------- | -------------- | ------ | ------------ | ------------- |
| `loss_weak_w1_s2_p1_g1` | 1.0 | 1.0 | 2.0 | 1.0 | 0.9451 | 0.8726 | 0.8726 |
| `loss_suppress_w1_s6_p1_g1` | 1.0 | 1.0 | 6.0 | 1.0 | 0.9408 | 0.8698 | 0.8698 |
| `loss_strong_w3_s6_p1_g1` | 1.0 | 3.0 | 6.0 | 1.0 | 0.9395 | 0.8717 | 0.8717 |
| `loss_strong_w3_s6_p0p5_g1` | 1.0 | 3.0 | 6.0 | 0.5 | 0.9382 | 0.8692 | 0.8692 |
| `loss_strong_w3_s6_p2_g1` | 1.0 | 3.0 | 6.0 | 2.0 | 0.9423 | 0.8648 | 0.8648 |
| `loss_strong_w3_s6_p1_g2` | 2.0 | 3.0 | 6.0 | 1.0 | 0.9388 | 0.8681 | 0.8681 |

重點觀察：
- `FEATURE_LOSS_WEIGHTS` 只要最後形成相同 target distribution，絕對 scale 幾乎不影響結果；`3,2,1,0` 與 `30,20,10,0` 完全相同。
- 在 perfect ranking 前提下，把 noise 權重再壓低到 `0.5` 反而讓 OOD 下降，表示目前 synthetic 上保留少量 noise mass 比過度集中在 causal 更穩。
- `REG_SCALE = 0.5` 是這輪最佳設定；`1.0` 明顯較差，`2.0` 與 `4.0` 接近。
- `LOSS_KWARGS` 的影響已比 imperfect-ranking 情況小很多；一旦 ranking 正確，強化 `weight_scale` / `suppress_scale` 只有小幅變動，沒有再帶來先前那種大幅提升。
- 代表性 plot 數值顯示，各 `REG_SCALE` 設定都已把 `spurious_main_grad_prob` 壓到約 `0.004` 到 `0.008`；差異主要來自 causal 與 noise 之間的分配，而不是 spurious 被重新撿回來。

#### 6. Term Ablation (`grad_scale`, `weight_scale`, `suppress_scale`)

固定使用：
- ranking target weights: `causal_main = 3`, `causal_aux = 2`, `noise = 1`, `spurious_main = 0`
- `REG_SCALE = 0.5`
- `target_power = 1.0`
- fixed model seeds: `[9803, 38224, 8113, 4854, 98825]`

單獨開啟單一 term：

| Name | **ID** | **OOD MEAN** | **OOD STD** |
| ---- | ------ | ------------ | ----------- |
| `g0_w0_s0` | 0.9693 | 0.6828 | 0.0221 |
| `g1_w0_s0` | 0.9698 | 0.7108 | 0.0116 |
| `g2_w0_s0` | 0.9698 | 0.7200 | 0.0101 |
| `g0_w1_s0` | 0.9682 | 0.7147 | 0.0257 |
| `g0_w3_s0` | 0.9660 | 0.7331 | 0.0638 |
| `g0_w0_s2` | 0.9645 | 0.8001 | 0.0343 |
| `g0_w0_s6` | 0.9523 | 0.8493 | 0.0306 |

主效應：對其他兩個 term 平均後的 OOD。

| Term | Level | **OOD MEAN** |
| ---- | ----- | ------------ |
| `grad_scale` | `0` | 0.7870 |
| `grad_scale` | `1` | 0.7944 |
| `grad_scale` | `2` | 0.7988 |
| `weight_scale` | `0` | 0.7739 |
| `weight_scale` | `1` | 0.7965 |
| `weight_scale` | `3` | 0.8099 |
| `suppress_scale` | `0` | 0.7368 |
| `suppress_scale` | `2` | 0.7916 |
| `suppress_scale` | `6` | 0.8519 |

最佳幾組：

| Name | `grad_scale` | `weight_scale` | `suppress_scale` | **ID** | **OOD MEAN** | **OOD STD** |
| ---- | ------------ | -------------- | ---------------- | ------ | ------------ | ----------- |
| `g2_w3_s6` | 2.0 | 3.0 | 6.0 | 0.9473 | 0.8758 | 0.0253 |
| `g1_w3_s6` | 1.0 | 3.0 | 6.0 | 0.9480 | 0.8751 | 0.0274 |
| `g2_w1_s6` | 2.0 | 1.0 | 6.0 | 0.9494 | 0.8610 | 0.0186 |
| `g0_w3_s6` | 0.0 | 3.0 | 6.0 | 0.9472 | 0.8579 | 0.0435 |
| `g0_w0_s6` | 0.0 | 0.0 | 6.0 | 0.9523 | 0.8493 | 0.0306 |

重點觀察：
- 三個 term 的重要性排序很明確：`suppress_scale` 最大、`weight_scale` 次之、`grad_scale` 最小。
- 單獨使用時，`suppress-only` 已能把 OOD 從 `0.6828` 拉到 `0.8493`；`weight-only` 最多到 `0.7331`；`grad-only` 最多到 `0.7200`。
- 主效應也一致：`suppress_scale` 從 `0 -> 6` 的平均增益約 `+0.1151`，`weight_scale` 從 `0 -> 3` 約 `+0.0360`，`grad_scale` 從 `0 -> 2` 約 `+0.0118`。
- `grad_scale` 單獨幫助很小，但在 `weight_scale = 3`、`suppress_scale = 6` 的 regime 仍有小幅正收益：`0.8579 -> 0.8751 -> 0.8758`。
- 代表性 plot 數值顯示：
  - `suppress_scale` 打開後，`spurious_main_grad_prob` 會從 `0.2570` 直接掉到約 `0.006`
  - `weight_scale` 進一步把 causal mass 往 `causal_main` / `causal_aux` 集中
  - `grad_scale` 主要是在 causal 與 noise 間再做細調，對壓低 `spurious_main` 的直接作用相對有限



## Appendix

### Environment Setup
```bash
conda create -n ood python=3.10.19
conda activate ood
pip install -r requirements.txt
```

### Run Experiments
- ACSIncome baseline / main experiment
```bash
python run_acsincome.py
```

- Synthetic OOD sanity check
```bash
python run_synthetic_ood.py
```

- ACSIncome GPT coarse-grouping helper
```bash
python run_acsincome_gpt_groups.py --configs gpt_3group_321 --repeat 5 --device cuda:0 --skip-analysis
```

- Check distributional shift & Causal features: modify `main.py`.
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

- Check $1.0$ or $-1.0$ weight for each feature's l2-norm: modify `main.py`.
```python
for feat in FEATURE_LOSS_WEIGHTS.keys():
    FEATURE_LOSS_WEIGHTS[feat] = 1.0
    # main experiment
    ID_base, OOD_MEAN_base, OOD_WORST_base = main(
        DATASET, TRAIN_VAL_STATE, TEST_STATES, FEATURE_INDEX, REMOVED_FEATURE_INDICES, FEATURE_LOSS_WEIGHTS,
        TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
        PATIENCE=PATIENCE, REPEAT=REPEAT, device=device,
    )
    print(f"### Results:\n- ID: {ID_base:.4f}\n- OOD MEAN: {OOD_MEAN_base:.4f}\n- OOD WORST: {OOD_WORST_base:.4f}")

    FEATURE_LOSS_WEIGHTS[feat] = -1.0
    # main experiment
    ID_base, OOD_MEAN_base, OOD_WORST_base = main(
        DATASET, TRAIN_VAL_STATE, TEST_STATES, FEATURE_INDEX, REMOVED_FEATURE_INDICES, FEATURE_LOSS_WEIGHTS,
        TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
        PATIENCE=PATIENCE, REPEAT=REPEAT, device=device,
    )
    print(f"### Results:\n- ID: {ID_base:.4f}\n- OOD MEAN: {OOD_MEAN_base:.4f}\n- OOD WORST: {OOD_WORST_base:.4f}")

    FEATURE_LOSS_WEIGHTS[feat] = 0.0
```
