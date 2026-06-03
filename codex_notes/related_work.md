# Related Work Survey Notes

這份檔案用來記錄已讀過、和本研究相關的 papers。

每篇 paper 至少保留：

- 論文核心方法與主張
- 與本研究的相同點
- 本研究相對可主張的 novelty
- 對後續實驗或寫作 positioning 的啟發

原始 PDF 放在 `codex_notes/surveyed_papers/`。

## Large Language Models as Attribution Regularizers for Efficient Model Training

Paper:

- `codex_notes/surveyed_papers/Large Language Models as Attribution Regularizers for Efficient Model Training - 2502.20268v3.pdf`
- arXiv: `2502.20268v3`
- Method name: LAAT, Large Language Model Attribution Aligned Training

### Core Idea

LAAT 使用 LLM 產生 global feature attribution scores，並把這些 scores 當成 smaller downstream model 的 training regularizer。

流程：

1. 給 LLM task description 與 feature descriptions。
2. 讓 LLM 對每個 feature 產生 `-10` 到 `10` 的 importance score。
3. 多次生成後平均，得到 `s_LLM`。
4. 訓練 binary classifier 時，使用 input gradient 作為 local attribution。
5. 把 local attribution 與 `s_LLM` normalization 後，用 MSE 對齊。

形式上接近：

```text
loss = BCE + gamma * MSE(normalized_input_gradient, normalized_LLM_scores)
```

主實驗設定：

- few-shot tabular classification
- biased-data experiments
- downstream model 主要是 logistic regression / MLP
- metric 以 ROC-AUC 為主

paper 的主 claim 是：LLM attribution prior 可以作為 data-efficient learning 的 inductive bias，在 few-shot 與 biased-data setting 下提升 generalization。

### Similarities To Our Work

- 都使用 LLM 或外部 prior 來產生 feature-level importance guidance。
- 都把 feature attribution regularization 放進 downstream model training，而不是讓 LLM 直接做 prediction。
- 都聚焦 tabular data，且保持 downstream model 相對簡單。
- 都使用 input-gradient attribution 作為 model feature usage 的一種 proxy。
- 都關心 biased / spurious correlations 對泛化的影響。
- 都依賴 feature descriptions 能被自然語言理解。

### Key Differences

LAAT 的核心是 attribution matching：

- 將 local input-gradient attribution 對齊 LLM 給的 global attribution vector。
- regularizer 是 MSE-style vector alignment。
- 主要目標是 few-shot efficiency。
- biased-data evaluation 是輔助實驗，bias 多由人工刪資料產生。

本研究目前的 `FeatureImportanceTargetCELoss` 更像 feature-use allocation control：

- 將 positive feature weights 轉成 target feature-importance distribution。
- 對齊 feature-wise input-gradient mass: `grad_target_loss`。
- 同時對齊 first-layer weight mass: `weight_target_loss`。
- 對 non-positive / suppressed features 額外加入 explicit mass penalty: `suppressed_prob_loss`。
- 追蹤 `grad_prob_on_suppressed` 與 `weight_prob_on_suppressed`，直接檢查 spurious feature usage 是否被壓低。
- 主要問題設定是 OOD generalization / environment shift，而不是 few-shot learning 本身。

Implementation reference:

- `src/loss.py`
  - `FeatureImportanceTargetCELoss`
  - `grad_target_loss`
  - `weight_target_loss`
  - `suppressed_prob_loss`
  - `grad_prob_on_suppressed`
  - `weight_prob_on_suppressed`

### Our Potential Novelty

The safest novelty positioning should not be "LLM attribution regularization" alone, because LAAT already claims that space.

Better positioning:

> Existing LLM-attribution regularization aligns model explanations with LLM-derived scores. Our work studies how LLM or domain feature priors can be converted into explicit feature-use constraints for OOD generalization, especially by suppressing non-invariant or spurious features.

Concrete novelty angles:

1. **From attribution matching to spurious feature suppression**

   LAAT matches local attribution to LLM scores. Our loss explicitly penalizes attribution and weight mass assigned to suppressed features. This gives a more direct mechanism for reducing reliance on non-transferable features.

2. **Dual regularization of sensitivity and parameter allocation**

   LAAT uses input gradients. Our full loss regularizes both:

   - input-gradient feature mass
   - first-layer weight feature mass

   Synthetic ablations suggest `weight_scale` and especially `suppress_scale` are important for OOD robustness.

3. **OOD / environment-shift objective**

   LAAT mainly evaluates few-shot ROC-AUC. Our ACSIncome setup evaluates state-level distribution shift and reports OOD mean / OOD worst, which targets robustness under environment-specific correlations.

4. **Ranking / grouping priors instead of calibrated numeric scores**

   LAAT asks LLMs for calibrated integer scores. Our setup can use rankings, coarse groups, or suppress sets, then convert them into a target distribution via `target_power` and `suppress_scale`. This is useful because LLM numeric scores may be poorly calibrated.

5. **LLM prior quality as a bottleneck**

   Current ACSIncome experiments suggest the loss can help when ranking is close to the transferable structure, but GPT ranking / grouping quality can be the bottleneck. This is more specific than LAAT's score-noise robustness analysis.

6. **Failure-mode diagnostics for attribution regularization**

   The ACSIncome standardization diagnosis showed that raw tabular feature scale can make gradient regularization misleading. Without standardization, large-code features such as `OCCP` can have tiny raw-unit gradients while still dominating SHAP. This is a useful methodological warning for attribution-regularized tabular learning.

### Recommended Experiments To Distinguish From LAAT

1. Add a LAAT-style baseline in `src/loss.py`.

   Minimal baseline:

   ```text
   BCE + gamma * MSE(normalized_grad_l2, normalized_target_scores)
   ```

   No first-layer weight term. No suppressed-mass penalty.

2. Compare these variants on synthetic OOD and ACSIncome:

   | method | grad target | weight target | suppressed mass |
   |---|---:|---:|---:|
   | CE | no | no | no |
   | LAAT-style MSE | yes | no | no |
   | ours grad-only target CE | yes | no | optional |
   | ours grad + weight | yes | yes | no |
   | ours full | yes | yes | yes |

3. Report mechanism metrics alongside accuracy:

   - `grad_prob_on_suppressed`
   - `weight_prob_on_suppressed`
   - OOD mean
   - OOD worst

4. Compare prior formats:

   - raw LLM numeric scores
   - LLM ranking
   - LLM coarse groups
   - LLM suppress set

5. Emphasize worst-environment accuracy.

   OOD worst is a stronger differentiator than mean ROC-AUC because it directly tests robustness to environment-specific shortcuts.

### Writing Positioning

Avoid titles or abstracts that sound like:

- "LLMs as attribution regularizers"
- "LLM-generated feature attribution for tabular learning"

Better candidate positioning:

- "LLM-Guided Spurious Feature Suppression for OOD Tabular Generalization"
- "Targeted Feature-Use Regularization for Out-of-Distribution Tabular Learning"
- "From LLM Feature Priors to Robust Feature Use: Suppressing Non-Invariant Signals in Tabular Models"

Possible contribution statement:

> We differ from prior LLM attribution-alignment methods by treating feature priors as constraints on transferable feature usage rather than calibrated attribution targets. Our objective combines gradient-distribution alignment, first-layer weight-distribution alignment, and explicit suppressed-feature mass penalties, and we evaluate these mechanisms under environment-level OOD shifts.

## LLM-Select: Feature Selection with Large Language Models

Paper:

- `codex_notes/surveyed_papers/LLM-Select: Feature Selection with Large Language Models.pdf`
- arXiv: `2407.02694v2`
- Published in TMLR, 04/2025
- Method name: LLM-Select

### Core Idea

LLM-Select studies whether LLMs can perform feature selection using only feature names
and a task description, without seeing downstream training data. The paper proposes three
query mechanisms:

1. **LLM-Score**

   Query one feature at a time and ask the LLM for a numerical importance score in
   `[0, 1]`. Features are ranked by score. The paper interprets this as marginal feature
   importance because each query only contains one candidate feature and the target
   concept.

2. **LLM-Rank**

   Query the LLM once with the full feature list and ask it to produce a complete ranking.
   The paper interprets ranks as relative importance with respect to the other available
   features.

3. **LLM-Seq**

   Sequentially select features in a dialogue. Starting from an empty set, ask the LLM to
   pick the next feature that would most improve downstream cross-validation performance
   given the already selected features. This is a greedy sequential selection analogue.

Default prompting is zero-shot and uses no dataset-specific context. The paper reports that
greedy decoding (`T=0`) is a strong baseline. For self-consistency, it samples multiple
responses with `T=0.5` and averages scores. On GPT-4, all three mechanisms have similar
feature-selection performance on small-scale datasets; LLM-Score is emphasized for larger
feature spaces because it scales better than full-rank or sequential selection.

### Similarities To Our Work

- Both use LLM feature priors without giving the LLM access to downstream training data.
- Both rely on semantic feature names / concepts and a target/task description.
- Both compare different ways to elicit feature importance from the LLM.
- Both are relevant to tabular prediction and Folktables-style ACS tasks.

### Key Differences

LLM-Select focuses on feature subset selection:

- The LLM output determines which top-k features are used for downstream training.
- Evaluation studies predictive performance as selected feature fraction varies.
- LLM-Score, LLM-Rank, and LLM-Seq are feature selection mechanisms.

Our current project uses LLM outputs as training priors:

- The downstream model still sees the configured feature set.
- The ranking is converted later into `FeatureImportanceTargetCELoss` weights.
- The objective is OOD generalization under state-level environment shift, especially OOD
  worst accuracy.
- We need to study ranking-to-weight conversion and feature suppression, not only top-k
  subset choice.

### Our Potential Novelty

LLM-Select already establishes that LLMs can rank or score features competitively for
feature selection. A safer novelty angle for this project is:

> Instead of using LLM feature priors to discard features, we study how different LLM prior
> elicitation mechanisms can be converted into feature-use regularizers for OOD
> generalization, including explicit suppression of non-transferable feature usage.

Additional differentiators:

- Compare rank/score/seq not only by downstream accuracy, but also by token cost and
  stability of generated priors.
- Evaluate cost/performance trade-offs under ACS state-level OOD shift.
- Study how score/rank/seq outputs should be mapped into continuous loss weights.

### Implementation Implications

- `rank_features.py` should support:
  - `rank`: one full-list ranking query, greedy decoding.
  - `score`: one-feature-at-a-time score queries, multiple samples at higher temperature,
    average scores, then rank.
  - `seq`: sequential greedy feature selection given already selected features.
- Ranking JSON should include input/output token counts for each method so we can compare
  cost with performance later.
- Score sampling should expose `temperature` and number of samples as knobs.

## LLM-Lasso: A Robust Framework for Domain-Informed Feature Selection and Regularization

Paper:

- `codex_notes/surveyed_papers/LLM-Lasso_ A Robust Framework for Domain-Informed Feature Selection and Regularization - 2502.10648v3.pdf`
- arXiv: `2502.10648v3`
- Method name: LLM-Lasso

### Core Idea

LLM-Lasso integrates LLM-derived feature knowledge into a Lasso-style embedded feature selector.
The LLM produces feature-level penalty factors, where lower penalties mean the model should shrink
that feature less. The downstream objective is weighted L1 regularization:

```text
prediction_loss + lambda * sum_j w_j |beta_j|
```

The paper's robustness mechanism is cross-validation over transformations of the LLM penalty
factors. In the inverse-importance family, `eta=0` maps all penalties to the standard Lasso case,
while larger `eta` values rely more strongly on the LLM prior. This lets validation fall back to
plain Lasso when the LLM prior is not useful.

### Implementation Implications

- `run_acs_llm_lasso.py` implements a score-only version because our existing ACS artifacts store
  LLM scores, not direct Lasso penalty factors.
- The runner converts score artifacts into inverse penalty factors and searches `eta`.
- `eta=0` is included so validation can choose the plain L1 logistic baseline.
- The implementation uses train-only `StandardScaler`, then rescales columns by the penalty factors
  before fitting ordinary L1 logistic regression.
