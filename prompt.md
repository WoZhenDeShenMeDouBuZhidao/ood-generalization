# Feature Ranking Prompt Template

你是一位負責做 OOD generalization feature-prior 分析的研究助理。

我會提供：

- dataset 名稱
- feature 名稱列表

你的任務是：

1. 針對這個 dataset 的預測目標與常見資料收集方式，判斷各 feature 對目標的「穩定可遷移重要性」排序
2. 額外判斷從哪一個 ranking 名次開始，後面的 feature 更可能只是 training set 內的 spurious correlation、proxy、sampling artifact、地理/人口子群分布偏差、或其他不穩定訊號

重要要求：

- 你必須自己使用網頁搜尋與深度思考補足資料集背景、任務定義、常見資料偏差來源、各 feature 的語意
- 不要要求我提供 dataset metadata
- 不要回覆「需要更多資料」；請直接根據你查到的資訊與合理推論完成任務
- 你要優先考慮 OOD generalization，而不是單純在 IID / training distribution 上最有預測力的 feature
- 排名時請更重視：
  - 機制上較直接、跨環境較穩定的 feature
  - 不容易只反映群體差異、標註流程、地區分布、年份分布、資料收集流程的 feature
- 排名時請降低：
  - 明顯敏感屬性
  - 容易只是 proxy 的 feature
  - 可能強烈依賴特定 training distribution 的 feature
- 如果你認為前段與後段相對明確、中段不確定，請明確說出來

請分析以下資料：

- Dataset name: `{DATASET_NAME}`
- Feature list:
  - `{FEATURE_1}`
  - `{FEATURE_2}`
  - `{FEATURE_3}`
  - `{FEATURE_4}`
  - `{FEATURE_5}`
  - `{FEATURE_6}`
  - `{FEATURE_7}`
  - `{FEATURE_8}`
  - `{FEATURE_9}`
  - `{FEATURE_10}`

請嚴格依照以下格式輸出，不要加其他段落。

## Output Format

### A. Ranking Summary

- `ranking`: 由高到低排列的 feature list
- `spurious_start_rank`: 一個整數 `k`
  - 代表你認為從第 `k` 名之後，feature 更可能開始落入 training-specific / spurious / proxy-heavy 區段
  - 若你認為前 `k-1` 名相對較穩定，從第 `k` 名開始不穩定性明顯上升，請填這個 `k`
- `high_confidence_front_features`: 你最有把握應該放在前段的 feature
- `high_risk_spurious_features`: 你最懷疑可能偏 spurious / proxy / unstable 的 feature

### B. Per-Feature Table

請對每個 feature 輸出一列，欄位固定為：

- `feature`
- `rank`
- `importance_score`
  - 用 `0` 到 `10`
  - 代表對 OOD generalization 有幫助的穩定重要性，不是單純訓練集預測力
- `spurious_risk_score`
  - 用 `0` 到 `10`
  - 分數越高代表越可能是 training-specific / proxy / unstable correlation
- `short_reason`
  - 一句話，簡述為什麼排在這裡

### C. Grouping Recommendation

請再額外給一個適合用來分配 `FEATURE_LOSS_WEIGHTS` 的粗分組建議：

- `recommended_grouping`
  - 只能用以下其中一種：
    - `2-group`
    - `3-group`
    - `4-group`
- `group_definition`
  - 說明每一組包含哪些 rank 區間，例如：
    - `group_1 = ranks 1-3`
    - `group_2 = ranks 4-7`
    - `group_3 = ranks 8-10`
- `group_rationale`
  - 簡述為什麼這樣切比逐名次細排更穩定

### D. Final JSON

最後請輸出一個可直接程式讀取的 JSON block，格式必須為：

```json
{
  "dataset_name": "{DATASET_NAME}",
  "ranking": ["..."],
  "spurious_start_rank": 0,
  "high_confidence_front_features": ["..."],
  "high_risk_spurious_features": ["..."],
  "per_feature": [
    {
      "feature": "...",
      "rank": 1,
      "importance_score": 0,
      "spurious_risk_score": 0,
      "short_reason": "..."
    }
  ],
  "recommended_grouping": "3-group",
  "group_definition": {
    "group_1": ["..."],
    "group_2": ["..."],
    "group_3": ["..."]
  }
}
```

額外限制：

- `ranking` 必須包含全部 feature，且不得重複
- `rank` 必須從 `1` 開始連續編號
- `spurious_start_rank` 必須落在 `1` 到 `feature 數量 + 1` 之間
  - 如果你認為沒有明顯 spurious 區段，請輸出 `feature 數量 + 1`
- `group_definition` 內必須覆蓋全部 feature，且不得重複
