import json
from typing import Any, Dict, List


def base_task_description(benchmark: str, dataset_name: str) -> str:
    if benchmark == "acs":
        return f"""
You are helping design a feature-importance prior for an out-of-distribution
generalization experiment on the Folktables ACS task `{dataset_name}`.

Research and use your knowledge of Folktables, ACS PUMS variables, and the task
definition. Prefer features that should remain semantically predictive across US
states, and penalize features that likely encode geography, demographic shortcuts,
or brittle state-specific distributional artifacts.
""".strip()

    return f"""
You are helping design a feature-importance prior for an out-of-distribution
generalization experiment on the `{benchmark}` benchmark dataset `{dataset_name}`.

Research and use your knowledge of the benchmark and task definition. Prefer
features that should remain semantically predictive across train/test domains,
and penalize features that likely encode domain identifiers, leakage,
demographic shortcuts, or brittle distributional artifacts.
""".strip()


def build_rank_prompt(benchmark: str, dataset_name: str, feature_cards: List[Dict[str, Any]]) -> str:
    feature_ids = [int(card["feature_id"]) for card in feature_cards]
    item_schema = (
        '{\n      "rank": 1,\n      "feature_id": 0\n    }'
        if len(feature_cards) > 150
        else '{\n      "rank": 1,\n      "feature_id": 0,\n      "rationale": "short reason"\n    }'
    )
    return f"""
{base_task_description(benchmark, dataset_name)}

Feature cards, in model input order:
{json.dumps(feature_cards, indent=2, ensure_ascii=False)}

Feature IDs to include exactly once:
{json.dumps(feature_ids)}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "ranking": [
    {item_schema}
  ]
}}

Rules:
- Include every feature_id exactly once in `ranking`.
- Use ranks 1 through {len(feature_cards)}, with no duplicates.
- Rank 1 is most important.
- If two features seem similarly important, still break ties.
- `feature_id` is the authoritative identifier. Include each feature_id exactly
  once and do not include feature names in the output.
- For large feature sets, omit rationales and return only rank/feature_id pairs.
""".strip()


def build_score_prompt(benchmark: str, dataset_name: str, feature_card: Dict[str, Any]) -> str:
    return f"""
{base_task_description(benchmark, dataset_name)}

Score this feature by its stable predictive usefulness for this
task, considered marginally and without seeing the downstream data.

Feature card:
{json.dumps(feature_card, indent=2, ensure_ascii=False)}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "feature_id": {int(feature_card["feature_id"])},
  "score": 0.0,
  "rationale": "short reason"
}}

Rules:
- `score` must be a number from 0.0 to 1.0.
- Higher score means stronger expected stable predictive usefulness.
- Penalize sensitive or domain-specific shortcuts, leakage, and brittle artifacts.
- `feature_id` is the authoritative identifier. Do not include feature names in
  the output.
""".strip()


def build_score_all_prompt(benchmark: str, dataset_name: str, feature_cards: List[Dict[str, Any]]) -> str:
    feature_ids = [int(card["feature_id"]) for card in feature_cards]
    item_schema = (
        '{\n      "feature_id": 0,\n      "score": 0.0\n    }'
        if len(feature_cards) > 150
        else '{\n      "feature_id": 0,\n      "score": 0.0,\n      "rationale": "short reason"\n    }'
    )
    return f"""
{base_task_description(benchmark, dataset_name)}

Score every feature by stable predictive usefulness for this task, considered
marginally and without seeing the downstream data.

Feature cards, in model input order:
{json.dumps(feature_cards, indent=2, ensure_ascii=False)}

Feature IDs to include exactly once:
{json.dumps(feature_ids)}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "scores": [
    {item_schema}
  ]
}}

Rules:
- Include every feature_id exactly once in `scores`.
- `score` must be a number from 0.0 to 1.0.
- Higher score means stronger expected stable predictive usefulness.
- Penalize sensitive or domain-specific shortcuts, leakage, and brittle artifacts.
- If two features seem similarly important, assign scores that still reflect your
  best estimate of their relative stable usefulness.
- `feature_id` is the authoritative identifier. Include each feature_id exactly
  once and do not include feature names in the output.
- For large feature sets, omit rationales and return only feature_id/score
  pairs.
""".strip()


def build_seq_prompt(
    benchmark: str,
    dataset_name: str,
    remaining_features: List[Dict[str, Any]],
    selected_features: List[Dict[str, Any]],
) -> str:
    selected_text = json.dumps(selected_features, indent=2, ensure_ascii=False)
    remaining_text = json.dumps(remaining_features, indent=2, ensure_ascii=False)
    allowed_ids = [int(card["feature_id"]) for card in remaining_features]
    allowed_ids_text = json.dumps(allowed_ids)
    return f"""
{base_task_description(benchmark, dataset_name)}

Sequentially select the next feature that would most improve a downstream model,
given the features already selected.

Already selected features, in order:
{selected_text}

Candidate feature cards not yet selected:
{remaining_text}

Allowed selected_feature_id values for this step:
{allowed_ids_text}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "selected_feature_id": 0,
  "rationale": "short reason"
}}

Rules:
- `selected_feature_id` must be exactly one feature_id from the candidate list.
- Never choose a feature_id from the already selected list.
- If an ID is not listed in the allowed selected_feature_id values above, it is
  invalid for this step even if it appears in the already selected list.
- Choose the feature with the largest marginal stable predictive value given the
  already selected features.
- `selected_feature_id` is the authoritative identifier. Do not include feature
  names in the output.
""".strip()
