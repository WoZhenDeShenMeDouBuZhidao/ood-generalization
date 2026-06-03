import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

from data.acs.config import ACS_DATASET_ORDER, feature_index_for
from src.paths import ranking_artifact_dir
from src.utils import call_llm


MODEL_NAME = "gpt-5.5"
DEFAULT_SCORE_SAMPLES = 5


def load_acs_task_metadata() -> Dict[str, Dict[int, str]]:
    return {dataset: feature_index_for(dataset) for dataset in ACS_DATASET_ORDER}


def feature_names(feature_index: Dict[int, str]) -> List[str]:
    return [feature_index[idx] for idx in sorted(feature_index)]


def base_task_description(dataset_name: str) -> str:
    return f"""
You are helping design a feature-importance prior for an out-of-distribution
generalization experiment on the Folktables ACS task `{dataset_name}`.

Research and use your knowledge of Folktables, ACS PUMS variables, and the task
definition. Prefer features that should remain semantically predictive across US
states, and penalize features that likely encode geography, demographic shortcuts,
or brittle state-specific distributional artifacts.
""".strip()


def build_rank_prompt(dataset_name: str, feature_index: Dict[int, str]) -> str:
    features = feature_names(feature_index)
    return f"""
{base_task_description(dataset_name)}

Feature list, in model input order:
{json.dumps(features, indent=2)}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "ranking": [
    {{
      "rank": 1,
      "feature": "FEATURE_NAME",
      "rationale": "short reason"
    }}
  ]
}}

Rules:
- Include every feature exactly once in `ranking`.
- Use ranks 1 through {len(features)}, with no duplicates.
- Rank 1 is most important.
- If two features seem similarly important, still break ties.
""".strip()


def build_score_prompt(dataset_name: str, feature_name: str) -> str:
    return f"""
{base_task_description(dataset_name)}

Score the feature `{feature_name}` by its stable predictive usefulness for this
task, considered marginally and without seeing the downstream data.

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "feature": "{feature_name}",
  "score": 0.0,
  "rationale": "short reason"
}}

Rules:
- `score` must be a number from 0.0 to 1.0.
- Higher score means stronger expected stable predictive usefulness.
- Penalize sensitive demographic shortcuts and state/geography artifacts.
""".strip()


def build_score_all_prompt(dataset_name: str, feature_index: Dict[int, str]) -> str:
    features = feature_names(feature_index)
    return f"""
{base_task_description(dataset_name)}

Score every feature by stable predictive usefulness for this task, considered
marginally and without seeing the downstream data.

Feature list, in model input order:
{json.dumps(features, indent=2)}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "task_summary": "one concise sentence about the prediction target",
  "scores": [
    {{
      "feature": "FEATURE_NAME",
      "score": 0.0,
      "rationale": "short reason"
    }}
  ]
}}

Rules:
- Include every feature exactly once in `scores`.
- `score` must be a number from 0.0 to 1.0.
- Higher score means stronger expected stable predictive usefulness.
- Penalize sensitive demographic shortcuts and state/geography artifacts.
- If two features seem similarly important, assign scores that still reflect your
  best estimate of their relative stable usefulness.
""".strip()


def build_seq_prompt(dataset_name: str, remaining_features: List[str], selected_features: List[str]) -> str:
    selected_text = json.dumps(selected_features, indent=2)
    remaining_text = json.dumps(remaining_features, indent=2)
    return f"""
{base_task_description(dataset_name)}

Sequentially select the next feature that would most improve a downstream model,
given the features already selected.

Already selected features, in order:
{selected_text}

Candidate features not yet selected:
{remaining_text}

Return only valid JSON. Do not wrap it in Markdown.
Use exactly this schema:
{{
  "selected_feature": "FEATURE_NAME",
  "rationale": "short reason"
}}

Rules:
- `selected_feature` must be exactly one feature from the candidate list.
- Choose the feature with the largest marginal stable predictive value given the
  already selected features.
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(match.group(0))


def token_usage_from_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_tokens = sum(call.get("input_tokens") or 0 for call in calls)
    output_tokens = sum(call.get("output_tokens") or 0 for call in calls)
    total_tokens = sum(call.get("total_tokens") or 0 for call in calls)
    return {
        "num_calls": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_llm_with_retry(
    model_name: str,
    message: List[Dict[str, str]],
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return call_llm(model_name, message, temperature=temperature)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            print(
                f"LLM call failed ({exc}); retrying in {retry_sleep:.1f}s "
                f"({attempt + 1}/{max_retries})"
            )
            time.sleep(retry_sleep)
    raise last_error


def validate_ranking(payload: Dict[str, Any], feature_index: Dict[int, str]) -> None:
    expected_features = feature_names(feature_index)
    expected_feature_set = set(expected_features)
    ranking = payload.get("ranking")

    if not isinstance(ranking, list):
        raise ValueError("`ranking` must be a list.")

    ranked_features = [item.get("feature") for item in ranking if isinstance(item, dict)]
    if set(ranked_features) != expected_feature_set or len(ranked_features) != len(expected_features):
        raise ValueError(
            f"`ranking` must contain each feature exactly once. "
            f"Expected {expected_features}, got {ranked_features}."
        )

    ranks = [int(item["rank"]) for item in ranking]
    expected_ranks = list(range(1, len(expected_features) + 1))
    if sorted(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be unique integers {expected_ranks}; got {ranks}.")


def validate_score_payload(payload: Dict[str, Any], feature_name: str) -> float:
    if payload.get("feature") != feature_name:
        raise ValueError(f"Expected score for {feature_name}, got {payload.get('feature')}.")
    score = float(payload["score"])
    if score < 0.0 or score > 1.0:
        raise ValueError(f"Score for {feature_name} must be in [0, 1], got {score}.")
    return score


def validate_score_all_payload(payload: Dict[str, Any], feature_index: Dict[int, str]) -> Dict[str, float]:
    expected_features = feature_names(feature_index)
    expected_feature_set = set(expected_features)
    scores = payload.get("scores")

    if not isinstance(scores, list):
        raise ValueError("`scores` must be a list.")

    scored_features = [item.get("feature") for item in scores if isinstance(item, dict)]
    if set(scored_features) != expected_feature_set or len(scored_features) != len(expected_features):
        raise ValueError(
            f"`scores` must contain each feature exactly once. "
            f"Expected {expected_features}, got {scored_features}."
        )

    score_by_feature = {}
    for item in scores:
        feature_name = item["feature"]
        score = float(item["score"])
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score for {feature_name} must be in [0, 1], got {score}.")
        score_by_feature[feature_name] = score
    return score_by_feature


def validate_seq_payload(payload: Dict[str, Any], remaining_features: List[str]) -> str:
    selected_feature = payload.get("selected_feature")
    if selected_feature not in remaining_features:
        raise ValueError(f"Expected one of {remaining_features}, got {selected_feature}.")
    return selected_feature


def output_path(dataset_name: str, model_name: str, method: str) -> Path:
    safe_model_name = model_name.replace("/", "_")
    return ranking_artifact_dir("acs", dataset_name) / f"{safe_model_name}_{method}_feature_ranking.json"


def select_datasets(requested_dataset: str, metadata: Dict[str, Dict[int, str]]) -> List[str]:
    if requested_dataset == "all":
        return list(metadata)
    if requested_dataset not in metadata:
        valid = ", ".join(["all"] + sorted(metadata))
        raise ValueError(f"Unsupported dataset: {requested_dataset}. Valid choices: {valid}")
    return [requested_dataset]


def select_methods(requested_method: str) -> List[str]:
    if requested_method == "all":
        return ["rank", "score", "score_all", "seq"]
    return [requested_method]


def write_result(
    dataset_name: str,
    model_name: str,
    method: str,
    temperature: float,
    response: Dict[str, Any],
    llm_calls: List[Dict[str, Any]],
    extra_config: Dict[str, Any] | None = None,
) -> Path:
    result = {
        "model": model_name,
        "method": method,
        "temperature": temperature,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "token_usage": token_usage_from_calls(llm_calls),
        "response": response,
        "llm_calls": llm_calls,
    }
    if extra_config:
        result["config"] = extra_config

    path = output_path(dataset_name, model_name, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def generate_rank(
    dataset_name: str,
    feature_index: Dict[int, str],
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> Path:
    prompt = build_rank_prompt(dataset_name, feature_index)
    llm_result = call_llm_with_retry(
        model_name,
        [{"role": "user", "content": prompt}],
        temperature,
        max_retries,
        retry_sleep,
    )
    payload = extract_json_object(llm_result["content"])
    validate_ranking(payload, feature_index)
    llm_call = {"prompt": prompt, **llm_result}
    return write_result(dataset_name, model_name, "rank", temperature, payload, [llm_call])


def generate_score(
    dataset_name: str,
    feature_index: Dict[int, str],
    model_name: str,
    temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
) -> Path:
    calls = []
    per_feature = []
    for feature_name in feature_names(feature_index):
        sample_payloads = []
        sample_scores = []
        for sample_idx in range(score_samples):
            prompt = build_score_prompt(dataset_name, feature_name)
            llm_result = call_llm_with_retry(
                model_name,
                [{"role": "user", "content": prompt}],
                temperature,
                max_retries,
                retry_sleep,
            )
            payload = extract_json_object(llm_result["content"])
            score = validate_score_payload(payload, feature_name)
            sample_payloads.append(payload)
            sample_scores.append(score)
            calls.append({
                "feature": feature_name,
                "sample": sample_idx + 1,
                "prompt": prompt,
                **llm_result,
            })
            if call_delay > 0:
                time.sleep(call_delay)
        per_feature.append({
            "feature": feature_name,
            "score_mean": mean(sample_scores),
            "score_std": pstdev(sample_scores) if len(sample_scores) > 1 else 0.0,
            "samples": sample_payloads,
        })

    ordered = sorted(per_feature, key=lambda item: (-item["score_mean"], item["feature"]))
    response = {
        "ranking": [
            {
                "rank": rank,
                "feature": item["feature"],
                "score_mean": item["score_mean"],
                "score_std": item["score_std"],
            }
            for rank, item in enumerate(ordered, start=1)
        ],
        "scores": per_feature,
    }
    validate_ranking(response, feature_index)
    return write_result(
        dataset_name,
        model_name,
        "score",
        temperature,
        response,
        calls,
        extra_config={"score_samples": score_samples},
    )


def generate_score_all(
    dataset_name: str,
    feature_index: Dict[int, str],
    model_name: str,
    temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
) -> Path:
    calls = []
    samples_by_feature = {feature_name: [] for feature_name in feature_names(feature_index)}
    sample_payloads = []

    for sample_idx in range(score_samples):
        prompt = build_score_all_prompt(dataset_name, feature_index)
        llm_result = call_llm_with_retry(
            model_name,
            [{"role": "user", "content": prompt}],
            temperature,
            max_retries,
            retry_sleep,
        )
        payload = extract_json_object(llm_result["content"])
        sample_scores = validate_score_all_payload(payload, feature_index)
        sample_payloads.append(payload)
        for feature_name, score in sample_scores.items():
            samples_by_feature[feature_name].append(score)
        calls.append({
            "sample": sample_idx + 1,
            "prompt": prompt,
            **llm_result,
        })
        if call_delay > 0:
            time.sleep(call_delay)

    per_feature = []
    for feature_name, sample_scores in samples_by_feature.items():
        per_feature.append({
            "feature": feature_name,
            "score_mean": mean(sample_scores),
            "score_std": pstdev(sample_scores) if len(sample_scores) > 1 else 0.0,
            "sample_scores": sample_scores,
        })

    ordered = sorted(per_feature, key=lambda item: (-item["score_mean"], item["feature"]))
    response = {
        "ranking": [
            {
                "rank": rank,
                "feature": item["feature"],
                "score_mean": item["score_mean"],
                "score_std": item["score_std"],
            }
            for rank, item in enumerate(ordered, start=1)
        ],
        "scores": per_feature,
        "samples": sample_payloads,
    }
    validate_ranking(response, feature_index)
    return write_result(
        dataset_name,
        model_name,
        "score_all",
        temperature,
        response,
        calls,
        extra_config={"score_samples": score_samples},
    )


def generate_seq(
    dataset_name: str,
    feature_index: Dict[int, str],
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> Path:
    calls = []
    selected_features = []
    remaining_features = feature_names(feature_index)
    steps = []

    while remaining_features:
        prompt = build_seq_prompt(dataset_name, remaining_features, selected_features)
        llm_result = call_llm_with_retry(
            model_name,
            [{"role": "user", "content": prompt}],
            temperature,
            max_retries,
            retry_sleep,
        )
        payload = extract_json_object(llm_result["content"])
        selected_feature = validate_seq_payload(payload, remaining_features)
        selected_features.append(selected_feature)
        remaining_features.remove(selected_feature)
        steps.append({
            "rank": len(selected_features),
            "feature": selected_feature,
            "rationale": payload.get("rationale", ""),
        })
        calls.append({
            "step": len(selected_features),
            "prompt": prompt,
            **llm_result,
        })

    response = {"ranking": steps}
    validate_ranking(response, feature_index)
    return write_result(dataset_name, model_name, "seq", temperature, response, calls)


def generate_method(
    method: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    model_name: str,
    rank_temperature: float,
    score_temperature: float,
    seq_temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
    skip_existing: bool,
) -> Path:
    path = output_path(dataset_name, model_name, method)
    if skip_existing and path.exists():
        return path
    if method == "rank":
        return generate_rank(dataset_name, feature_index, model_name, rank_temperature, max_retries, retry_sleep)
    if method == "score":
        return generate_score(
            dataset_name,
            feature_index,
            model_name,
            score_temperature,
            score_samples,
            call_delay,
            max_retries,
            retry_sleep,
        )
    if method == "score_all":
        return generate_score_all(
            dataset_name,
            feature_index,
            model_name,
            score_temperature,
            score_samples,
            call_delay,
            max_retries,
            retry_sleep,
        )
    if method == "seq":
        return generate_seq(dataset_name, feature_index, model_name, seq_temperature, max_retries, retry_sleep)
    raise ValueError(f"Unsupported method: {method}")


def print_dry_run(
    selected_datasets: Iterable[str],
    selected_methods: Iterable[str],
    metadata: Dict[str, Dict[int, str]],
) -> None:
    for dataset_name in selected_datasets:
        feature_index = metadata[dataset_name]
        for method in selected_methods:
            print(f"### {dataset_name} / {method}")
            if method == "rank":
                print(build_rank_prompt(dataset_name, feature_index))
            elif method == "score":
                print(build_score_prompt(dataset_name, feature_names(feature_index)[0]))
            elif method == "score_all":
                print(build_score_all_prompt(dataset_name, feature_index))
            elif method == "seq":
                print(build_seq_prompt(dataset_name, feature_names(feature_index), []))
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM feature rankings for ACS tasks.")
    parser.add_argument("--dataset", default="acsincome", help="ACS dataset key to rank, or `all`.")
    parser.add_argument("--method", default="all", choices=["all", "rank", "score", "score_all", "seq"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--rank-temperature", type=float, default=0.0)
    parser.add_argument("--score-temperature", type=float, default=0.5)
    parser.add_argument("--seq-temperature", type=float, default=0.0)
    parser.add_argument("--score-samples", type=int, default=DEFAULT_SCORE_SAMPLES)
    parser.add_argument("--call-delay", type=float, default=1.0, help="Seconds to sleep between score calls.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--skip-existing", action="store_true", help="Do not regenerate existing method JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the LLM.")
    args = parser.parse_args()

    load_dotenv()
    metadata = load_acs_task_metadata()
    selected_datasets = select_datasets(args.dataset, metadata)
    selected_methods = select_methods(args.method)

    if args.dry_run:
        print_dry_run(selected_datasets, selected_methods, metadata)
        return

    for dataset_name in selected_datasets:
        feature_index = metadata[dataset_name]
        for method in selected_methods:
            path = generate_method(
                method,
                dataset_name,
                feature_index,
                args.model,
                args.rank_temperature,
                args.score_temperature,
                args.seq_temperature,
                args.score_samples,
                args.call_delay,
                args.max_retries,
                args.retry_sleep,
                args.skip_existing,
            )
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
