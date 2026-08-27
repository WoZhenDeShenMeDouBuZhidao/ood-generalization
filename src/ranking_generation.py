import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List

from src.ranking import RANKING_METHODS, ranking_artifact_path
from src.ranking_prompts import (
    build_rank_prompt,
    build_score_all_prompt,
    build_score_prompt,
    build_seq_prompt,
)
from src.utils import call_llm


def feature_names(feature_index: Dict[int, str]) -> List[str]:
    return [feature_index[idx] for idx in sorted(feature_index)]


def feature_card_by_id(feature_cards: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(card["feature_id"]): card for card in feature_cards}


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
    use_web_search: bool,
) -> Dict[str, Any]:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return call_llm(
                model_name,
                message,
                temperature=temperature,
                use_web_search=use_web_search,
            )
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


def call_validated_payload_with_retry(
    model_name: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    validate_payload: Callable[[Dict[str, Any]], Any],
):
    last_error = None
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        try:
            llm_result = call_llm_with_retry(
                model_name,
                messages,
                temperature,
                0,
                retry_sleep,
                use_web_search,
            )
            payload = extract_json_object(llm_result["content"])
            validation_result = validate_payload(payload)
            return payload, llm_result, validation_result
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            print(
                f"LLM output validation failed ({exc}); retrying in {retry_sleep:.1f}s "
                f"({attempt + 1}/{max_retries})"
            )
            messages = [
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "The previous response was invalid: "
                        f"{str(exc)[:500]}. Return a new valid JSON object that "
                        "satisfies the original schema and candidate constraints."
                    ),
                },
            ]
            time.sleep(retry_sleep)
    raise last_error


def _canonicalize_feature_item(item: Dict[str, Any], feature_index: Dict[int, str]) -> str:
    if "feature_id" in item:
        feature_id = int(item["feature_id"])
        if feature_id not in feature_index:
            raise ValueError(f"Unknown feature_id: {feature_id}")
        item["feature_id"] = feature_id
        item["feature"] = feature_index[feature_id]
        return item["feature"]
    return item.get("feature")


def validate_ranking(payload: Dict[str, Any], feature_index: Dict[int, str]) -> None:
    expected_features = feature_names(feature_index)
    expected_feature_set = set(expected_features)
    ranking = payload.get("ranking")

    if not isinstance(ranking, list):
        raise ValueError("`ranking` must be a list.")

    ranked_features = [
        _canonicalize_feature_item(item, feature_index)
        for item in ranking
        if isinstance(item, dict)
    ]
    if set(ranked_features) != expected_feature_set or len(ranked_features) != len(expected_features):
        raise ValueError(
            f"`ranking` must contain each feature exactly once. "
            f"Expected {expected_features}, got {ranked_features}."
        )

    ranks = [int(item["rank"]) for item in ranking]
    expected_ranks = list(range(1, len(expected_features) + 1))
    if sorted(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be unique integers {expected_ranks}; got {ranks}.")


def validate_score_payload(payload: Dict[str, Any], feature_id: int, feature_name: str) -> float:
    if "feature_id" in payload:
        payload_feature_id = int(payload["feature_id"])
        if payload_feature_id != int(feature_id):
            raise ValueError(f"Expected feature_id {feature_id}, got {payload_feature_id}.")
        payload["feature_id"] = payload_feature_id
        payload["feature"] = feature_name
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

    scored_features = [
        _canonicalize_feature_item(item, feature_index)
        for item in scores
        if isinstance(item, dict)
    ]
    if set(scored_features) != expected_feature_set or len(scored_features) != len(expected_features):
        raise ValueError(
            f"`scores` must contain each feature exactly once. "
            f"Expected {expected_features}, got {scored_features}."
        )

    score_by_feature = {}
    for item in scores:
        feature_name = _canonicalize_feature_item(item, feature_index)
        score = float(item["score"])
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score for {feature_name} must be in [0, 1], got {score}.")
        score_by_feature[feature_name] = score
    return score_by_feature


def validate_seq_payload(
    payload: Dict[str, Any],
    remaining_feature_ids: List[int],
    feature_index: Dict[int, str],
) -> tuple[int, str]:
    if "selected_feature_id" in payload:
        selected_feature_id = int(payload["selected_feature_id"])
        if selected_feature_id not in remaining_feature_ids:
            raise ValueError(f"selected_feature_id {selected_feature_id} is not in the remaining candidate IDs")
        payload["selected_feature_id"] = selected_feature_id
        payload["selected_feature"] = feature_index[selected_feature_id]
        return selected_feature_id, payload["selected_feature"]

    selected_feature = payload.get("selected_feature")
    remaining_features = [feature_index[idx] for idx in remaining_feature_ids]
    if selected_feature not in remaining_features:
        raise ValueError(f"selected_feature {selected_feature} is not in the remaining candidate features")
    selected_feature_id = {feature_index[idx]: idx for idx in remaining_feature_ids}[selected_feature]
    payload["selected_feature_id"] = selected_feature_id
    return selected_feature_id, selected_feature


def output_path(
    benchmark: str,
    dataset_name: str,
    model_name: str,
    method: str,
    feature_space: str = "semantic",
) -> Path:
    return ranking_artifact_path(
        dataset_name,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )


def prompt_output_path(
    benchmark: str,
    dataset_name: str,
    model_name: str,
    method: str,
    feature_space: str = "semantic",
) -> Path:
    prompt_name = f"{method}_prompt.txt" if feature_space == "semantic" else f"{method}_{feature_space}_prompt.txt"
    return output_path(benchmark, dataset_name, model_name, method, feature_space).with_name(prompt_name)


def write_result(
    benchmark: str,
    dataset_name: str,
    model_name: str,
    method: str,
    temperature: float,
    response: Dict[str, Any],
    llm_calls: List[Dict[str, Any]],
    extra_config: Dict[str, Any] | None = None,
    prompt_sections: List[Dict[str, str]] | None = None,
    feature_space: str = "semantic",
) -> Path:
    path = output_path(benchmark, dataset_name, model_name, method, feature_space)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_output_path(benchmark, dataset_name, model_name, method, feature_space)

    if prompt_sections is not None:
        prompt_text = [
            f"# {benchmark}/{dataset_name} {method} prompts",
            "",
        ]
        for idx, section in enumerate(prompt_sections, start=1):
            label = section.get("label") or f"prompt {idx}"
            prompt_text.extend([f"## {idx}. {label}", section["prompt"], ""])
        prompt_path.write_text("\n".join(prompt_text), encoding="utf-8")

    result = {
        "benchmark": benchmark,
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "feature_space": feature_space,
        "temperature": temperature,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "token_usage": token_usage_from_calls(llm_calls),
        "response": response,
        "llm_calls": llm_calls,
        "prompt_path": str(prompt_path),
    }
    if extra_config:
        result["config"] = extra_config

    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def generate_rank(
    benchmark: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    feature_cards: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    feature_space: str = "semantic",
) -> Path:
    prompt = build_rank_prompt(benchmark, dataset_name, feature_cards)
    payload, llm_result, _ = call_validated_payload_with_retry(
        model_name,
        prompt,
        temperature,
        max_retries,
        retry_sleep,
        use_web_search,
        lambda payload: validate_ranking(payload, feature_index),
    )
    llm_call = {"prompt": prompt, **llm_result}
    return write_result(
        benchmark,
        dataset_name,
        model_name,
        "rank",
        temperature,
        payload,
        [llm_call],
        prompt_sections=[{"label": "rank", "prompt": prompt}],
        feature_space=feature_space,
    )


def generate_score(
    benchmark: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    feature_cards: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    feature_space: str = "semantic",
) -> Path:
    calls = []
    prompt_sections = []
    per_feature = []
    for feature_card in feature_cards:
        feature_id = int(feature_card["feature_id"])
        feature_name = feature_card["feature"]
        sample_payloads = []
        sample_scores = []
        for sample_idx in range(score_samples):
            prompt = build_score_prompt(benchmark, dataset_name, feature_card)
            payload, llm_result, score = call_validated_payload_with_retry(
                model_name,
                prompt,
                temperature,
                max_retries,
                retry_sleep,
                use_web_search,
                lambda payload, feature_id=feature_id, feature_name=feature_name: validate_score_payload(
                    payload,
                    feature_id,
                    feature_name,
                ),
            )
            sample_payloads.append(payload)
            sample_scores.append(score)
            calls.append({
                "feature_id": feature_id,
                "feature": feature_name,
                "sample": sample_idx + 1,
                "prompt": prompt,
                **llm_result,
            })
            if sample_idx == 0:
                prompt_sections.append({
                    "label": f"feature_id={feature_id} feature={feature_name}",
                    "prompt": prompt,
                })
            if call_delay > 0:
                time.sleep(call_delay)
        per_feature.append({
            "feature_id": feature_id,
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
                "feature_id": int(next(card["feature_id"] for card in feature_cards if card["feature"] == item["feature"])),
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
        benchmark,
        dataset_name,
        model_name,
        "score",
        temperature,
        response,
        calls,
        extra_config={"score_samples": score_samples},
        prompt_sections=prompt_sections,
        feature_space=feature_space,
    )


def generate_score_all(
    benchmark: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    feature_cards: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    feature_space: str = "semantic",
) -> Path:
    calls = []
    prompt_sections = []
    samples_by_feature = {feature_name: [] for feature_name in feature_names(feature_index)}
    sample_payloads = []

    for sample_idx in range(score_samples):
        prompt = build_score_all_prompt(benchmark, dataset_name, feature_cards)
        payload, llm_result, sample_scores = call_validated_payload_with_retry(
            model_name,
            prompt,
            temperature,
            max_retries,
            retry_sleep,
            use_web_search,
            lambda payload: validate_score_all_payload(payload, feature_index),
        )
        sample_payloads.append(payload)
        for feature_name, score in sample_scores.items():
            samples_by_feature[feature_name].append(score)
        calls.append({
            "sample": sample_idx + 1,
            "prompt": prompt,
            **llm_result,
        })
        if sample_idx == 0:
            prompt_sections.append({
                "label": "score_all",
                "prompt": prompt,
            })
        if call_delay > 0:
            time.sleep(call_delay)

    per_feature = []
    for feature_name, sample_scores in samples_by_feature.items():
        per_feature.append({
            "feature_id": next(
                int(card["feature_id"])
                for card in feature_cards
                if card["feature"] == feature_name
            ),
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
                "feature_id": int(next(card["feature_id"] for card in feature_cards if card["feature"] == item["feature"])),
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
        benchmark,
        dataset_name,
        model_name,
        "score_all",
        temperature,
        response,
        calls,
        extra_config={"score_samples": score_samples},
        prompt_sections=prompt_sections,
        feature_space=feature_space,
    )


def generate_seq(
    benchmark: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    feature_cards: List[Dict[str, Any]],
    model_name: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    feature_space: str = "semantic",
) -> Path:
    path = output_path(benchmark, dataset_name, model_name, "seq", feature_space)
    partial_path = path.with_suffix(".partial.json")
    calls = []
    prompt_sections = []
    selected_features = []
    steps = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        steps = partial["response"]["ranking"]
        calls = partial.get("llm_calls", [])
        prompt_sections = partial.get("prompt_sections", [])
        selected_features = [
            {"feature_id": int(step["feature_id"]), "feature": step["feature"]}
            for step in steps
        ]

    def write_partial() -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial = {
            "benchmark": benchmark,
            "dataset": dataset_name,
            "model": model_name,
            "method": "seq",
            "feature_space": feature_space,
            "temperature": temperature,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "token_usage": token_usage_from_calls(calls),
            "response": {"ranking": steps},
            "llm_calls": calls,
            "prompt_sections": prompt_sections,
        }
        partial_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cards_by_id = feature_card_by_id(feature_cards)
    selected_ids = {int(item["feature_id"]) for item in selected_features}
    remaining_feature_ids = [idx for idx in sorted(feature_index) if idx not in selected_ids]

    while remaining_feature_ids:
        remaining_features = [
            cards_by_id[idx]
            for idx in remaining_feature_ids
        ]
        prompt = build_seq_prompt(benchmark, dataset_name, remaining_features, selected_features)
        payload, llm_result, selected = call_validated_payload_with_retry(
            model_name,
            prompt,
            temperature,
            max_retries,
            retry_sleep,
            use_web_search,
            lambda payload: validate_seq_payload(payload, remaining_feature_ids, feature_index),
        )
        selected_feature_id, selected_feature = selected
        remaining_feature_ids.remove(selected_feature_id)
        selected_features.append({"feature_id": selected_feature_id, "feature": selected_feature})
        steps.append({
            "rank": len(selected_features),
            "feature_id": selected_feature_id,
            "feature": selected_feature,
            "rationale": payload.get("rationale", ""),
        })
        calls.append({
            "step": len(selected_features),
            "prompt": prompt,
            **llm_result,
        })
        prompt_sections.append({
            "label": f"step={len(selected_features)}",
            "prompt": prompt,
        })
        write_partial()

    response = {"ranking": steps}
    validate_ranking(response, feature_index)
    final_path = write_result(
        benchmark,
        dataset_name,
        model_name,
        "seq",
        temperature,
        response,
        calls,
        prompt_sections=prompt_sections,
        feature_space=feature_space,
    )
    if partial_path.exists():
        partial_path.unlink()
    return final_path


def generate_method(
    method: str,
    benchmark: str,
    dataset_name: str,
    feature_index: Dict[int, str],
    feature_cards: List[Dict[str, Any]],
    model_name: str,
    rank_temperature: float,
    score_temperature: float,
    seq_temperature: float,
    score_samples: int,
    call_delay: float,
    max_retries: int,
    retry_sleep: float,
    use_web_search: bool,
    skip_existing: bool,
    feature_space: str = "semantic",
) -> Path:
    path = output_path(benchmark, dataset_name, model_name, method, feature_space)
    if skip_existing and path.exists():
        return path
    if method == "rank":
        return generate_rank(
            benchmark,
            dataset_name,
            feature_index,
            feature_cards,
            model_name,
            rank_temperature,
            max_retries,
            retry_sleep,
            use_web_search,
            feature_space,
        )
    if method == "score":
        return generate_score(
            benchmark,
            dataset_name,
            feature_index,
            feature_cards,
            model_name,
            score_temperature,
            score_samples,
            call_delay,
            max_retries,
            retry_sleep,
            use_web_search,
            feature_space,
        )
    if method == "score_all":
        return generate_score_all(
            benchmark,
            dataset_name,
            feature_index,
            feature_cards,
            model_name,
            score_temperature,
            score_samples,
            call_delay,
            max_retries,
            retry_sleep,
            use_web_search,
            feature_space,
        )
    if method == "seq":
        return generate_seq(
            benchmark,
            dataset_name,
            feature_index,
            feature_cards,
            model_name,
            seq_temperature,
            max_retries,
            retry_sleep,
            use_web_search,
            feature_space,
        )
    valid = ", ".join(RANKING_METHODS)
    raise ValueError(f"Unsupported method: {method}. Valid methods: {valid}")
