import math
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.paths import ranking_artifact_dir


DEFAULT_RANKING_MODEL = "gpt-5.4"
RANKING_METHODS = ("rank", "score", "score_all", "seq")
RANKING_FEATURE_SPACES = ("semantic", "model")
FEATURE_WEIGHT_MODES = ("rank", "score")
SCORE_WEIGHT_METHODS = ("score", "score_all")
SELECTION_MODES = ("top_p", "score_threshold")


def ranking_artifact_path(
    dataset: str,
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> Path:
    if method not in RANKING_METHODS:
        valid = ", ".join(RANKING_METHODS)
        raise ValueError(f"Unsupported ranking method: {method}. Valid methods: {valid}")
    if feature_space not in RANKING_FEATURE_SPACES:
        valid = ", ".join(RANKING_FEATURE_SPACES)
        raise ValueError(f"Unsupported ranking feature space: {feature_space}. Valid spaces: {valid}")

    safe_model_name = model_name.replace("/", "_")
    space_suffix = "" if feature_space == "semantic" else f"_{feature_space}"
    return (
        ranking_artifact_dir(benchmark, dataset)
        / f"{safe_model_name}_{method}{space_suffix}_feature_ranking.json"
    )


def ranking_to_feature_loss_weights(
    ranking: List[Dict[str, Any]],
    feature_index: Dict[int, str],
    weight_mode: str = "rank",
) -> Dict[str, float]:
    if weight_mode not in FEATURE_WEIGHT_MODES:
        valid = ", ".join(FEATURE_WEIGHT_MODES)
        raise ValueError(f"Unsupported feature weight mode: {weight_mode}. Valid modes: {valid}")

    ranking_items = normalized_ranking_items(ranking, feature_index)
    expected_features = [feature_index[idx] for idx in sorted(feature_index)]
    feature_count = len(expected_features)
    score_by_feature = {}
    rank_by_feature = {}
    for item in ranking_items:
        feature_name = item.get("feature")
        rank_by_feature[feature_name] = int(item["rank"])
        if weight_mode == "score":
            try:
                score_by_feature[feature_name] = float(item["score_mean"])
            except KeyError as exc:
                raise ValueError(f"Ranking item missing score_mean for feature: {feature_name}") from exc

    if weight_mode == "score":
        weights = {
            feature_name: score_by_feature[feature_name]
            for feature_name in expected_features
        }
    else:
        weights = {
            feature_name: float(feature_count - rank_by_feature[feature_name] + 1)
            for feature_name in expected_features
        }

    return weights


def normalized_ranking_items(
    ranking: List[Dict[str, Any]],
    feature_index: Dict[int, str],
) -> List[Dict[str, Any]]:
    expected_features = [feature_index[idx] for idx in sorted(feature_index)]
    expected_feature_set = set(expected_features)

    if not isinstance(ranking, list):
        raise ValueError("`ranking` must be a list.")

    items_by_feature = {}
    for item in ranking:
        if not isinstance(item, dict):
            raise ValueError("Every ranking item must be a dict.")
        feature_name = item.get("feature")
        if feature_name not in expected_feature_set:
            continue
        if feature_name in items_by_feature:
            raise ValueError(f"Duplicate feature in ranking: {feature_name}")
        items_by_feature[feature_name] = dict(item)

    if set(items_by_feature) != expected_feature_set:
        missing = sorted(expected_feature_set - set(items_by_feature))
        extra = sorted(
            str(item.get("feature"))
            for item in ranking
            if isinstance(item, dict) and item.get("feature") not in expected_feature_set
        )
        raise ValueError(f"Ranking feature mismatch. Missing: {missing}; extra: {extra}")

    ranks = [int(item["rank"]) for item in items_by_feature.values()]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"Ranks must be unique after filtering unexpected features; got {ranks}.")

    ordered = sorted(items_by_feature.values(), key=lambda item: int(item["rank"]))
    normalized = []
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
        normalized.append(item)
    return normalized


def feature_loss_weights_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    weight_mode: str = "rank",
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> Dict[str, float]:
    if weight_mode == "score" and method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score feature weights are only supported for methods: {valid}")

    path = ranking_artifact_path(
        dataset,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Ranking artifact not found: {path}")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact_method = artifact.get("method")
    if artifact_method != method:
        raise ValueError(f"Expected method `{method}` in {path}, got `{artifact_method}`.")
    artifact_feature_space = artifact.get("feature_space", "semantic")
    if artifact_feature_space != feature_space:
        raise ValueError(
            f"Expected feature_space `{feature_space}` in {path}, got `{artifact_feature_space}`."
        )

    response = artifact.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Ranking artifact missing response object: {path}")

    return ranking_to_feature_loss_weights(
        response.get("ranking"),
        feature_index,
        weight_mode=weight_mode,
    )


def _ranking_response(
    dataset: str,
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> Dict[str, Any]:
    path = ranking_artifact_path(
        dataset,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Ranking artifact not found: {path}")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact_method = artifact.get("method")
    if artifact_method != method:
        raise ValueError(f"Expected method `{method}` in {path}, got `{artifact_method}`.")
    artifact_feature_space = artifact.get("feature_space", "semantic")
    if artifact_feature_space != feature_space:
        raise ValueError(
            f"Expected feature_space `{feature_space}` in {path}, got `{artifact_feature_space}`."
        )

    response = artifact.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Ranking artifact missing response object: {path}")
    return response


def ranking_items_from_artifact(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> List[Dict[str, Any]]:
    response = _ranking_response(
        dataset,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    ranking = response.get("ranking")
    return normalized_ranking_items(ranking, feature_index)


def selected_feature_names_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    selection_mode: str,
    top_p: float = 1.0,
    score_threshold: float = 0.0,
    model_name: str = DEFAULT_RANKING_MODEL,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> List[str]:
    if selection_mode not in SELECTION_MODES:
        valid = ", ".join(SELECTION_MODES)
        raise ValueError(f"Unsupported selection mode: {selection_mode}. Valid modes: {valid}")
    if selection_mode == "score_threshold" and method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score threshold selection only supports methods: {valid}")

    ranking = ranking_items_from_artifact(
        dataset,
        feature_index,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    if selection_mode == "score_threshold":
        selected = [
            item["feature"]
            for item in ranking
            if float(item["score_mean"]) >= float(score_threshold)
        ]
        return selected

    top_p = float(top_p)
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    keep_count = max(1, math.ceil(len(ranking) * top_p))
    return [item["feature"] for item in ranking[:keep_count]]


def removed_feature_indices_from_selected_names(
    feature_index: Dict[int, str],
    selected_feature_names: Sequence[str],
    base_removed_feature_indices: Sequence[int] = (),
) -> List[int]:
    selected_set = set(selected_feature_names)
    removed = {
        idx
        for idx, feature_name in feature_index.items()
        if feature_name not in selected_set
    }
    removed.update(int(idx) for idx in base_removed_feature_indices)
    return sorted(removed)


def score_by_feature_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> Dict[str, float]:
    if method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score-based penalties only support methods: {valid}")

    ranking = ranking_items_from_artifact(
        dataset,
        feature_index,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    score_by_feature = {}
    for item in ranking:
        feature_name = item["feature"]
        try:
            score_by_feature[feature_name] = float(item["score_mean"])
        except KeyError as exc:
            raise ValueError(f"Ranking item missing score_mean for feature: {feature_name}") from exc
    return score_by_feature


def llm_lasso_penalty_factors_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    eta: float,
    model_name: str = DEFAULT_RANKING_MODEL,
    penalty_floor: float = 0.1,
    benchmark: str = "acs",
    feature_space: str = "semantic",
) -> Dict[str, float]:
    score_by_feature = score_by_feature_from_ranking(
        dataset,
        feature_index,
        method,
        model_name,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    penalty_floor = float(penalty_floor)
    if penalty_floor <= 0.0 or penalty_floor >= 1.0:
        raise ValueError(f"penalty_floor must be in (0, 1), got {penalty_floor}.")

    penalties = {}
    for idx in sorted(feature_index):
        feature_name = feature_index[idx]
        score = min(max(score_by_feature[feature_name], 0.0), 1.0)
        base_penalty = max(penalty_floor, 1.0 - score)
        penalties[feature_name] = float(base_penalty ** float(eta))

    mean_penalty = sum(penalties.values()) / len(penalties)
    if mean_penalty <= 0.0:
        return {feature_name: 1.0 for feature_name in penalties}
    return {
        feature_name: penalty / mean_penalty
        for feature_name, penalty in penalties.items()
    }
