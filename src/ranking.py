import math
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.paths import dataset_artifact_dir


DEFAULT_RANKING_MODEL = "gpt-5.5"
RANKING_METHODS = ("rank", "score", "score_all", "seq")
FEATURE_WEIGHT_MODES = ("rank", "score")
SCORE_WEIGHT_METHODS = ("score", "score_all")
SELECTION_MODES = ("top_p", "score_threshold")


def ranking_artifact_path(
    dataset: str,
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
) -> Path:
    if method not in RANKING_METHODS:
        valid = ", ".join(RANKING_METHODS)
        raise ValueError(f"Unsupported ranking method: {method}. Valid methods: {valid}")

    safe_model_name = model_name.replace("/", "_")
    return (
        dataset_artifact_dir(dataset)
        / "rankings"
        / f"{safe_model_name}_{method}_feature_ranking.json"
    )


def ranking_to_feature_loss_weights(
    ranking: List[Dict[str, Any]],
    feature_index: Dict[int, str],
    weight_mode: str = "rank",
    suppress_bound: float = 0.0,
) -> Dict[str, float]:
    if weight_mode not in FEATURE_WEIGHT_MODES:
        valid = ", ".join(FEATURE_WEIGHT_MODES)
        raise ValueError(f"Unsupported feature weight mode: {weight_mode}. Valid modes: {valid}")
    suppress_bound = float(suppress_bound)

    expected_features = [feature_index[idx] for idx in sorted(feature_index)]
    expected_feature_set = set(expected_features)
    feature_count = len(expected_features)

    if not isinstance(ranking, list):
        raise ValueError("`ranking` must be a list.")

    rank_by_feature = {}
    score_by_feature = {}
    for item in ranking:
        if not isinstance(item, dict):
            raise ValueError("Every ranking item must be a dict.")
        feature_name = item.get("feature")
        if feature_name not in expected_feature_set:
            raise ValueError(f"Unexpected feature in ranking: {feature_name}")
        if feature_name in rank_by_feature:
            raise ValueError(f"Duplicate feature in ranking: {feature_name}")
        rank_by_feature[feature_name] = int(item["rank"])
        if weight_mode == "score":
            try:
                score_by_feature[feature_name] = float(item["score_mean"])
            except KeyError as exc:
                raise ValueError(f"Ranking item missing score_mean for feature: {feature_name}") from exc

    if set(rank_by_feature) != expected_feature_set:
        missing = sorted(expected_feature_set - set(rank_by_feature))
        extra = sorted(set(rank_by_feature) - expected_feature_set)
        raise ValueError(f"Ranking feature mismatch. Missing: {missing}; extra: {extra}")

    ranks = list(rank_by_feature.values())
    expected_ranks = list(range(1, feature_count + 1))
    if sorted(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be unique integers {expected_ranks}; got {ranks}.")

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

    return {
        feature_name: 0.0 if weight < suppress_bound else weight
        for feature_name, weight in weights.items()
    }


def feature_loss_weights_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    weight_mode: str = "rank",
    suppress_bound: float = 0.0,
) -> Dict[str, float]:
    if weight_mode == "score" and method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score feature weights are only supported for methods: {valid}")

    path = ranking_artifact_path(dataset, method, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"Ranking artifact not found: {path}")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact_method = artifact.get("method")
    if artifact_method != method:
        raise ValueError(f"Expected method `{method}` in {path}, got `{artifact_method}`.")

    response = artifact.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Ranking artifact missing response object: {path}")

    return ranking_to_feature_loss_weights(
        response.get("ranking"),
        feature_index,
        weight_mode=weight_mode,
        suppress_bound=suppress_bound,
    )


def _ranking_response(
    dataset: str,
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
) -> Dict[str, Any]:
    path = ranking_artifact_path(dataset, method, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"Ranking artifact not found: {path}")

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact_method = artifact.get("method")
    if artifact_method != method:
        raise ValueError(f"Expected method `{method}` in {path}, got `{artifact_method}`.")

    response = artifact.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Ranking artifact missing response object: {path}")
    return response


def ranking_items_from_artifact(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    model_name: str = DEFAULT_RANKING_MODEL,
) -> List[Dict[str, Any]]:
    response = _ranking_response(dataset, method, model_name)
    ranking = response.get("ranking")
    ranking_to_feature_loss_weights(ranking, feature_index, weight_mode="rank")
    return sorted(ranking, key=lambda item: int(item["rank"]))


def selected_feature_names_from_ranking(
    dataset: str,
    feature_index: Dict[int, str],
    method: str,
    selection_mode: str,
    top_p: float = 1.0,
    score_threshold: float = 0.0,
    model_name: str = DEFAULT_RANKING_MODEL,
) -> List[str]:
    if selection_mode not in SELECTION_MODES:
        valid = ", ".join(SELECTION_MODES)
        raise ValueError(f"Unsupported selection mode: {selection_mode}. Valid modes: {valid}")
    if selection_mode == "score_threshold" and method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score threshold selection only supports methods: {valid}")

    ranking = ranking_items_from_artifact(dataset, feature_index, method, model_name)
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
) -> Dict[str, float]:
    if method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"Score-based penalties only support methods: {valid}")

    ranking = ranking_items_from_artifact(dataset, feature_index, method, model_name)
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
) -> Dict[str, float]:
    score_by_feature = score_by_feature_from_ranking(dataset, feature_index, method, model_name)
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
