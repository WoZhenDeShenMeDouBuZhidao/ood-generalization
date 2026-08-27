from typing import Any, Dict, Sequence, Tuple

from data.acs.config import feature_index_for as acs_feature_index_for
from data.whyshift.config import feature_index_for as whyshift_feature_index_for


SEMANTIC_EXPANSION_POLICIES = ("shared", "split")


def unique_in_order(values: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def tableshift_predictors(dataset: str) -> Tuple[str, ...]:
    try:
        from tableshift.core.tasks import _TASK_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "TableShift semantic feature metadata requires the optional "
            "`tableshift` package."
        ) from exc
    return tuple(str(feature) for feature in _TASK_REGISTRY[dataset].feature_list.predictors)


def longest_prefix_feature(feature_name: str, semantic_features: Sequence[str]) -> str:
    if feature_name in semantic_features:
        return feature_name
    for semantic_feature in sorted(semantic_features, key=len, reverse=True):
        if feature_name.startswith(f"{semantic_feature}_"):
            return semantic_feature
    return feature_name


def whyshift_semantic_name(dataset: str, feature_name: str) -> str:
    if dataset == "accident":
        for prefix in ("Wind_Direction", "Weather_Condition", "Civil_Twilight"):
            if feature_name.startswith(f"{prefix}_"):
                return prefix
    return feature_name


def semantic_feature_index_for_spec(spec: Any) -> Dict[int, str]:
    if spec.benchmark == "tableshift":
        predictors = tableshift_predictors(spec.dataset)
        if spec.feature_index:
            used_features = {
                longest_prefix_feature(spec.feature_index[idx], predictors)
                for idx in sorted(spec.feature_index)
            }
            predictors = tuple(
                feature_name
                for feature_name in predictors
                if feature_name in used_features
            )
        return {
            idx: feature_name
            for idx, feature_name in enumerate(predictors)
        }

    if spec.benchmark == "whyshift":
        original_semantic_features = unique_in_order(
            whyshift_semantic_name(spec.dataset, name)
            for name in whyshift_feature_index_for(spec.dataset).values()
        )
        semantic_features = unique_in_order(
            longest_prefix_feature(spec.feature_index[idx], original_semantic_features)
            for idx in sorted(spec.feature_index)
        )
        return {
            idx: feature_name
            for idx, feature_name in enumerate(semantic_features)
        }

    if spec.benchmark == "acs":
        original_features = tuple(acs_feature_index_for(spec.dataset).values())
        semantic_features = unique_in_order(
            longest_prefix_feature(spec.feature_index[idx], original_features)
            for idx in sorted(spec.feature_index)
        )
        return {
            idx: feature_name
            for idx, feature_name in enumerate(semantic_features)
        }

    return dict(spec.feature_index)


def model_to_semantic_feature_map_for_spec(spec: Any) -> Dict[str, str]:
    if spec.benchmark == "tableshift":
        predictors = tableshift_predictors(spec.dataset)
        return {
            spec.feature_index[idx]: longest_prefix_feature(spec.feature_index[idx], predictors)
            for idx in sorted(spec.feature_index)
        }

    if spec.benchmark == "whyshift":
        semantic_features = unique_in_order(
            whyshift_semantic_name(spec.dataset, name)
            for name in whyshift_feature_index_for(spec.dataset).values()
        )
        return {
            spec.feature_index[idx]: longest_prefix_feature(
                spec.feature_index[idx], semantic_features
            )
            for idx in sorted(spec.feature_index)
        }

    if spec.benchmark == "acs":
        semantic_features = tuple(acs_feature_index_for(spec.dataset).values())
        return {
            spec.feature_index[idx]: longest_prefix_feature(
                spec.feature_index[idx], semantic_features
            )
            for idx in sorted(spec.feature_index)
        }

    return {
        spec.feature_index[idx]: spec.feature_index[idx]
        for idx in sorted(spec.feature_index)
    }


def semantic_values_to_model_values(
    spec: Any,
    semantic_values: Dict[str, float],
    expansion_policy: str = "shared",
) -> Dict[str, float]:
    if expansion_policy not in SEMANTIC_EXPANSION_POLICIES:
        valid = ", ".join(SEMANTIC_EXPANSION_POLICIES)
        raise ValueError(
            f"Unsupported semantic expansion policy: {expansion_policy}. "
            f"Valid policies: {valid}"
        )
    mapping = model_to_semantic_feature_map_for_spec(spec)
    missing = sorted(set(mapping.values()) - set(semantic_values))
    if missing:
        raise ValueError(f"Missing semantic feature values for {spec.benchmark}/{spec.dataset}: {missing[:10]}")
    expansion_counts: Dict[str, int] = {}
    for semantic_name in mapping.values():
        expansion_counts[semantic_name] = expansion_counts.get(semantic_name, 0) + 1
    return {
        feature_name: float(semantic_values[semantic_name]) / (
            expansion_counts[semantic_name] if expansion_policy == "split" else 1
        )
        for feature_name, semantic_name in mapping.items()
    }
