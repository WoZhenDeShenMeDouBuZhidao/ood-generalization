import argparse
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from data.acs.config import (
    ACS_DATASET_ORDER,
    ACS_TASK_CONFIGS,
    feature_index_for as acs_feature_index_for,
    test_states_for as acs_test_states_for,
)
from data.synthetic_ood.dataset import (
    SYNTHETIC_DATASET_ORDER,
    SYNTHETIC_REPORT_ID_GROUPS,
    SYNTHETIC_REPORT_OOD_GROUPS,
    SYNTHETIC_TEST_GROUPS,
    SyntheticConfig,
    feature_index_for as synthetic_feature_index_for,
    feature_loss_weights_for as synthetic_feature_loss_weights_for,
)
from data.tableshift.config import (
    TABLESHIFT_DATASET_ORDER,
    TABLESHIFT_REPORT_ID_GROUPS,
    TABLESHIFT_REPORT_OOD_GROUPS,
    TABLESHIFT_TASK_CONFIGS,
    TABLESHIFT_TEST_GROUPS,
)
from data.whyshift.config import (
    WHYSHIFT_DATASET_ORDER,
    WHYSHIFT_TASK_CONFIGS,
    feature_index_for as whyshift_feature_index_for,
    test_domains_for as whyshift_test_domains_for,
)
from src.data_cache import load_or_build_dataset_cache
from src.paths import result_output_path
from src.ranking import (
    DEFAULT_RANKING_MODEL,
    RANKING_FEATURE_SPACES,
    SCORE_WEIGHT_METHODS,
    feature_loss_weights_from_ranking,
    llm_lasso_penalty_factors_from_ranking,
    removed_feature_indices_from_selected_names,
    selected_feature_names_from_ranking,
)
from src.semantic_features import (
    model_to_semantic_feature_map_for_spec,
    semantic_feature_index_for_spec,
    semantic_values_to_model_values,
)
from src.utils import set_seeds


BENCHMARK_ORDER = ("acs", "whyshift", "tableshift", "synthetic_ood")
MODEL_SEEDS = [9803, 38224, 8113, 4854, 98825]

ACS_REWEIGHTING_BY_DATASET = {
    "acsincome": True,
    "acsemployment": True,
    "acsemploymentfiltered": True,
    "acshealthinsurance": True,
    "acsincomepovertyratio": True,
    "acsmobility": True,
    "acspubliccoverage": True,
    "acstraveltime": True,
}

WHYSHIFT_REWEIGHTING_BY_DATASET = {
    "taxi": True,
    "accident": True,
}

TABLESHIFT_REWEIGHTING_BY_DATASET = {
    "college_scorecard": True,
    "diabetes_readmission": True,
    "nhanes_lead": True,
    "nhanes_cholesterol": True,
    "anes": True,
    "acsfoodstamps": True,
    "brfss_diabetes": True,
    "brfss_blood_pressure": True,
}

DATASET_SIZE_DEFAULTS = {
    "acs": {
        "max_train_val_size": 10000,
        "max_per_test_size": 0,
    },
    "whyshift": {
        "taxi": {
            "max_train_val_size": 8000,
            "max_per_test_size": 8000,
        },
        "accident": {
            "max_train_val_size": 8000,
            "max_per_test_size": 8000,
        },
    },
    "tableshift": {
        "max_train_val_size": 10000,
        "max_per_test_size": 0,
    },
    "synthetic_ood": {
        "max_train_val_size": 0,
        "max_per_test_size": 0,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    benchmark: str
    dataset: str
    loader_dataset: str
    artifact_dataset: str
    train_val_group: str
    test_groups: Tuple[str, ...]
    feature_index: Dict[int, str]
    removed_feature_indices: Tuple[int, ...]
    data_removed_feature_indices: Tuple[int, ...]
    dataset_config: Optional[Any]
    reweighting: bool
    report_id_groups: Tuple[str, ...] = ()
    report_ood_groups: Tuple[str, ...] = ()


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=BENCHMARK_ORDER, default="acs")
    parser.add_argument("--dataset")
    parser.add_argument("--max-train-val-size", type=int)
    parser.add_argument("--max-per-test-size", type=int)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument(
        "--categorical-encoding",
        choices=("auto", "integer", "one_hot"),
        default="auto",
        help="Categorical feature encoding; auto preserves each dataset's original format.",
    )


def dataset_names_for_benchmark(benchmark: str) -> Tuple[str, ...]:
    if benchmark == "acs":
        return ACS_DATASET_ORDER
    if benchmark == "whyshift":
        return WHYSHIFT_DATASET_ORDER
    if benchmark == "tableshift":
        return TABLESHIFT_DATASET_ORDER
    if benchmark == "synthetic_ood":
        return SYNTHETIC_DATASET_ORDER
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _normalize_dataset_name(benchmark: str, dataset: Optional[str]) -> str:
    if dataset is None:
        raise ValueError("dataset must be provided here.")
    valid_names = dataset_names_for_benchmark(benchmark)
    if dataset not in valid_names:
        valid = ", ".join(valid_names)
        raise ValueError(f"Unknown dataset `{dataset}` for benchmark `{benchmark}`. Valid: {valid}")
    return dataset


def _default_size_config(benchmark: str, dataset: str) -> Dict[str, int]:
    if benchmark == "whyshift":
        defaults = DATASET_SIZE_DEFAULTS["whyshift"][dataset]
    else:
        defaults = DATASET_SIZE_DEFAULTS[benchmark]
    return dict(defaults)


def _size_config(args: argparse.Namespace, benchmark: str, dataset: str) -> Dict[str, int]:
    defaults = _default_size_config(benchmark, dataset)
    config = {
        "max_train_val_size": defaults["max_train_val_size"],
        "max_per_test_size": defaults["max_per_test_size"],
    }
    for key in config:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = int(value)
    if config["max_train_val_size"] < 0 or config["max_per_test_size"] < 0:
        raise ValueError("max-train-val-size and max-per-test-size must be non-negative; use 0 for no cap.")
    return config


def _size_cache_suffix(size_config: Dict[str, int]) -> str:
    return (
        f"trv{size_config['max_train_val_size']}"
        f"__test{size_config['max_per_test_size']}"
    )


def _whyshift_dataset_config(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    need_preprocess = not bool(getattr(args, "no_preprocess", False))
    categorical_encoding = getattr(args, "categorical_encoding", "auto")
    size_config = _size_config(args, "whyshift", dataset)
    native_encoding = "integer" if dataset == "taxi" else "one_hot"
    encoding_suffix = (
        ""
        if categorical_encoding in {"auto", native_encoding}
        else f"__enc{categorical_encoding}"
    )
    return {
        "need_preprocess": need_preprocess,
        "categorical_encoding": categorical_encoding,
        **size_config,
        "random_state": 42,
        "cache_suffix": (
            f"prep{int(need_preprocess)}"
            f"__{_size_cache_suffix(size_config)}"
            f"{encoding_suffix}"
        ),
    }


def _tableshift_dataset_config(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    categorical_encoding = getattr(args, "categorical_encoding", "auto")
    size_config = _size_config(args, "tableshift", "default")
    native_encoding = "integer" if dataset == "college_scorecard" else "one_hot"
    encoding_suffix = (
        ""
        if categorical_encoding in {"auto", native_encoding}
        else f"__enc{categorical_encoding}"
    )
    return {
        "categorical_encoding": categorical_encoding,
        **size_config,
        "random_state": 42,
        "cache_suffix": f"oodtest__{_size_cache_suffix(size_config)}{encoding_suffix}",
    }


def dataset_spec(
    benchmark: str,
    dataset: str,
    args: Optional[argparse.Namespace] = None,
) -> DatasetSpec:
    dataset = _normalize_dataset_name(benchmark, dataset)
    size_config = _size_config(args or argparse.Namespace(), benchmark, dataset)
    if benchmark == "acs":
        config = ACS_TASK_CONFIGS[dataset]
        categorical_encoding = getattr(args or argparse.Namespace(), "categorical_encoding", "auto")
        source_removed = tuple(config.removed_feature_indices)
        dynamic_features = categorical_encoding == "one_hot"
        dataset_config = {
            "resampling": False,
            "standardize": False,
            "categorical_encoding": categorical_encoding,
            **size_config,
        }
        if dynamic_features:
            dataset_config["cache_suffix"] = (
                f"enc{categorical_encoding}__rs0__std0__{_size_cache_suffix(size_config)}"
            )
        return DatasetSpec(
            benchmark="acs",
            dataset=dataset,
            loader_dataset=dataset,
            artifact_dataset=dataset,
            train_val_group=config.train_val_state,
            test_groups=tuple(acs_test_states_for(dataset)),
            feature_index={} if dynamic_features else acs_feature_index_for(dataset),
            removed_feature_indices=() if dynamic_features else source_removed,
            data_removed_feature_indices=source_removed,
            dataset_config=dataset_config,
            reweighting=ACS_REWEIGHTING_BY_DATASET[dataset],
        )

    if benchmark == "whyshift":
        config = WHYSHIFT_TASK_CONFIGS[dataset]
        categorical_encoding = getattr(args or argparse.Namespace(), "categorical_encoding", "auto")
        source_removed = tuple(config.removed_feature_indices)
        native_encoding = "integer" if dataset == "taxi" else "one_hot"
        dynamic_features = categorical_encoding not in {"auto", native_encoding}
        return DatasetSpec(
            benchmark="whyshift",
            dataset=dataset,
            loader_dataset=dataset,
            artifact_dataset=dataset,
            train_val_group=config.train_val_domain,
            test_groups=tuple(whyshift_test_domains_for(dataset)),
            feature_index={} if dynamic_features else whyshift_feature_index_for(dataset),
            removed_feature_indices=() if dynamic_features else source_removed,
            data_removed_feature_indices=source_removed,
            dataset_config=_whyshift_dataset_config(args or argparse.Namespace(), dataset),
            reweighting=WHYSHIFT_REWEIGHTING_BY_DATASET[dataset],
        )

    if benchmark == "tableshift":
        config = TABLESHIFT_TASK_CONFIGS[dataset]
        return DatasetSpec(
            benchmark="tableshift",
            dataset=dataset,
            loader_dataset=dataset,
            artifact_dataset=dataset,
            train_val_group="train",
            test_groups=TABLESHIFT_TEST_GROUPS,
            feature_index={},
            removed_feature_indices=tuple(config.removed_feature_indices),
            data_removed_feature_indices=tuple(config.removed_feature_indices),
            dataset_config=_tableshift_dataset_config(args or argparse.Namespace(), dataset),
            reweighting=TABLESHIFT_REWEIGHTING_BY_DATASET[dataset],
            report_id_groups=TABLESHIFT_REPORT_ID_GROUPS,
            report_ood_groups=TABLESHIFT_REPORT_OOD_GROUPS,
        )

    if benchmark == "synthetic_ood":
        cache_suffix = _size_cache_suffix(size_config)
        removed_feature_indices = (
            (0, 1)
            if dataset in {"categorical_integer", "categorical_onehot"}
            else ()
        )
        return DatasetSpec(
            benchmark="synthetic_ood",
            dataset=dataset,
            loader_dataset="synthetic_ood",
            artifact_dataset=dataset,
            train_val_group="train_env",
            test_groups=SYNTHETIC_TEST_GROUPS,
            feature_index=synthetic_feature_index_for(dataset),
            removed_feature_indices=removed_feature_indices,
            data_removed_feature_indices=removed_feature_indices,
            dataset_config=SyntheticConfig(
                dataset=dataset,
                max_train_val_size=size_config["max_train_val_size"],
                max_per_test_size=size_config["max_per_test_size"],
                cache_suffix=cache_suffix,
            ),
            reweighting=False,
            report_id_groups=SYNTHETIC_REPORT_ID_GROUPS,
            report_ood_groups=SYNTHETIC_REPORT_OOD_GROUPS,
        )

    raise ValueError(f"Unknown benchmark: {benchmark}")


def iter_dataset_specs(args: argparse.Namespace) -> Iterator[DatasetSpec]:
    benchmark = args.benchmark
    datasets = [args.dataset] if args.dataset else dataset_names_for_benchmark(benchmark)
    for dataset in datasets:
        yield dataset_spec(benchmark, dataset, args)


def result_path_for(spec: DatasetSpec, method: str, filename: str = "result.json"):
    return result_output_path(spec.benchmark, method, spec.artifact_dataset, filename=filename)


def load_cached_dataset(spec: DatasetSpec):
    from src.main import data_loading_wrapper

    removed_feature_indices = list(spec.data_removed_feature_indices)

    def _build_dataset():
        set_seeds(67)
        return data_loading_wrapper[spec.loader_dataset](
            removed_feature_indices,
            spec.train_val_group,
            list(spec.test_groups),
            0.2,
            spec.dataset_config,
        )

    return load_or_build_dataset_cache(
        spec.benchmark,
        spec.artifact_dataset,
        removed_feature_indices,
        spec.dataset_config,
        _build_dataset,
    )


def feature_index_from_dataset(spec: DatasetSpec, train=None) -> Dict[int, str]:
    if spec.feature_index:
        return dict(spec.feature_index)
    feature_names = getattr(train, "feature_names", None)
    if feature_names is None:
        raise ValueError(f"Feature index for {spec.benchmark}/{spec.dataset} is only available after loading data.")
    return {idx: str(name) for idx, name in enumerate(feature_names)}


def spec_with_feature_index(spec: DatasetSpec) -> DatasetSpec:
    if spec.feature_index:
        return spec
    train, _, _, _, _ = load_cached_dataset(spec)
    return replace(spec, feature_index=feature_index_from_dataset(spec, train))


def _semantic_values_to_model_values(
    spec: DatasetSpec,
    semantic_values: Dict[str, float],
    semantic_expansion_policy: str = "shared",
) -> Dict[str, float]:
    return semantic_values_to_model_values(
        spec,
        semantic_values,
        expansion_policy=semantic_expansion_policy,
    )


def _ordered_feature_names(feature_index: Dict[int, str]) -> Tuple[str, ...]:
    return tuple(feature_index[idx] for idx in sorted(feature_index))


def ranking_feature_index_for_spec(
    spec: DatasetSpec,
    ranking_feature_space: str,
) -> Tuple[Dict[int, str], str]:
    if ranking_feature_space not in RANKING_FEATURE_SPACES:
        valid = ", ".join(RANKING_FEATURE_SPACES)
        raise ValueError(f"Unsupported ranking feature space: {ranking_feature_space}. Valid spaces: {valid}")
    semantic_feature_index = semantic_feature_index_for_spec(spec)
    if ranking_feature_space == "semantic":
        return semantic_feature_index, "semantic"
    if _ordered_feature_names(spec.feature_index) == _ordered_feature_names(semantic_feature_index):
        return semantic_feature_index, "semantic"
    return dict(spec.feature_index), "model"


def feature_loss_weights_for_spec(
    spec: DatasetSpec,
    ranking_method: str,
    feature_weight_mode: str,
    model_name: str = DEFAULT_RANKING_MODEL,
    ranking_feature_space: str = "semantic",
    semantic_expansion_policy: str = "shared",
) -> Dict[str, float]:
    if spec.benchmark == "synthetic_ood":
        return synthetic_feature_loss_weights_for(spec.dataset)
    ranking_feature_index, resolved_feature_space = ranking_feature_index_for_spec(
        spec,
        ranking_feature_space,
    )
    ranking_weights = feature_loss_weights_from_ranking(
        spec.dataset,
        ranking_feature_index,
        method=ranking_method,
        model_name=model_name,
        weight_mode=feature_weight_mode,
        benchmark=spec.benchmark,
        feature_space=resolved_feature_space,
    )
    if resolved_feature_space == "semantic":
        return _semantic_values_to_model_values(
            spec,
            ranking_weights,
            semantic_expansion_policy=semantic_expansion_policy,
        )
    return ranking_weights


def selected_feature_names_for_spec(
    spec: DatasetSpec,
    ranking_method: str,
    selection_mode: str,
    top_p: float = 1.0,
    score_threshold: float = 0.0,
    model_name: str = DEFAULT_RANKING_MODEL,
    ranking_feature_space: str = "semantic",
) -> Tuple[str, ...]:
    if spec.benchmark == "synthetic_ood":
        feature_weights = synthetic_feature_loss_weights_for(spec.dataset)
        ranked_features = sorted(
            feature_weights,
            key=lambda name: feature_weights[name],
            reverse=True,
        )
        if selection_mode == "score_threshold":
            return tuple(
                feature
                for feature in ranked_features
                if feature_weights[feature] >= score_threshold
            )
        keep_count = max(1, int(np.ceil(len(ranked_features) * float(top_p))))
        return tuple(ranked_features[:keep_count])

    ranking_feature_index, resolved_feature_space = ranking_feature_index_for_spec(
        spec,
        ranking_feature_space,
    )
    selected_features = set(selected_feature_names_from_ranking(
        spec.dataset,
        ranking_feature_index,
        method=ranking_method,
        selection_mode=selection_mode,
        top_p=top_p,
        score_threshold=score_threshold,
        model_name=model_name,
        benchmark=spec.benchmark,
        feature_space=resolved_feature_space,
    ))
    if resolved_feature_space == "model":
        return tuple(
            feature_name
            for idx in sorted(spec.feature_index)
            for feature_name in (spec.feature_index[idx],)
            if feature_name in selected_features
        )
    mapping = model_to_semantic_feature_map_for_spec(spec)
    return tuple(
        feature_name
        for idx in sorted(spec.feature_index)
        for feature_name in (spec.feature_index[idx],)
        if mapping[feature_name] in selected_features
    )


def removed_feature_indices_for_selected(
    spec: DatasetSpec,
    selected_feature_names: Tuple[str, ...],
) -> Tuple[int, ...]:
    return tuple(removed_feature_indices_from_selected_names(
        spec.feature_index,
        selected_feature_names,
        spec.removed_feature_indices,
    ))


def llm_lasso_penalty_factors_for_spec(
    spec: DatasetSpec,
    ranking_method: str,
    eta: float,
    penalty_floor: float,
    model_name: str = DEFAULT_RANKING_MODEL,
    ranking_feature_space: str = "semantic",
) -> Dict[str, float]:
    if spec.benchmark == "synthetic_ood":
        feature_weights = synthetic_feature_loss_weights_for(spec.dataset)
        max_score = max(feature_weights.values())
        penalties = {}
        for idx in sorted(spec.feature_index):
            feature_name = spec.feature_index[idx]
            score = feature_weights[feature_name] / max_score
            penalties[feature_name] = max(float(penalty_floor), 1.0 - score) ** float(eta)
        mean_penalty = sum(penalties.values()) / len(penalties)
        return {feature: penalty / mean_penalty for feature, penalty in penalties.items()}

    if ranking_method not in SCORE_WEIGHT_METHODS:
        valid = ", ".join(SCORE_WEIGHT_METHODS)
        raise ValueError(f"LLM-Lasso requires score-based ranking methods: {valid}")
    ranking_feature_index, resolved_feature_space = ranking_feature_index_for_spec(
        spec,
        ranking_feature_space,
    )
    ranking_penalties = llm_lasso_penalty_factors_from_ranking(
        spec.dataset,
        ranking_feature_index,
        method=ranking_method,
        eta=eta,
        model_name=model_name,
        penalty_floor=penalty_floor,
        benchmark=spec.benchmark,
        feature_space=resolved_feature_space,
    )
    if resolved_feature_space == "semantic":
        return _semantic_values_to_model_values(spec, ranking_penalties)
    return ranking_penalties
