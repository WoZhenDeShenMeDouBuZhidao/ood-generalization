import argparse
import datetime
import math
import warnings

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmark_config import (
    MODEL_SEEDS,
    add_benchmark_args,
    feature_index_from_dataset,
    feature_loss_weights_for_spec,
    iter_dataset_specs,
    llm_lasso_penalty_factors_for_spec,
    load_cached_dataset,
    ranking_feature_index_for_spec,
    removed_feature_indices_for_selected,
    result_path_for,
    selected_feature_names_for_spec,
    spec_with_feature_index,
)
from src.main import main
from src.metrics import binary_balanced_accuracy_macro_f1
from src.ranking import (
    DEFAULT_RANKING_MODEL,
    FEATURE_WEIGHT_MODES,
    RANKING_FEATURE_SPACES,
    RANKING_METHODS,
    SELECTION_MODES,
)
from src.semantic_features import (
    SEMANTIC_EXPANSION_POLICIES,
    model_to_semantic_feature_map_for_spec,
)
from src.utils import (
    ExperimentMetrics,
    experiment_metrics_to_dict,
    hyperparameter_selection_score,
    metric_summary,
    metric_summary_to_dict,
    report_metrics_from_values,
    save_json_result,
)


METHODS = (
    "mlp_ce",
    "gradient_regularized_ce",
    "fitce",
    "laat",
    "llm_select",
    "linear",
    "llm_lasso",
)
TORCH_METHODS = {
    "mlp_ce",
    "gradient_regularized_ce",
    "fitce",
    "laat",
    "llm_select",
}


def resolved_reweighting(mode: str, default: bool) -> bool:
    if mode == "auto":
        return default
    return mode == "on"


def _result_method(args: argparse.Namespace) -> str:
    return args.result_method or args.method


def _prior_metadata(args: argparse.Namespace, spec, **details) -> dict:
    if spec.benchmark == "synthetic_ood":
        return {"prior": {"source": "oracle"}}
    _, resolved_feature_space = ranking_feature_index_for_spec(
        spec,
        args.ranking_feature_space,
    )
    return {
        "prior": {
            "source": "ranking_artifact",
            "method": args.ranking_method,
            "model": args.ranking_model,
            "feature_space": args.ranking_feature_space,
            "resolved_feature_space": resolved_feature_space,
            **details,
        }
    }


def _semantic_gradient_group_config(
    spec,
    feature_loss_weights: dict,
    semantic_expansion_policy: str,
) -> tuple[dict, dict]:
    feature_groups = model_to_semantic_feature_map_for_spec(spec)
    grouped_values = {}
    for feature_name, group_name in feature_groups.items():
        grouped_values.setdefault(group_name, []).append(
            float(feature_loss_weights[feature_name])
        )

    group_weights = {}
    for group_name, values in grouped_values.items():
        if semantic_expansion_policy == "split":
            group_weights[group_name] = sum(values)
        else:
            if max(values) - min(values) > 1e-8:
                raise ValueError(
                    f"Shared semantic group {group_name!r} has unequal model weights."
                )
            group_weights[group_name] = values[0]
    return feature_groups, group_weights


def _gradient_alignment_active_groups(
    spec,
    feature_loss_weights: dict,
    semantic_expansion_policy: str,
    resolved_ranking_feature_space: str,
    grad_alignment_space: str,
    top_p: float,
) -> tuple[list[str], list[str], dict[str, str], dict[str, float]]:
    """Select top semantic features and expand them to alignment groups."""
    removed_indices = set(spec.removed_feature_indices)
    all_feature_groups = model_to_semantic_feature_map_for_spec(spec)
    feature_groups = {}
    grouped_values = {}
    semantic_order = []
    for feature_index in sorted(spec.feature_index):
        if feature_index in removed_indices:
            continue
        feature_name = spec.feature_index[feature_index]
        group_name = all_feature_groups[feature_name]
        feature_groups[feature_name] = group_name
        if group_name not in grouped_values:
            grouped_values[group_name] = []
            semantic_order.append(group_name)
        grouped_values[group_name].append(float(feature_loss_weights[feature_name]))

    semantic_weights = {}
    for group_name, values in grouped_values.items():
        if (
            resolved_ranking_feature_space == "semantic"
            and semantic_expansion_policy == "shared"
        ):
            if max(values) - min(values) > 1e-8:
                raise ValueError(
                    f"Shared semantic group {group_name!r} has unequal model weights."
                )
            semantic_weights[group_name] = values[0]
        else:
            semantic_weights[group_name] = sum(values)

    keep_count = max(1, math.ceil(len(semantic_order) * top_p))
    semantic_position = {
        group_name: position for position, group_name in enumerate(semantic_order)
    }
    ranked_semantic_groups = sorted(
        semantic_order,
        key=lambda group_name: (
            -semantic_weights[group_name],
            semantic_position[group_name],
        ),
    )
    selected_semantic_groups = ranked_semantic_groups[:keep_count]
    selected_set = set(selected_semantic_groups)
    if grad_alignment_space == "semantic":
        active_groups = selected_semantic_groups
    else:
        active_groups = [
            feature_name
            for feature_name, group_name in feature_groups.items()
            if group_name in selected_set
        ]
    return active_groups, selected_semantic_groups, feature_groups, semantic_weights


def _torch_main(
    args: argparse.Namespace,
    spec,
    loss_name: str,
    feature_loss_weights: dict,
    loss_kwargs: dict,
    metadata: dict,
    removed_feature_indices=None,
    result_path=None,
    shap_alignment_feature_groups=None,
    shap_alignment_group_weights=None,
    shap_alignment_active_groups=None,
) -> ExperimentMetrics:
    result_method = _result_method(args)
    removed_feature_indices = (
        list(spec.removed_feature_indices)
        if removed_feature_indices is None
        else list(removed_feature_indices)
    )
    data_removed_feature_indices = list(spec.data_removed_feature_indices)
    return main(
        spec.loader_dataset,
        spec.train_val_group,
        list(spec.test_groups),
        spec.feature_index,
        removed_feature_indices,
        feature_loss_weights,
        TRAIN_BATCH=args.train_batch,
        EVAL_BATCH=args.eval_batch,
        LR=args.lr,
        REG_SCALE=args.reg_scale,
        REG_WARMUP_EPOCHS=args.reg_warmup_epochs,
        PATIENCE=args.patience,
        REPEAT=args.repeat,
        MAX_EPOCHS=args.max_epochs,
        DATASET_CONFIG=spec.dataset_config,
        DATA_REMOVED_FEATURE_INDICES=data_removed_feature_indices,
        PLOT_CURVE=args.plot_curve,
        PLOT_SHAP=args.plot_shap or args.plot_test_shap,
        PLOT_TEST_SHAP=args.plot_test_shap,
        SHAP_SAMPLE_SIZE=args.shap_sample_size,
        MODEL_NAME="mlp",
        LOSS_NAME=loss_name,
        LOSS_KWARGS=loss_kwargs,
        device=args.device,
        MODEL_SEEDS=MODEL_SEEDS,
        RESULT_PATH=result_path or result_path_for(spec, result_method),
        RESULT_METADATA=metadata,
        BENCHMARK=spec.benchmark,
        DATASET_ARTIFACT_NAME=spec.artifact_dataset,
        REPORT_ID_GROUPS=list(spec.report_id_groups),
        REPORT_OOD_GROUPS=list(spec.report_ood_groups),
        SHOW_PROGRESS=args.show_progress,
        RECORD_BEST_GRAD_L2=args.record_best_grad_l2,
        SHAP_ALIGNMENT_FEATURE_GROUPS=shap_alignment_feature_groups,
        SHAP_ALIGNMENT_GROUP_WEIGHTS=shap_alignment_group_weights,
        SHAP_ALIGNMENT_ACTIVE_GROUPS=shap_alignment_active_groups,
    )


def _run_torch_method(args: argparse.Namespace, spec) -> ExperimentMetrics:
    result_method = _result_method(args)
    reweighting = resolved_reweighting(args.reweighting, spec.reweighting)
    base_metadata = {
        "method": result_method,
        "baseline": args.method,
    }

    if args.method == "mlp_ce":
        diagnostic_feature_loss_weights = (
            feature_loss_weights_for_spec(
                spec,
                args.ranking_method,
                args.feature_weight_mode,
                model_name=args.ranking_model,
                ranking_feature_space=args.ranking_feature_space,
            )
            if spec.benchmark == "synthetic_ood"
            else {}
        )
        return _torch_main(
            args,
            spec,
            "cross_entropy",
            diagnostic_feature_loss_weights,
            {"reweighting": reweighting},
            base_metadata,
        )

    if args.method == "gradient_regularized_ce":
        spec = spec_with_feature_index(spec)
        loss_kwargs = {"reweighting": reweighting}
        importance_scale = None if args.importance_scale == "none" else args.importance_scale
        if importance_scale is not None:
            loss_kwargs["importance_scale"] = importance_scale
        return _torch_main(
            args,
            spec,
            "gradient_regularized_ce",
            {},
            loss_kwargs,
            base_metadata,
        )

    if args.method in {"fitce", "laat"}:
        spec = spec_with_feature_index(spec)
        resolved_ranking_feature_space = None
        if args.method == "fitce":
            _, resolved_ranking_feature_space = ranking_feature_index_for_spec(
                spec,
                args.ranking_feature_space,
            )
        prior_metadata = _prior_metadata(
            args,
            spec,
            feature_weight_mode=args.feature_weight_mode,
            semantic_expansion_policy=args.semantic_expansion_policy,
        )
        feature_loss_weights = feature_loss_weights_for_spec(
            spec,
            args.ranking_method,
            args.feature_weight_mode,
            model_name=args.ranking_model,
            ranking_feature_space=args.ranking_feature_space,
            semantic_expansion_policy=args.semantic_expansion_policy,
        )
        if args.method == "fitce":
            importance_scale = None if args.importance_scale == "none" else args.importance_scale
            loss_kwargs = {
                "reweighting": reweighting,
                "reweighting_scope": args.reweighting_scope,
                "grad_scale": args.loss_lambda * args.loss_alpha,
                "weight_scale": args.loss_lambda * (1.0 - args.loss_alpha),
                "grad_prob_temperature": args.grad_prob_temperature,
                "importance_scale": importance_scale,
            }
            (
                gradient_active_groups,
                selected_semantic_groups,
                semantic_feature_groups,
                semantic_group_weights,
            ) = _gradient_alignment_active_groups(
                spec,
                feature_loss_weights,
                args.semantic_expansion_policy,
                resolved_ranking_feature_space,
                args.grad_alignment_space,
                args.grad_alignment_top_p,
            )
            if len(selected_semantic_groups) < len(semantic_group_weights):
                loss_kwargs["gradient_active_groups"] = gradient_active_groups
            if args.grad_alignment_space == "semantic":
                if resolved_ranking_feature_space != "semantic":
                    raise ValueError(
                        "Semantic gradient alignment requires a semantic ranking artifact."
                    )
                gradient_feature_groups, gradient_group_weights = (
                    _semantic_gradient_group_config(
                        spec,
                        feature_loss_weights,
                        args.semantic_expansion_policy,
                    )
                )
                loss_kwargs.update({
                    "gradient_feature_groups": gradient_feature_groups,
                    "gradient_group_weights": gradient_group_weights,
                })
            metadata = {
                **base_metadata,
                **prior_metadata,
                "grad_alignment_space": args.grad_alignment_space,
                "grad_alignment_top_p": args.grad_alignment_top_p,
            }
            if args.grad_alignment_top_p < 1.0:
                metadata["grad_alignment_semantic_features"] = selected_semantic_groups
            return _torch_main(
                args,
                spec,
                "feature_importance_target_ce",
                feature_loss_weights,
                loss_kwargs,
                metadata,
                shap_alignment_feature_groups=semantic_feature_groups,
                shap_alignment_group_weights=semantic_group_weights,
                shap_alignment_active_groups=selected_semantic_groups,
            )

        return _torch_main(
            args,
            spec,
            "llm_attribution_aligned_ce",
            feature_loss_weights,
            {"reweighting": reweighting},
            {
                **base_metadata,
                **prior_metadata,
            },
        )

    return _run_llm_select(args, spec)


def _run_llm_select(args: argparse.Namespace, spec) -> ExperimentMetrics:
    spec = spec_with_feature_index(spec)
    prior_metadata = _prior_metadata(args, spec)
    if args.selection_mode == "score_threshold":
        selection_value = args.score_threshold
        selected_features = selected_feature_names_for_spec(
            spec,
            args.ranking_method,
            args.selection_mode,
            score_threshold=selection_value,
            model_name=args.ranking_model,
            ranking_feature_space=args.ranking_feature_space,
        )
    else:
        selection_value = args.top_p
        selected_features = selected_feature_names_for_spec(
            spec,
            args.ranking_method,
            args.selection_mode,
            top_p=selection_value,
            model_name=args.ranking_model,
            ranking_feature_space=args.ranking_feature_space,
        )
    removed_feature_indices = removed_feature_indices_for_selected(spec, selected_features)
    reweighting = resolved_reweighting(args.reweighting, spec.reweighting)
    result_method = _result_method(args)
    return _torch_main(
        args,
        spec,
        "cross_entropy",
        {},
        {"reweighting": reweighting},
        {
            "method": result_method,
            "baseline": "llm_select",
            **prior_metadata,
            "selection": {
                "mode": args.selection_mode,
                "value": selection_value,
                "features": selected_features,
            },
        },
        removed_feature_indices=removed_feature_indices,
        result_path=result_path_for(spec, result_method),
    )


def _as_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    return dataset.X.detach().cpu().numpy(), dataset.Y.detach().cpu().numpy()


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return binary_balanced_accuracy_macro_f1(y, model.predict(X))


def _metrics_from_report(report_metrics: dict) -> ExperimentMetrics:
    return (
        metric_summary(report_metrics["balanced_accuracy"]["id"]),
        metric_summary(report_metrics["balanced_accuracy"]["ood_mean"]),
        metric_summary(report_metrics["balanced_accuracy"]["ood_worst"]),
        metric_summary(report_metrics["balanced_accuracy"]["ood_std"]),
        metric_summary(report_metrics["macro_f1"]["id"]),
        metric_summary(report_metrics["macro_f1"]["ood_mean"]),
        metric_summary(report_metrics["macro_f1"]["ood_worst"]),
        metric_summary(report_metrics["macro_f1"]["ood_std"]),
    )


def _zero_feature_sklearn_result(args: argparse.Namespace, spec, baseline: str, metadata: dict) -> ExperimentMetrics:
    metrics = tuple((0.0, 0.0) for _ in range(8))
    save_json_result(result_path_for(spec, _result_method(args)), {
        "schema_version": 2,
        "status": "all_features_removed",
        "completed_at": datetime.datetime.now(),
        "metadata": metadata,
        "benchmark": spec.benchmark,
        "dataset": spec.loader_dataset,
        "artifact_dataset": spec.artifact_dataset,
        "report_id_groups": list(spec.report_id_groups),
        "report_ood_groups": list(spec.report_ood_groups or spec.test_groups),
        "features": {
            "names": [],
            "removed_indices": sorted(spec.removed_feature_indices),
            "data_removed_indices": sorted(spec.data_removed_feature_indices),
        },
        "selection_metrics": {
            "balanced_accuracy": metric_summary_to_dict((0.0, 0.0)),
            "macro_f1": metric_summary_to_dict((0.0, 0.0)),
        },
        "metrics": experiment_metrics_to_dict(metrics),
        "message": f"No input features remain for {baseline}; recording zero metrics.",
    })
    return metrics


def _run_linear(args: argparse.Namespace, spec) -> ExperimentMetrics:
    result_method = _result_method(args)
    train, val, tests, data_dir, cache_hit = load_cached_dataset(spec)
    feature_index = feature_index_from_dataset(spec, train)
    kept_feature_indices = [
        idx for idx in sorted(feature_index)
        if idx not in set(spec.removed_feature_indices)
    ]
    metadata = {
        "method": result_method,
        "baseline": "linear",
    }
    if not kept_feature_indices:
        return _zero_feature_sklearn_result(args, spec, "linear", metadata)

    X_train, y_train = _as_numpy(train)
    X_val, y_val = _as_numpy(val)
    class_weight = "balanced" if spec.reweighting else None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            class_weight=class_weight,
            max_iter=args.max_iter,
            penalty="l2",
            solver="lbfgs",
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    val_balanced_accuracy, val_macro_f1 = _evaluate(model, X_val, y_val)
    metrics, test_group_metrics = _sklearn_report_metrics(
        model,
        spec,
        tests,
        val_balanced_accuracy,
        val_macro_f1,
    )
    score = hyperparameter_selection_score(metrics)

    _save_sklearn_result(
        args,
        spec,
        metadata,
        feature_index,
        data_dir,
        cache_hit,
        {
            "name": "logistic_regression",
            "C": args.c,
            "max_iter": args.max_iter,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": class_weight,
            "standard_scaler": "fit_on_train_only",
        },
        {"C": args.c},
        metrics,
        test_group_metrics,
        val_balanced_accuracy,
        val_macro_f1,
        score,
    )
    return metrics


def _weighted_l1_features(X: np.ndarray, penalty_factors: np.ndarray) -> np.ndarray:
    return X / np.clip(penalty_factors, 1e-6, None)


def _run_llm_lasso(args: argparse.Namespace, spec) -> ExperimentMetrics:
    spec = spec_with_feature_index(spec)
    prior_metadata = _prior_metadata(args, spec)
    result_method = _result_method(args)
    feature_index = spec.feature_index
    kept_feature_indices = [
        idx for idx in sorted(feature_index)
        if idx not in set(spec.removed_feature_indices)
    ]
    metadata = {
        "method": result_method,
        "baseline": "llm_lasso",
        **prior_metadata,
    }
    if not kept_feature_indices:
        return _zero_feature_sklearn_result(args, spec, "llm_lasso", metadata)

    train, val, tests, data_dir, cache_hit = load_cached_dataset(spec)
    X_train, y_train = _as_numpy(train)
    X_val, y_val = _as_numpy(val)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    class_weight = None if args.class_weight == "none" else args.class_weight
    penalties_by_feature = llm_lasso_penalty_factors_for_spec(
        spec,
        args.ranking_method,
        eta=args.eta,
        penalty_floor=args.penalty_floor,
        model_name=args.ranking_model,
        ranking_feature_space=args.ranking_feature_space,
    )
    penalty_factors = np.array([
        penalties_by_feature[feature_index[idx]]
        for idx in kept_feature_indices
    ], dtype=float)
    X_train_weighted = _weighted_l1_features(X_train_scaled, penalty_factors)

    model = LogisticRegression(
        C=args.c,
        class_weight=class_weight,
        max_iter=args.max_iter,
        penalty="l1",
        random_state=67,
        solver="saga",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_weighted, y_train)
    val_balanced_accuracy, val_macro_f1 = _evaluate_weighted_l1(
        model,
        scaler,
        penalty_factors,
        X_val,
        y_val,
    )
    metrics, test_group_metrics = _weighted_l1_report_metrics(
        model,
        scaler,
        penalty_factors,
        spec,
        tests,
        val_balanced_accuracy,
        val_macro_f1,
    )
    score = hyperparameter_selection_score(metrics)

    _save_sklearn_result(
        args,
        spec,
        metadata,
        feature_index,
        data_dir,
        cache_hit,
        {
            "name": "weighted_l1_logistic_regression",
            "eta": args.eta,
            "C": args.c,
            "max_iter": args.max_iter,
            "penalty": "l1",
            "solver": "saga",
            "class_weight": class_weight,
            "standard_scaler": "fit_on_train_only",
            "penalty_floor": args.penalty_floor,
            "llm_lasso_reference": "official implementation uses LLM penalty factors with a weighted Lasso solver; this sklearn path applies the same weighted-L1 objective via feature rescaling.",
        },
        {"eta": args.eta, "C": args.c},
        metrics,
        test_group_metrics,
        val_balanced_accuracy,
        val_macro_f1,
        score,
    )
    return metrics


def _evaluate_weighted_l1(
    model,
    scaler: StandardScaler,
    penalty_factors: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    preds = model.predict(_weighted_l1_features(scaler.transform(X), penalty_factors))
    return binary_balanced_accuracy_macro_f1(y, preds)


def _sklearn_report_metrics(
    model,
    spec,
    tests,
    val_balanced_accuracy: float,
    val_macro_f1: float,
):
    test_balanced_accuracies = []
    test_macro_f1s = []
    test_group_metrics = []
    for group, test in zip(spec.test_groups, tests):
        X_test, y_test = _as_numpy(test)
        test_balanced_accuracy, test_macro_f1 = _evaluate(model, X_test, y_test)
        test_balanced_accuracies.append(test_balanced_accuracy)
        test_macro_f1s.append(test_macro_f1)
        test_group_metrics.append({
            "group": group,
            "balanced_accuracy": test_balanced_accuracy,
            "macro_f1": test_macro_f1,
        })
    return (
        _report_metrics(
            spec,
            test_balanced_accuracies,
            test_macro_f1s,
            val_balanced_accuracy,
            val_macro_f1,
        ),
        test_group_metrics,
    )


def _weighted_l1_report_metrics(
    model,
    scaler: StandardScaler,
    penalty_factors: np.ndarray,
    spec,
    tests,
    val_balanced_accuracy: float,
    val_macro_f1: float,
):
    test_balanced_accuracies = []
    test_macro_f1s = []
    test_group_metrics = []
    for group, test in zip(spec.test_groups, tests):
        X_test, y_test = _as_numpy(test)
        test_balanced_accuracy, test_macro_f1 = _evaluate_weighted_l1(
            model,
            scaler,
            penalty_factors,
            X_test,
            y_test,
        )
        test_balanced_accuracies.append(test_balanced_accuracy)
        test_macro_f1s.append(test_macro_f1)
        test_group_metrics.append({
            "group": group,
            "balanced_accuracy": test_balanced_accuracy,
            "macro_f1": test_macro_f1,
        })
    return (
        _report_metrics(
            spec,
            test_balanced_accuracies,
            test_macro_f1s,
            val_balanced_accuracy,
            val_macro_f1,
        ),
        test_group_metrics,
    )


def _report_metrics(
    spec,
    test_balanced_accuracies,
    test_macro_f1s,
    val_balanced_accuracy,
    val_macro_f1,
):
    report_metrics = report_metrics_from_values(
        val_balanced_accuracy,
        val_macro_f1,
        spec.test_groups,
        test_balanced_accuracies,
        test_macro_f1s,
        report_id_groups=spec.report_id_groups,
        report_ood_groups=spec.report_ood_groups,
    )
    return _metrics_from_report(report_metrics)


def _save_sklearn_result(
    args,
    spec,
    metadata,
    feature_index,
    data_dir,
    cache_hit,
    model_metadata,
    hyperparameters,
    metrics,
    test_group_metrics,
    val_balanced_accuracy,
    val_macro_f1,
    score,
) -> None:
    removed_feature_indices = set(spec.removed_feature_indices)
    kept_feature_names = [
        feature_index[index]
        for index in sorted(feature_index)
        if index not in removed_feature_indices
    ]
    save_json_result(result_path_for(spec, _result_method(args)), {
        "schema_version": 2,
        "status": "ok",
        "completed_at": datetime.datetime.now(),
        "metadata": metadata,
        "benchmark": spec.benchmark,
        "dataset": spec.loader_dataset,
        "artifact_dataset": spec.artifact_dataset,
        "train_val_group": spec.train_val_group,
        "test_groups": list(spec.test_groups),
        "report_id_groups": list(spec.report_id_groups),
        "report_ood_groups": list(spec.report_ood_groups or spec.test_groups),
        "features": {
            "names": kept_feature_names,
            "removed_indices": sorted(removed_feature_indices),
            "data_removed_indices": sorted(spec.data_removed_feature_indices),
        },
        "data_cache_key": data_dir.name,
        "data_cache_hit": cache_hit,
        "model": model_metadata,
        "hyperparameters": hyperparameters,
        "hyperparameter_selection_objective": (
            "balanced_accuracy_ood_mean + balanced_accuracy_ood_worst + "
            "macro_f1_ood_mean + macro_f1_ood_worst"
        ),
        "hyperparameter_selection_score": score,
        "test_group_metrics": test_group_metrics,
        "selection_metrics": {
            "balanced_accuracy": metric_summary_to_dict(
                metric_summary(val_balanced_accuracy)
            ),
            "macro_f1": metric_summary_to_dict(metric_summary(val_macro_f1)),
        },
        "metrics": experiment_metrics_to_dict(metrics),
    })


if __name__ == "__main__":
    # Parse CLI.
    parser = argparse.ArgumentParser()

    # Dataset/general experiment arguments.
    add_benchmark_args(parser)
    parser.add_argument("--method", choices=METHODS, default="mlp_ce")
    parser.add_argument("--result-method")

    # Shared PyTorch training arguments: mlp_ce, gradient_regularized_ce, fitce, laat, llm_select.
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--plot-curve", action="store_true", default=False)
    parser.add_argument("--plot-shap", action="store_true", default=False)
    parser.add_argument("--plot-test-shap", action="store_true", default=False)
    parser.add_argument("--shap-sample-size", type=int, default=500)
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument(
        "--record-best-grad-l2",
        action="store_true",
        default=False,
        help=(
            "Record raw logit-margin Grad-L2 at each repeat's best checkpoint; "
            "class aggregation follows the resolved reweighting setting"
        ),
    )
    parser.add_argument("--reweighting", choices=("auto", "on", "off"), default="auto")

    # Shared feature-prior arguments: GPT artifacts for real data, oracle for synthetic data.
    parser.add_argument("--ranking-method", choices=RANKING_METHODS, default="score_all")
    parser.add_argument("--ranking-feature-space", choices=RANKING_FEATURE_SPACES, default="semantic")
    parser.add_argument("--feature-weight-mode", choices=FEATURE_WEIGHT_MODES, default="score")
    parser.add_argument("--ranking-model", default=DEFAULT_RANKING_MODEL)
    parser.add_argument(
        "--semantic-expansion-policy",
        choices=SEMANTIC_EXPANSION_POLICIES,
        default="shared",
        help=(
            "How a semantic feature weight is expanded over one-hot model columns: "
            "'shared' copies the full weight; 'split' divides it equally"
        ),
    )

    # Regularized CE arguments: gradient_regularized_ce, fitce, laat.
    parser.add_argument("--reg-scale", type=float, default=0.0)
    parser.add_argument("--reg-warmup-epochs", type=int, default=0)

    # FITCE-only arguments.
    parser.add_argument("--loss-lambda", type=float, default=16.0)
    parser.add_argument("--loss-alpha", type=float, default=0.75)
    parser.add_argument("--grad-prob-temperature", type=float, default=1.0)
    parser.add_argument(
        "--grad-alignment-space",
        choices=("model", "semantic"),
        default="model",
        help=(
            "FITCE gradient alignment space: model columns or semantic feature "
            "groups aggregated before temperature and normalization"
        ),
    )
    parser.add_argument(
        "--grad-alignment-top-p",
        type=float,
        default=1.0,
        help=(
            "FITCE semantic-feature fraction included in gradient alignment; "
            "masked features remain available to the model and CE loss"
        ),
    )
    parser.add_argument(
        "--reweighting-scope",
        choices=("all", "ce"),
        default="all",
        help=(
            "FITCE class-reweighting scope: 'all' balances CE and regularizer "
            "terms; 'ce' balances CE only"
        ),
    )
    parser.add_argument(
        "--importance-scale",
        choices=("none", "train_std"),
        default="none",
    )

    # LLM-Select-only arguments.
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="top_p")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    # Sklearn arguments: linear and llm_lasso. eta/penalty-floor are LLM-Lasso-only.
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--penalty-floor", type=float, default=0.1)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    args = parser.parse_args()
    if not 0.0 < args.grad_alignment_top_p <= 1.0:
        parser.error("--grad-alignment-top-p must be in (0, 1]")

    # Run selected datasets.
    result_method = _result_method(args)
    for spec in iter_dataset_specs(args):
        if args.method in TORCH_METHODS:
            _run_torch_method(args, spec)
        elif args.method == "linear":
            _run_linear(args, spec)
        else:
            _run_llm_lasso(args, spec)
        print(f"saved {result_path_for(spec, result_method)}")
