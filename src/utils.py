import dataclasses
import datetime
import json
import os
import random
import torch
from openai import OpenAI
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Result formatting

MetricSummary = Tuple[float, float]
ExperimentMetrics = Tuple[
    MetricSummary,
    MetricSummary,
    MetricSummary,
    MetricSummary,
    MetricSummary,
    MetricSummary,
    MetricSummary,
    MetricSummary,
]


def metric_summary(value: float) -> MetricSummary:
    return float(value), 0.0


def hyperparameter_selection_score(metrics: ExperimentMetrics) -> float:
    (
        _balanced_accuracy_id,
        balanced_accuracy_ood_mean,
        balanced_accuracy_ood_worst,
        _balanced_accuracy_ood_std,
        _macro_f1_id,
        macro_f1_ood_mean,
        macro_f1_ood_worst,
        _macro_f1_ood_std,
    ) = metrics
    return (
        float(balanced_accuracy_ood_mean[0])
        + float(balanced_accuracy_ood_worst[0])
        + float(macro_f1_ood_mean[0])
        + float(macro_f1_ood_worst[0])
    )


def report_metrics_from_values(
    validation_balanced_accuracy: float,
    validation_macro_f1: float,
    test_groups: Sequence[str],
    test_balanced_accuracies: Sequence[float],
    test_macro_f1s: Sequence[float],
    report_id_groups: Sequence[str] = (),
    report_ood_groups: Sequence[str] = (),
) -> Dict[str, Dict[str, float]]:
    def _selected_values(values: Sequence[float], groups: Sequence[str], role: str) -> np.ndarray:
        if len(test_groups) != len(values):
            raise ValueError("test_groups and metric values must have the same length.")
        group_values = {str(group): float(value) for group, value in zip(test_groups, values)}
        missing = [str(group) for group in groups if str(group) not in group_values]
        if missing:
            raise ValueError(f"Unknown {role} groups: {', '.join(missing)}")
        return np.asarray([group_values[str(group)] for group in groups], dtype=float)

    def _ood_summary(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            raise ValueError("At least one OOD report group is required.")
        return {
            "ood_mean": float(values.mean()),
            "ood_worst": float(values.min()),
            "ood_std": float(values.std()),
        }

    id_balanced_accuracy_values = (
        _selected_values(test_balanced_accuracies, report_id_groups, "ID report")
        if report_id_groups
        else np.asarray([validation_balanced_accuracy], dtype=float)
    )
    id_macro_f1_values = (
        _selected_values(test_macro_f1s, report_id_groups, "ID report")
        if report_id_groups
        else np.asarray([validation_macro_f1], dtype=float)
    )
    ood_groups = tuple(report_ood_groups) if report_ood_groups else tuple(test_groups)
    ood_balanced_accuracy_values = _selected_values(
        test_balanced_accuracies,
        ood_groups,
        "OOD report",
    )
    ood_macro_f1_values = _selected_values(test_macro_f1s, ood_groups, "OOD report")

    return {
        "balanced_accuracy": {
            "selection": float(validation_balanced_accuracy),
            "id": float(id_balanced_accuracy_values.mean()),
            **_ood_summary(ood_balanced_accuracy_values),
        },
        "macro_f1": {
            "selection": float(validation_macro_f1),
            "id": float(id_macro_f1_values.mean()),
            **_ood_summary(ood_macro_f1_values),
        },
    }


def format_metric(metric: MetricSummary) -> str:
    mean, interval_range = metric
    return f"{mean:.4f} +/- {interval_range:.4f}"


def print_results(metrics: ExperimentMetrics) -> None:
    (
        balanced_accuracy_id,
        balanced_accuracy_ood_mean,
        balanced_accuracy_ood_worst,
        balanced_accuracy_ood_std,
        macro_f1_id,
        macro_f1_ood_mean,
        macro_f1_ood_worst,
        macro_f1_ood_std,
    ) = metrics
    print(
        f"### Balanced Accuracy Results:\n"
        f"- ID: {format_metric(balanced_accuracy_id)}\n"
        f"- OOD MEAN: {format_metric(balanced_accuracy_ood_mean)}\n"
        f"- OOD WORST: {format_metric(balanced_accuracy_ood_worst)}\n"
        f"- OOD STD: {format_metric(balanced_accuracy_ood_std)}\n"
        f"### Macro-F1 Results:\n"
        f"- ID: {format_metric(macro_f1_id)}\n"
        f"- OOD MEAN: {format_metric(macro_f1_ood_mean)}\n"
        f"- OOD WORST: {format_metric(macro_f1_ood_worst)}\n"
        f"- OOD STD: {format_metric(macro_f1_ood_std)}"
    )


def metric_summary_to_dict(metric: MetricSummary) -> Dict[str, float]:
    mean, interval_range = metric
    return {
        "mean": float(mean),
        "ci_range": float(interval_range),
    }


def experiment_metrics_to_dict(metrics: ExperimentMetrics) -> Dict[str, Dict[str, Dict[str, float]]]:
    (
        balanced_accuracy_id,
        balanced_accuracy_ood_mean,
        balanced_accuracy_ood_worst,
        balanced_accuracy_ood_std,
        macro_f1_id,
        macro_f1_ood_mean,
        macro_f1_ood_worst,
        macro_f1_ood_std,
    ) = metrics
    return {
        "balanced_accuracy": {
            "id": metric_summary_to_dict(balanced_accuracy_id),
            "ood_mean": metric_summary_to_dict(balanced_accuracy_ood_mean),
            "ood_worst": metric_summary_to_dict(balanced_accuracy_ood_worst),
            "ood_std": metric_summary_to_dict(balanced_accuracy_ood_std),
        },
        "macro_f1": {
            "id": metric_summary_to_dict(macro_f1_id),
            "ood_mean": metric_summary_to_dict(macro_f1_ood_mean),
            "ood_worst": metric_summary_to_dict(macro_f1_ood_worst),
            "ood_std": metric_summary_to_dict(macro_f1_ood_std),
        },
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def save_json_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# Reproducibility

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# LLM calls

def call_llm(
    model_name: str,
    message: List[Dict[str, str]],
    temperature: float = 0.0,
    use_web_search: bool = False,
) -> Dict[str, Any]:
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('API_URL'),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "120")),
    )
    kwargs = {}
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]
        kwargs["tool_choice"] = "auto"

    try:
        response = client.responses.create(
            model=model_name,
            input=message,
            temperature=temperature,
            **kwargs,
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404 or use_web_search:
            raise
        response = client.chat.completions.create(
            model=model_name,
            messages=message,
            temperature=temperature,
        )
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_id": response.id,
            "response_model": response.model,
        }

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    content = getattr(response, "output_text", None)
    if content is None:
        content = ""
    return {
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "response_id": response.id,
        "response_model": response.model,
    }


# Plotting

def _feature_weight_vector(
    feature_names: Sequence[str],
    feature_loss_weights: Optional[Dict[str, float]],
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    if not feature_loss_weights:
        return None
    weights = np.asarray([
        max(0.0, float(feature_loss_weights.get(feature_name, 0.0)))
        for feature_name in feature_names
    ], dtype=float)
    if weights.sum() <= eps:
        return None
    return weights


def _average_descending_ranks(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(sorted_values[end] - sorted_values[start]) <= eps:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _normalized_spearman_footrule(
    target_values: np.ndarray,
    observed_values: np.ndarray,
    eps: float = 1e-12,
) -> Optional[float]:
    n_features = len(target_values)
    if n_features <= 1 or target_values.max() - target_values.min() <= eps:
        return None
    target_ranks = _average_descending_ranks(target_values, eps=eps)
    observed_ranks = _average_descending_ranks(observed_values, eps=eps)
    max_distance = (n_features * n_features / 2.0) if n_features % 2 == 0 else ((n_features * n_features - 1.0) / 2.0)
    if max_distance <= eps:
        return None
    return float(min(np.abs(target_ranks - observed_ranks).sum() / max_distance, 1.0))


def _total_variation_distance(
    target_values: np.ndarray,
    observed_values: np.ndarray,
    eps: float = 1e-12,
) -> Optional[float]:
    target_sum = target_values.sum()
    observed_sum = observed_values.sum()
    if target_sum <= eps or observed_sum <= eps:
        return None
    target_probs = target_values / target_sum
    observed_probs = observed_values / observed_sum
    return float(0.5 * np.abs(target_probs - observed_probs).sum())


def _alignment_distances(
    feature_names: Sequence[str],
    feature_loss_weights: Optional[Dict[str, float]],
    observed_values: Sequence[float],
) -> Optional[Dict[str, Optional[float]]]:
    target_values = _feature_weight_vector(feature_names, feature_loss_weights)
    if target_values is None:
        return None
    observed_array = np.asarray([
        max(0.0, float(value))
        for value in observed_values
    ], dtype=float)
    if observed_array.shape != target_values.shape:
        raise ValueError(
            f"Observed feature values have shape {observed_array.shape}, "
            f"expected {target_values.shape}."
        )
    return {
        "rank": _normalized_spearman_footrule(target_values, observed_array),
        "weight": _total_variation_distance(target_values, observed_array),
    }


def _semantic_shap_importance(
    shap_array: np.ndarray,
    feature_names: Sequence[str],
    feature_groups: Dict[str, str],
    active_groups: Sequence[str],
) -> np.ndarray:
    if shap_array.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP array, got shape {shap_array.shape}.")
    if shap_array.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature count {shap_array.shape[1]} does not match "
            f"{len(feature_names)} feature names."
        )
    missing_features = [
        feature_name
        for feature_name in feature_names
        if feature_name not in feature_groups
    ]
    if missing_features:
        raise ValueError(
            f"Missing semantic SHAP groups for features: {missing_features[:10]}"
        )

    group_indices: Dict[str, List[int]] = {}
    for index, feature_name in enumerate(feature_names):
        group_indices.setdefault(feature_groups[feature_name], []).append(index)
    missing_groups = [
        group_name for group_name in active_groups if group_name not in group_indices
    ]
    if missing_groups:
        raise ValueError(
            f"Selected semantic SHAP groups have no model columns: {missing_groups[:10]}"
        )

    return np.asarray([
        np.abs(shap_array[:, group_indices[group_name]].sum(axis=1)).mean()
        for group_name in active_groups
    ], dtype=float)


def _format_distance(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _distance_pair_text(label: str, distances: Optional[Dict[str, Optional[float]]]) -> str:
    if distances is None:
        return f"{label}=N/A"
    return (
        f"{label}: "
        f"RankDist={_format_distance(distances['rank'])}, "
        f"WeightDist={_format_distance(distances['weight'])}"
    )


def plot_training_curves(
    dataset: str,
    ID_balanced_accuracy: float,
    OOD_MEAN_balanced_accuracy: float,
    OOD_WORST_balanced_accuracy: float,
    OOD_STD_balanced_accuracy: float,
    ID_macro_f1: float,
    OOD_MEAN_macro_f1: float,
    OOD_WORST_macro_f1: float,
    OOD_STD_macro_f1: float,
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[str],
    repeat_i: int, EPOCH: int, best_epoch_idx: int,
    train_losses: Dict[str, List[float]], val_losses: Dict[str, List[float]],
    train_balanced_accuracies: List[float], val_balanced_accuracies: List[float],
    train_macro_f1s: List[float], val_macro_f1s: List[float],
    train_grads: Dict[str, List[float]], date: str,
    output_dir: Path,
    plot_label: Optional[str] = None,
    feature_loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[Dict[str, Optional[float]]]]:
    import matplotlib.pyplot as plt

    plot_path = Path(output_dir) / "curve"
    plot_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, EPOCH + 1)
    best_epoch = best_epoch_idx + 1
    fig = plt.figure(figsize=(22, 14))
    active_feature_names = [
        FEATURE_INDEX[idx]
        for idx in sorted(FEATURE_INDEX)
        if idx not in REMOVED_FEATURE_INDICES
    ]

    def _history_values_at_best(suffix: str) -> Optional[List[float]]:
        values = []
        for feat_name in active_feature_names:
            history = train_grads.get(f"{feat_name}_{suffix}")
            if history is None or best_epoch_idx >= len(history):
                return None
            values.append(history[best_epoch_idx])
        return values

    grad_values = _history_values_at_best("grad_l2")
    weight_values = _history_values_at_best("weight_abs")
    grad_distances = (
        _alignment_distances(active_feature_names, feature_loss_weights, grad_values)
        if grad_values is not None else None
    )
    weight_distances = (
        _alignment_distances(active_feature_names, feature_loss_weights, weight_values)
        if weight_values is not None else None
    )
    distance_title = ""
    if feature_loss_weights:
        distance_title = (
            "\n"
            + _distance_pair_text("Grad", grad_distances)
            + " | "
            + _distance_pair_text("Weight", weight_distances)
        )

    fig.suptitle(
        "Balanced ACC "
        f"ID={ID_balanced_accuracy:.4f}, OOD MEAN={OOD_MEAN_balanced_accuracy:.4f}, "
        f"OOD WORST={OOD_WORST_balanced_accuracy:.4f}, "
        f"OOD STD={OOD_STD_balanced_accuracy:.4f} | "
        "Macro-F1 "
        f"ID={ID_macro_f1:.4f}, OOD MEAN={OOD_MEAN_macro_f1:.4f}, "
        f"OOD WORST={OOD_WORST_macro_f1:.4f}, OOD STD={OOD_STD_macro_f1:.4f}"
        + distance_title
    )
    max_feature_legend_items = 12

    def _legend_value(value: float) -> str:
        return f"{value:.3e}"

    def _plot_metric_group(ax, title: str, suffix: str, ylabel: str) -> None:
        ax.axvline(x=best_epoch, color='gray', linestyle='--')
        curves = []
        for feat_name in active_feature_names:
            grad_name = f"{feat_name}_{suffix}"
            if grad_name not in train_grads:
                continue
            grad_list = train_grads[grad_name]
            curves.append((feat_name, grad_list))

        legend_features = {
            feat_name
            for feat_name, grad_list in sorted(
                curves,
                key=lambda item: item[1][best_epoch_idx],
                reverse=True,
            )[:max_feature_legend_items]
        }
        for feat_name, grad_list in curves:
            show_legend = feat_name in legend_features
            ax.plot(
                epochs,
                grad_list,
                linewidth=0.9,
                alpha=0.75 if show_legend else 0.28,
                label=(
                    f"{feat_name} "
                    f"(best={_legend_value(grad_list[best_epoch_idx])}, "
                    f"last={_legend_value(grad_list[-1])})"
                    if show_legend else "_nolegend_"
                )
            )
        ax.set_xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if curves:
            legend = ax.legend(fontsize=7, title=f"Top {min(len(curves), max_feature_legend_items)}")
            if legend is not None:
                legend.get_title().set_fontsize(7)
            if len(curves) > max_feature_legend_items:
                ax.text(
                    0.01,
                    0.01,
                    f"{len(curves) - max_feature_legend_items} more curves hidden from legend",
                    fontsize=7,
                    transform=ax.transAxes,
                )
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

    plt.subplot(2,4,1)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    for loss_name, loss_list in train_losses.items():
        plt.plot(
            epochs,
            loss_list,
            label=(
                f"{loss_name} "
                f"(best={_legend_value(loss_list[best_epoch_idx])}, "
                f"last={_legend_value(loss_list[-1])})"
            ),
        )
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss')
    plt.legend()

    plt.subplot(2,4,2)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    for loss_name, loss_list in val_losses.items():
        plt.plot(
            epochs,
            loss_list,
            label=(
                f"{loss_name} "
                f"(best={_legend_value(loss_list[best_epoch_idx])}, "
                f"last={_legend_value(loss_list[-1])})"
            ),
        )
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Valid Loss')
    plt.title('Valid Loss')
    plt.legend()

    plt.subplot(2,4,3)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    plt.plot(
        epochs,
        train_balanced_accuracies,
        label=(
            "Train Balanced Acc "
            f"(best={_legend_value(train_balanced_accuracies[best_epoch_idx])}, "
            f"last={_legend_value(train_balanced_accuracies[-1])})"
        ),
    )
    plt.plot(
        epochs,
        val_balanced_accuracies,
        label=(
            "Validation Balanced Acc "
            f"(best={_legend_value(val_balanced_accuracies[best_epoch_idx])}, "
            f"last={_legend_value(val_balanced_accuracies[-1])})"
        ),
    )
    plt.plot(
        epochs,
        train_macro_f1s,
        linestyle=':',
        label=(
            "Train Macro-F1 "
            f"(best={_legend_value(train_macro_f1s[best_epoch_idx])}, "
            f"last={_legend_value(train_macro_f1s[-1])})"
        ),
    )
    plt.plot(
        epochs,
        val_macro_f1s,
        linestyle=':',
        label=(
            "Validation Macro-F1 "
            f"(best={_legend_value(val_macro_f1s[best_epoch_idx])}, "
            f"last={_legend_value(val_macro_f1s[-1])})"
        ),
    )
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Balanced Accuracy / Macro-F1')
    plt.legend()

    plt.subplot(2,4,4)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    summary_terms = [
        "total_grad_l2",
        "total_weight_abs",
    ]
    has_summary = False
    for grad_name in summary_terms:
        if grad_name not in train_grads:
            continue
        grad_list = train_grads[grad_name]
        plt.plot(
            epochs,
            grad_list,
            label=(
                f"{grad_name} "
                f"(best={_legend_value(grad_list[best_epoch_idx])}, "
                f"last={_legend_value(grad_list[-1])})"
            ),
        )
        has_summary = True
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Summary')
    plt.title('Summary Terms')
    if has_summary:
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, "N/A", ha="center", va="center", transform=plt.gca().transAxes)

    _plot_metric_group(plt.subplot(2,4,5), "Feature Grad L2", "grad_l2", "Grad L2")
    _plot_metric_group(plt.subplot(2,4,6), "Feature Grad Prob", "grad_prob", "Grad Prob")
    _plot_metric_group(plt.subplot(2,4,7), "Feature Weight Abs", "weight_abs", "Weight Abs")
    _plot_metric_group(plt.subplot(2,4,8), "Feature Weight Prob", "weight_prob", "Weight Prob")

    plt.tight_layout()
    rmfeature_names = ", ".join([FEATURE_INDEX[feat_i] for feat_i in REMOVED_FEATURE_INDICES])
    plot_name = "All features" if len(REMOVED_FEATURE_INDICES) == 0 else f"Remove {rmfeature_names}"
    if plot_label:
        safe_label = "".join(char if char.isalnum() or char in "._-" else "_" for char in plot_label)
        plot_name = f"{safe_label} - {plot_name}"
    plt.savefig(plot_path / f"{plot_name} (repeat={repeat_i + 1}, datetime={date}).png")
    plt.close()
    return {
        "grad": grad_distances,
        "weight": weight_distances,
    }


def plot_accdelta_bars(
    dataset: str, rmfeature_accdelta: Dict[str, Dict[str, float]],
    ID_base: float, OOD_MEAN_base: float, OOD_WORST_base: float, REPEAT: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plot_path = Path(output_dir) / "accdelta"
    plot_path.mkdir(parents=True, exist_ok=True)
    if not rmfeature_accdelta:
        return

    # Rank removed features by OOD WORST (best-to-worst from left to right).
    sorted_items = sorted(
        rmfeature_accdelta.items(),
        key=lambda x: x[1].get("OOD WORST", 0.0),
        reverse=True
    )

    features = [feat for feat, _ in sorted_items]
    id_vals = [vals.get("ID", 0.0) for _, vals in sorted_items]
    ood_mean_vals = [vals.get("OOD MEAN", 0.0) for _, vals in sorted_items]
    ood_worst_vals = [vals.get("OOD WORST", 0.0) for _, vals in sorted_items]

    x = np.arange(len(features))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_id = ax.bar(x - width, id_vals, width=width, label=f"ID (base acc = {ID_base:.4f})")
    bars_ood_mean = ax.bar(x, ood_mean_vals, width=width, label=f"OOD MEAN (base acc = {OOD_MEAN_base:.4f})")
    bars_ood_worst = ax.bar(x + width, ood_worst_vals, width=width, label=f"OOD WORST (base acc = {OOD_WORST_base:.4f})")

    ax.axhline(0.0, color="C0", linewidth=1.2)
    ax.set_title("Relative Accuracy vs. Removed Feature")
    ax.set_xlabel("Removed Feature")
    ax.set_ylabel("Δ Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right")
    ax.legend(loc="upper right")

    all_vals = id_vals + ood_mean_vals + ood_worst_vals
    max_abs = max(abs(v) for v in all_vals) if all_vals else 0.0
    y_pad = max(0.0004, max_abs * 0.06)

    def _annotate_bars(bar_container):
        for bar in bar_container:
            h = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = h + y_pad if h >= 0 else h - y_pad
            va = "bottom" if h >= 0 else "top"
            ax.text(xpos, ypos, f"{h:+.4f}", ha="center", va=va, fontsize=12)

    _annotate_bars(bars_id)
    _annotate_bars(bars_ood_mean)
    _annotate_bars(bars_ood_worst)

    ax.set_ylim(min(all_vals) - 3 * y_pad, max(all_vals) + 3 * y_pad)
    fig.tight_layout()
    fig.savefig(plot_path / f"accdelta_bars_repeat{REPEAT}.png", dpi=200)
    plt.close(fig)


def plot_shap_values(
    dataset: str, TRAIN_VAL_GROUP: str, TEST_GROUPS: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int],
    shap_values: List[np.ndarray], repeat_i: int, date: str,
    output_dir: Path,
    plot_label: Optional[str] = None,
    feature_loss_weights: Optional[Dict[str, float]] = None,
    alignment_feature_groups: Optional[Dict[str, str]] = None,
    alignment_group_weights: Optional[Dict[str, float]] = None,
    alignment_active_groups: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt

    plot_path = Path(output_dir) / "shap"
    plot_path.mkdir(parents=True, exist_ok=True)

    active_feature_names = [
        FEATURE_INDEX[idx]
        for idx in sorted(FEATURE_INDEX)
        if idx not in REMOVED_FEATURE_INDICES
    ]
    alignment_options = (
        alignment_feature_groups,
        alignment_group_weights,
        alignment_active_groups,
    )
    if any(option is not None for option in alignment_options) and not all(
        option is not None for option in alignment_options
    ):
        raise ValueError(
            "Semantic SHAP alignment requires feature groups, group weights, "
            "and active groups together."
        )
    semantic_alignment = alignment_active_groups is not None
    if semantic_alignment:
        distance_feature_names = list(alignment_active_groups)
        if not distance_feature_names:
            raise ValueError("Semantic SHAP alignment requires at least one active group.")
        if len(set(distance_feature_names)) != len(distance_feature_names):
            raise ValueError("Semantic SHAP active groups must be unique.")
        missing_weights = [
            group_name
            for group_name in distance_feature_names
            if group_name not in alignment_group_weights
        ]
        if missing_weights:
            raise ValueError(
                f"Missing semantic SHAP weights for groups: {missing_weights[:10]}"
            )
        distance_weights = alignment_group_weights
    else:
        distance_feature_names = active_feature_names
        distance_weights = feature_loss_weights
    states = [TRAIN_VAL_GROUP] + TEST_GROUPS[:max(len(shap_values) - 1, 0)]
    max_shap_features = 30
    safe_label = None
    if plot_label:
        safe_label = "".join(char if char.isalnum() or char in "._-" else "_" for char in plot_label)

    shap_distance_records = []
    for state, shap_value in zip(states, shap_values):
        shap_array = np.asarray(shap_value)
        if shap_array.ndim == 3:
            # For binary classification, use class 1 attribution by default.
            class_idx = 1 if shap_array.shape[-1] > 1 else 0
            shap_array = shap_array[:, :, class_idx]
        elif shap_array.ndim != 2:
            raise ValueError(
                f"Expected SHAP values with 2 or 3 dims, got shape {shap_array.shape} for state {state}."
            )

        if shap_array.shape[1] != len(active_feature_names):
            raise ValueError(
                f"Feature count mismatch for state {state}: "
                f"{shap_array.shape[1]} SHAP columns vs {len(active_feature_names)} feature names."
            )

        if semantic_alignment:
            mean_abs_shap = _semantic_shap_importance(
                shap_array,
                active_feature_names,
                alignment_feature_groups,
                distance_feature_names,
            )
        else:
            mean_abs_shap = np.abs(shap_array).mean(axis=0)
        shap_distances = _alignment_distances(
            distance_feature_names,
            distance_weights,
            mean_abs_shap,
        ) if distance_weights else None
        distance_record = {
            "group": state,
            "rank": None if shap_distances is None else shap_distances["rank"],
            "weight": None if shap_distances is None else shap_distances["weight"],
        }
        if semantic_alignment:
            distance_record.update({
                "feature_space": "selected_semantic",
                "feature_count": len(distance_feature_names),
            })
        shap_distance_records.append(distance_record)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        shown_indices = sorted_indices[:max_shap_features]
        sorted_features = [distance_feature_names[idx] for idx in shown_indices]
        sorted_values = mean_abs_shap[shown_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(sorted_features))))
        y_pos = np.arange(len(sorted_features))
        bars = ax.barh(y_pos, sorted_values, color="C0")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Mean |semantic SHAP value|" if semantic_alignment else "Mean |SHAP value|"
        )
        ax.set_ylabel("Semantic feature" if semantic_alignment else "Feature")
        title_prefix = "Selected Semantic SHAP" if semantic_alignment else "SHAP"
        title = f"{title_prefix} Feature Importance ({state})"
        if distance_weights:
            title += "\n" + _distance_pair_text(title_prefix, shap_distances)
        ax.set_title(title)
        if len(distance_feature_names) > max_shap_features:
            ax.text(
                0.01,
                0.01,
                f"Top {max_shap_features} of {len(distance_feature_names)} features shown",
                fontsize=8,
                transform=ax.transAxes,
            )

        max_val = sorted_values.max() if len(sorted_values) > 0 else 0.0
        x_pad = max(1e-4, max_val * 0.02)
        for bar, value in zip(bars, sorted_values):
            ax.text(
                value + x_pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3e}",
                va="center",
                fontsize=10,
            )

        ax.set_xlim(0.0, max_val + 6 * x_pad if max_val > 0 else 1.0)
        fig.tight_layout()
        plot_name = f"shap_{state}"
        if safe_label:
            plot_name = f"{safe_label} - {plot_name}"
        fig.savefig(plot_path / f"{plot_name} (repeat={repeat_i + 1}, datetime={date}).png", dpi=200)
        plt.close(fig)
    return shap_distance_records
