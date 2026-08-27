import datetime
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional
from scipy.stats import t as student_t
from src.mlp import MLP, Linear
from src.loss import (
    CrossEntropyCELoss,
    GradientRegularizedCELoss,
    FeatureImportanceTargetCELoss,
    LLMAttributionAlignedCELoss,
)
from src.trainer import Trainer
from src.data_cache import load_or_build_dataset_cache
from src.utils import (
    ExperimentMetrics,
    MetricSummary,
    experiment_metrics_to_dict,
    metric_summary_to_dict,
    report_metrics_from_values,
    save_json_result,
    set_seeds,
    plot_training_curves, plot_shap_values,
)
from data.acs.dataset import ACS_TASKS, acs_task_load_data, acsincome_load_data_acc_upperbound_test
from data.synthetic_ood.dataset import synthetic_ood_load_data
from data.tableshift.dataset import TABLESHIFT_TASKS, tableshift_load_data
from data.whyshift.dataset import WHYSHIFT_TASKS, whyshift_load_data

data_loading_wrapper = {
    dataset: partial(acs_task_load_data, dataset)
    for dataset in ACS_TASKS
}
data_loading_wrapper.update({
    dataset: partial(whyshift_load_data, dataset)
    for dataset in WHYSHIFT_TASKS
})
data_loading_wrapper.update({
    dataset: partial(tableshift_load_data, dataset)
    for dataset in TABLESHIFT_TASKS
})
data_loading_wrapper.update({
    "acsincome_acc_upperbound_test": acsincome_load_data_acc_upperbound_test,
    "synthetic_ood": synthetic_ood_load_data,
})

def _build_model(
    model_name: str,
    num_features: int,
    hidden_size: int,
) -> nn.Module:
    if model_name == "mlp":
        return MLP(
            num_features=num_features,
            hidden_size=hidden_size,
            num_classes=2,
        )
    if model_name == "linear":
        return Linear(
            num_features=num_features,
            num_classes=2,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _build_criterion(
    loss_name: str,
    FEATURE_INDEX: Dict[int, str],
    REMOVED_FEATURE_INDICES: List[int],
    FEATURE_LOSS_WEIGHTS: Dict[str, float],
    REG_SCALE: float,
    device: str,
    loss_kwargs: Optional[Dict[str, Any]],
) -> nn.Module:
    criterion_kwargs = dict(loss_kwargs or {})
    common_kwargs = dict(
        FEATURE_INDEX=FEATURE_INDEX,
        REMOVED_FEATURE_INDICES=REMOVED_FEATURE_INDICES,
        FEATURE_LOSS_WEIGHTS=FEATURE_LOSS_WEIGHTS,
        reg_scale=REG_SCALE,
        device=device,
    )

    if loss_name == "cross_entropy":
        return CrossEntropyCELoss(**criterion_kwargs)
    if loss_name == "gradient_regularized_ce":
        return GradientRegularizedCELoss(**common_kwargs, **criterion_kwargs)
    if loss_name == "llm_attribution_aligned_ce":
        return LLMAttributionAlignedCELoss(**common_kwargs, **criterion_kwargs)
    if loss_name == "feature_importance_target_ce":
        return FeatureImportanceTargetCELoss(**common_kwargs, **criterion_kwargs)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def _resolve_loss_kwargs_from_train(
    loss_kwargs: Optional[Dict[str, Any]],
    train,
    verbose: bool = False,
) -> Dict[str, Any]:
    resolved_kwargs = dict(loss_kwargs or {})
    if resolved_kwargs.get("importance_scale") != "train_std":
        return resolved_kwargs

    feature_scale = train.X.std(dim=0, unbiased=False)
    resolved_kwargs["importance_scale"] = feature_scale.clamp_min(1e-6).tolist()
    if verbose:
        print({
            "resolved_importance_scale": [
                round(float(value), 6)
                for value in resolved_kwargs["importance_scale"]
            ]
        })
    return resolved_kwargs


def _mean_confidence_interval(values: List[float], confidence: float = 0.95) -> MetricSummary:
    values_np = np.asarray(values, dtype=float)
    if values_np.size == 0:
        return 0.0, 0.0

    mean = float(values_np.mean())
    if values_np.size <= 1:
        return mean, 0.0

    standard_error = values_np.std(ddof=1) / np.sqrt(values_np.size)
    if standard_error == 0.0:
        return mean, 0.0

    interval_range = student_t.ppf((1.0 + confidence) / 2.0, values_np.size - 1) * standard_error
    return mean, float(interval_range)


def _result_feature_record(
    feature_names: List[str],
    removed_feature_indices: List[int],
    data_removed_feature_indices: List[int],
    feature_loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "names": feature_names,
        "removed_indices": sorted(removed_feature_indices),
        "data_removed_indices": sorted(data_removed_feature_indices),
    }
    if feature_loss_weights:
        record["importance"] = [
            float(feature_loss_weights[feature_name])
            for feature_name in feature_names
        ]
    return record


def _persisted_loss_kwargs(loss_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    derived_keys = {
        "gradient_feature_groups",
        "gradient_group_weights",
        "gradient_active_groups",
    }
    return {
        key: value
        for key, value in (loss_kwargs or {}).items()
        if key not in derived_keys and value is not None
    }


def main(
    dataset: str, TRAIN_VAL_GROUP: str, TEST_GROUPS: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int], FEATURE_LOSS_WEIGHTS: Dict[str, float],
    TRAIN_BATCH: int=256, EVAL_BATCH: int=1024, VAL_RATE: float=0.2, LR: int=1e-4, REG_SCALE: int=1.0,
    REG_WARMUP_EPOCHS: int=0, PATIENCE: int=50, REPEAT: int=10, MAX_EPOCHS: int=5000,
    DATASET_CONFIG: Optional[Any]=None, PLOT_CURVE: bool=True, PLOT_SHAP: bool=True, PLOT_TEST_SHAP: bool=False,
    DATA_REMOVED_FEATURE_INDICES: Optional[List[int]]=None,
    SHAP_SAMPLE_SIZE: int=500,
    MODEL_NAME: str="mlp", HIDDEN_SIZE: int=64,
    LOSS_NAME: str="cross_entropy", LOSS_KWARGS: Optional[Dict[str, Any]]=None,
    MODEL_SEEDS: Optional[List[int]]=None,
    device: str="cuda",
    RESULT_PATH: Optional[Path]=None,
    RESULT_METADATA: Optional[Dict[str, Any]]=None,
    BENCHMARK: str="acs",
    DATASET_ARTIFACT_NAME: Optional[str]=None,
    REPORT_ID_GROUPS: Optional[List[str]]=None,
    REPORT_OOD_GROUPS: Optional[List[str]]=None,
    SHOW_PROGRESS: bool=False,
    RECORD_BEST_GRAD_L2: bool=False,
    SHAP_ALIGNMENT_FEATURE_GROUPS: Optional[Dict[str, str]]=None,
    SHAP_ALIGNMENT_GROUP_WEIGHTS: Optional[Dict[str, float]]=None,
    SHAP_ALIGNMENT_ACTIVE_GROUPS: Optional[List[str]]=None,
    VERBOSE: bool=False,
) -> ExperimentMetrics:
    artifact_dataset = DATASET_ARTIFACT_NAME or dataset
    removed_feature_indices = set(REMOVED_FEATURE_INDICES)
    data_removed_feature_indices = list(
        REMOVED_FEATURE_INDICES
        if DATA_REMOVED_FEATURE_INDICES is None
        else DATA_REMOVED_FEATURE_INDICES
    )
    report_id_groups = list(REPORT_ID_GROUPS or [])
    report_ood_groups = list(REPORT_OOD_GROUPS or [])
    plot_output_dir = Path(RESULT_PATH).parent if RESULT_PATH is not None else None
    if (PLOT_CURVE or PLOT_SHAP) and plot_output_dir is None:
        raise ValueError("RESULT_PATH is required when curve or SHAP plotting is enabled.")

    # a fixed seed for data split
    set_seeds(67)

    # dataset
    def _build_dataset():
        return data_loading_wrapper[dataset](
            data_removed_feature_indices,
            TRAIN_VAL_GROUP,
            TEST_GROUPS,
            VAL_RATE,
            DATASET_CONFIG,
        )

    train, val, tests, data_dir, cache_hit = load_or_build_dataset_cache(
        BENCHMARK,
        artifact_dataset,
        data_removed_feature_indices,
        DATASET_CONFIG,
        _build_dataset,
    )
    if VERBOSE:
        print({"data_cache_dir": str(data_dir), "cache_hit": cache_hit})

    if not FEATURE_INDEX:
        if REMOVED_FEATURE_INDICES:
            raise ValueError("FEATURE_INDEX is required when REMOVED_FEATURE_INDICES is non-empty.")
        feature_names = getattr(train, "feature_names", None)
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(train.X.shape[1])]
        FEATURE_INDEX = {idx: str(name) for idx, name in enumerate(feature_names)}

    kept_feature_indices = [
        idx for idx in sorted(FEATURE_INDEX)
        if idx not in removed_feature_indices
    ]
    kept_feature_names = [FEATURE_INDEX[idx] for idx in kept_feature_indices]
    if not kept_feature_indices:
        metrics = tuple((0.0, 0.0) for _ in range(8))
        if RESULT_PATH is not None:
            save_json_result(RESULT_PATH, {
                "schema_version": 2,
                "status": "all_features_removed",
                "completed_at": datetime.datetime.now(),
                "metadata": RESULT_METADATA or {},
                "benchmark": BENCHMARK,
                "dataset": dataset,
                "artifact_dataset": artifact_dataset,
                "train_val_group": TRAIN_VAL_GROUP,
                "test_groups": TEST_GROUPS,
                "report_id_groups": report_id_groups,
                "report_ood_groups": report_ood_groups or TEST_GROUPS,
                "data_cache_key": data_dir.name,
                "data_cache_hit": cache_hit,
                "features": _result_feature_record(
                    [],
                    list(removed_feature_indices),
                    data_removed_feature_indices,
                    {},
                ),
                "metrics": experiment_metrics_to_dict(metrics),
                "selection_metrics": {
                    "balanced_accuracy": metric_summary_to_dict((0.0, 0.0)),
                    "macro_f1": metric_summary_to_dict((0.0, 0.0)),
                },
                "repeats": [],
                "message": "No input features remain; recording zero metrics.",
            })
        return metrics
    
    train_loader = DataLoader(train, batch_size=TRAIN_BATCH, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True)
    test_loaders = [DataLoader(test, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True) for test in tests]

    # loss
    resolved_loss_kwargs = _resolve_loss_kwargs_from_train(LOSS_KWARGS, train, verbose=VERBOSE)
    criterion = _build_criterion(
        LOSS_NAME,
        FEATURE_INDEX,
        REMOVED_FEATURE_INDICES,
        FEATURE_LOSS_WEIGHTS,
        REG_SCALE,
        device,
        resolved_loss_kwargs,
    )

    # repeat experiments
    ID_balanced_accuracies = []
    OOD_MEAN_balanced_accuracies = []
    OOD_WORST_balanced_accuracies = []
    OOD_STD_balanced_accuracies = []
    ID_macro_f1s = []
    OOD_MEAN_macro_f1s = []
    OOD_WORST_macro_f1s = []
    OOD_STD_macro_f1s = []
    selection_balanced_accuracies = []
    selection_macro_f1s = []
    selection_scores = []
    repeat_details = []
    for repeat_i in range(REPEAT):
        # random seeds for model initialization
        model_seed = MODEL_SEEDS[repeat_i] if MODEL_SEEDS is not None else random.randint(0, 100000)
        set_seeds(model_seed)
        if VERBOSE:
            print(f"repeat {repeat_i + 1} model_seed={model_seed}")

        # model, optimizer
        model = _build_model(
            MODEL_NAME,
            num_features=len(kept_feature_indices),
            hidden_size=HIDDEN_SIZE,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # training loop
        trainer = Trainer(
            device, PATIENCE, MAX_EPOCHS, PLOT_SHAP, PLOT_TEST_SHAP,
            train, val, tests, train_loader, val_loader, test_loaders,
            model, criterion, optimizer,
            REG_SCALE=REG_SCALE,
            REG_WARMUP_EPOCHS=REG_WARMUP_EPOCHS,
            SHAP_SAMPLE_SIZE=SHAP_SAMPLE_SIZE,
            SHOW_PROGRESS=SHOW_PROGRESS,
            TRACK_DIAGNOSTICS=PLOT_CURVE and repeat_i < 3,
            RECORD_BEST_GRAD_L2=RECORD_BEST_GRAD_L2,
            FEATURE_NAMES=kept_feature_names,
        )
        (
            epoch,
            train_losses,
            val_losses,
            train_balanced_accuracies,
            val_balanced_accuracies,
            train_macro_f1s,
            val_macro_f1s,
            test_state_balanced_accuracies,
            test_state_macro_f1s,
            train_grad_terms_sum,
            best_grad_l2,
            shape_values,
        ) = trainer.run_training(repeat_i)
        val_scores = (
            np.asarray(val_balanced_accuracies)
            + np.asarray(val_macro_f1s)
        )
        best_epoch_idx = int(np.argmax(val_scores))
        selection_balanced_accuracy = val_balanced_accuracies[best_epoch_idx]
        selection_macro_f1 = val_macro_f1s[best_epoch_idx]
        selection_score = val_scores[best_epoch_idx]
        report_metrics = report_metrics_from_values(
            selection_balanced_accuracy,
            selection_macro_f1,
            TEST_GROUPS,
            test_state_balanced_accuracies,
            test_state_macro_f1s,
            report_id_groups=report_id_groups,
            report_ood_groups=report_ood_groups,
        )
        ID_balanced_accuracy = report_metrics["balanced_accuracy"]["id"]
        OOD_MEAN_balanced_accuracy = report_metrics["balanced_accuracy"]["ood_mean"]
        OOD_WORST_balanced_accuracy = report_metrics["balanced_accuracy"]["ood_worst"]
        OOD_STD_balanced_accuracy = report_metrics["balanced_accuracy"]["ood_std"]
        ID_macro_f1 = report_metrics["macro_f1"]["id"]
        OOD_MEAN_macro_f1 = report_metrics["macro_f1"]["ood_mean"]
        OOD_WORST_macro_f1 = report_metrics["macro_f1"]["ood_worst"]
        OOD_STD_macro_f1 = report_metrics["macro_f1"]["ood_std"]
        
        # check convergence & SHAP
        curve_alignment_distances = None
        shap_alignment_distances = []
        if repeat_i < 3:
            date = datetime.datetime.now()
            if PLOT_CURVE:
                curve_alignment_distances = plot_training_curves(
                    dataset,
                    ID_balanced_accuracy,
                    OOD_MEAN_balanced_accuracy,
                    OOD_WORST_balanced_accuracy,
                    OOD_STD_balanced_accuracy,
                    ID_macro_f1,
                    OOD_MEAN_macro_f1,
                    OOD_WORST_macro_f1,
                    OOD_STD_macro_f1,
                    FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                    repeat_i, epoch, best_epoch_idx,
                    train_losses, val_losses,
                    train_balanced_accuracies, val_balanced_accuracies,
                    train_macro_f1s, val_macro_f1s,
                    train_grad_terms_sum, date,
                    output_dir=plot_output_dir,
                    plot_label=(RESULT_METADATA or {}).get("method"),
                    feature_loss_weights=FEATURE_LOSS_WEIGHTS,
                )
            if PLOT_SHAP:
                shap_alignment_distances = plot_shap_values(
                    dataset, TRAIN_VAL_GROUP, TEST_GROUPS,
                    FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                    shape_values, repeat_i, date,
                    output_dir=plot_output_dir,
                    plot_label=(RESULT_METADATA or {}).get("method"),
                    feature_loss_weights=FEATURE_LOSS_WEIGHTS,
                    alignment_feature_groups=SHAP_ALIGNMENT_FEATURE_GROUPS,
                    alignment_group_weights=SHAP_ALIGNMENT_GROUP_WEIGHTS,
                    alignment_active_groups=SHAP_ALIGNMENT_ACTIVE_GROUPS,
                )
        
        # logging
        ID_balanced_accuracies.append(ID_balanced_accuracy)
        OOD_MEAN_balanced_accuracies.append(OOD_MEAN_balanced_accuracy)
        OOD_WORST_balanced_accuracies.append(OOD_WORST_balanced_accuracy)
        OOD_STD_balanced_accuracies.append(OOD_STD_balanced_accuracy)
        ID_macro_f1s.append(ID_macro_f1)
        OOD_MEAN_macro_f1s.append(OOD_MEAN_macro_f1)
        OOD_WORST_macro_f1s.append(OOD_WORST_macro_f1)
        OOD_STD_macro_f1s.append(OOD_STD_macro_f1)
        selection_balanced_accuracies.append(selection_balanced_accuracy)
        selection_macro_f1s.append(selection_macro_f1)
        selection_scores.append(selection_score)
        repeat_detail = {
            "repeat": repeat_i + 1,
            "model_seed": model_seed,
            "epochs": epoch,
            "best_epoch": best_epoch_idx + 1,
            "effective_reg_scale": REG_SCALE * (
                min(best_epoch_idx / REG_WARMUP_EPOCHS, 1.0)
                if REG_WARMUP_EPOCHS
                else 1.0
            ),
            "selection": {
                "balanced_accuracy": float(selection_balanced_accuracy),
                "macro_f1": float(selection_macro_f1),
                "score": float(selection_score),
            },
            "balanced_accuracy": {
                "id": float(ID_balanced_accuracy),
                "ood_mean": float(OOD_MEAN_balanced_accuracy),
                "ood_worst": float(OOD_WORST_balanced_accuracy),
                "ood_std": float(OOD_STD_balanced_accuracy),
                "test_groups": {
                    str(group): float(value)
                    for group, value in zip(
                        TEST_GROUPS,
                        test_state_balanced_accuracies,
                    )
                },
            },
            "macro_f1": {
                "id": float(ID_macro_f1),
                "ood_mean": float(OOD_MEAN_macro_f1),
                "ood_worst": float(OOD_WORST_macro_f1),
                "ood_std": float(OOD_STD_macro_f1),
                "test_groups": {
                    str(group): float(value)
                    for group, value in zip(TEST_GROUPS, test_state_macro_f1s)
                },
            },
        }
        alignment_distances = {}
        if curve_alignment_distances is not None:
            alignment_distances["curve"] = curve_alignment_distances
        if shap_alignment_distances:
            alignment_distances["shap"] = shap_alignment_distances
        if alignment_distances:
            repeat_detail["alignment_distances"] = alignment_distances
        if best_grad_l2 is not None:
            if len(best_grad_l2) != len(kept_feature_names):
                raise ValueError(
                    "Best-checkpoint Grad-L2 feature count mismatch: "
                    f"{len(best_grad_l2)} values for {len(kept_feature_names)} features."
                )
            repeat_detail["best_grad_l2"] = [
                float(value) for value in best_grad_l2
            ]
        repeat_details.append(repeat_detail)
        if VERBOSE:
            print(
                "Balanced ACC: "
                f"{ID_balanced_accuracy:.4f}, {OOD_MEAN_balanced_accuracy:.4f}, "
                f"{OOD_WORST_balanced_accuracy:.4f}, {OOD_STD_balanced_accuracy:.4f}; "
                "Macro-F1: "
                f"{ID_macro_f1:.4f}, {OOD_MEAN_macro_f1:.4f}, "
                f"{OOD_WORST_macro_f1:.4f}, {OOD_STD_macro_f1:.4f}"
            )

    metrics = (
        _mean_confidence_interval(ID_balanced_accuracies),
        _mean_confidence_interval(OOD_MEAN_balanced_accuracies),
        _mean_confidence_interval(OOD_WORST_balanced_accuracies),
        _mean_confidence_interval(OOD_STD_balanced_accuracies),
        _mean_confidence_interval(ID_macro_f1s),
        _mean_confidence_interval(OOD_MEAN_macro_f1s),
        _mean_confidence_interval(OOD_WORST_macro_f1s),
        _mean_confidence_interval(OOD_STD_macro_f1s),
    )

    if RESULT_PATH is not None:
        save_json_result(RESULT_PATH, {
            "schema_version": 2,
            "status": "ok",
            "completed_at": datetime.datetime.now(),
            "metadata": RESULT_METADATA or {},
            "benchmark": BENCHMARK,
            "dataset": dataset,
            "artifact_dataset": artifact_dataset,
            "train_val_group": TRAIN_VAL_GROUP,
            "test_groups": TEST_GROUPS,
            "report_id_groups": report_id_groups,
            "report_ood_groups": report_ood_groups or TEST_GROUPS,
            "data_cache_key": data_dir.name,
            "data_cache_hit": cache_hit,
            "features": _result_feature_record(
                kept_feature_names,
                REMOVED_FEATURE_INDICES,
                data_removed_feature_indices,
                FEATURE_LOSS_WEIGHTS,
            ),
            "dataset_config": DATASET_CONFIG,
            "model": {
                "name": MODEL_NAME,
                "hidden_size": HIDDEN_SIZE,
            },
            "training": {
                "device": device,
                "train_batch": TRAIN_BATCH,
                "eval_batch": EVAL_BATCH,
                "val_rate": VAL_RATE,
                "lr": LR,
                "reg_warmup_epochs": REG_WARMUP_EPOCHS,
                "patience": PATIENCE,
                "repeat": REPEAT,
                "max_epochs": MAX_EPOCHS,
                "shap_sample_size": SHAP_SAMPLE_SIZE,
            },
            "loss": {
                "name": LOSS_NAME,
                "reg_scale": REG_SCALE,
                "kwargs": _persisted_loss_kwargs(LOSS_KWARGS),
            },
            "selection_metrics": {
                "balanced_accuracy": metric_summary_to_dict(
                    _mean_confidence_interval(selection_balanced_accuracies)
                ),
                "macro_f1": metric_summary_to_dict(
                    _mean_confidence_interval(selection_macro_f1s)
                ),
                "balanced_accuracy_plus_macro_f1": metric_summary_to_dict(
                    _mean_confidence_interval(selection_scores)
                ),
            },
            "metrics": experiment_metrics_to_dict(metrics),
            "repeats": repeat_details,
        })

    return metrics
