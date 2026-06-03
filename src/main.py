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
    FeatureGradCELoss,
    FeatureImportanceTargetCELoss,
    FirstLayerWeightCELoss,
    LLMAttributionAlignedCELoss,
)
from src.trainer import Trainer
from src.data_cache import load_or_build_dataset_cache
from src.utils import (
    ExperimentMetrics,
    MetricSummary,
    experiment_metrics_to_dict,
    save_json_result,
    set_seeds, 
    plot_training_curves, plot_shap_values
)
from data.acs.dataset import ACS_TASKS, acs_task_load_data, acsincome_load_data_acc_upperbound_test
from data.synthetic_ood.dataset import synthetic_ood_load_data

data_loading_wrapper = {
    dataset: partial(acs_task_load_data, dataset)
    for dataset in ACS_TASKS
}
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
    if loss_name == "feature_grad_ce":
        return FeatureGradCELoss(**common_kwargs, **criterion_kwargs)
    if loss_name == "first_layer_weight_ce":
        return FirstLayerWeightCELoss(**common_kwargs, **criterion_kwargs)
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

    feature_std = train.X.std(dim=0, unbiased=False).clamp_min(1e-6)
    resolved_kwargs["importance_scale"] = feature_std.tolist()
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


def main(
    dataset: str, TRAIN_VAL_GROUP: str, TEST_GROUPS: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int], FEATURE_LOSS_WEIGHTS: Dict[str, float],
    TRAIN_BATCH: int=256, EVAL_BATCH: int=1024, VAL_RATE: float=0.2, LR: int=1e-4, REG_SCALE: int=1.0,
    PATIENCE: int=50, REPEAT: int=10, MAX_EPOCHS: int=5000,
    DATASET_CONFIG: Optional[Any]=None, PLOT_CURVE: bool=True, PLOT_SHAP: bool=True, PLOT_TEST_SHAP: bool=False,
    MODEL_NAME: str="mlp", HIDDEN_SIZE: int=64,
    LOSS_NAME: str="feature_grad_ce", LOSS_KWARGS: Optional[Dict[str, Any]]=None,
    MODEL_SEEDS: Optional[List[int]]=None,
    device: str="cuda",
    RESULT_PATH: Optional[Path]=None,
    RESULT_METADATA: Optional[Dict[str, Any]]=None,
    BENCHMARK: str="acs",
    DATASET_ARTIFACT_NAME: Optional[str]=None,
    SHOW_PROGRESS: bool=False,
    VERBOSE: bool=False,
) -> ExperimentMetrics:
    artifact_dataset = DATASET_ARTIFACT_NAME or dataset
    removed_feature_indices = set(REMOVED_FEATURE_INDICES)
    kept_feature_indices = [
        idx for idx in sorted(FEATURE_INDEX)
        if idx not in removed_feature_indices
    ]
    if not kept_feature_indices:
        metrics = tuple((0.0, 0.0) for _ in range(8))
        if RESULT_PATH is not None:
            save_json_result(RESULT_PATH, {
                "status": "all_features_removed",
                "completed_at": datetime.datetime.now(),
                "metadata": RESULT_METADATA or {},
                "benchmark": BENCHMARK,
                "dataset": dataset,
                "artifact_dataset": artifact_dataset,
                "train_val_group": TRAIN_VAL_GROUP,
                "test_groups": TEST_GROUPS,
                "feature_index": FEATURE_INDEX,
                "removed_feature_indices": sorted(removed_feature_indices),
                "feature_loss_weights": FEATURE_LOSS_WEIGHTS,
                "metrics": experiment_metrics_to_dict(metrics),
                "repeats": [],
                "message": "No input features remain; recording zero ACC/F1 metrics.",
            })
        return metrics

    # a fixed seed for data split
    set_seeds(67)

    # dataset
    def _build_dataset():
        return data_loading_wrapper[dataset](
            REMOVED_FEATURE_INDICES,
            TRAIN_VAL_GROUP,
            TEST_GROUPS,
            VAL_RATE,
            DATASET_CONFIG,
        )

    train, val, tests, data_dir, cache_hit = load_or_build_dataset_cache(
        BENCHMARK,
        artifact_dataset,
        REMOVED_FEATURE_INDICES,
        DATASET_CONFIG,
        _build_dataset,
    )
    if VERBOSE:
        print({"data_cache_dir": str(data_dir), "cache_hit": cache_hit})
    
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
    ID_accs = []
    OOD_MEAN_accs = []
    OOD_WORST_accs = []
    OOD_STD_accs = []
    ID_f1s = []
    OOD_MEAN_f1s = []
    OOD_WORST_f1s = []
    OOD_STD_f1s = []
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
            model, criterion, optimizer, SHOW_PROGRESS=SHOW_PROGRESS
        )
        (
            epoch,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            train_f1s,
            val_f1s,
            test_state_accs,
            test_state_f1s,
            train_grad_terms_sum,
            shape_values,
        ) = trainer.run_training(repeat_i)
        best_epoch_idx = int(np.argmax(val_accs))
        test_state_accs_np = np.array(test_state_accs)
        test_state_f1s_np = np.array(test_state_f1s)
        ID_acc = val_accs[best_epoch_idx]
        OOD_MEAN_acc = test_state_accs_np.mean()
        OOD_WORST_acc = test_state_accs_np.min()
        OOD_STD_acc = test_state_accs_np.std()
        ID_f1 = val_f1s[best_epoch_idx]
        OOD_MEAN_f1 = test_state_f1s_np.mean()
        OOD_WORST_f1 = test_state_f1s_np.min()
        OOD_STD_f1 = test_state_f1s_np.std()
        
        # check convergence & SHAP
        if repeat_i < 3:
            date = datetime.datetime.now()
            if PLOT_CURVE:
                plot_training_curves(
                    dataset,
                    ID_acc, OOD_MEAN_acc, OOD_WORST_acc, OOD_STD_acc,
                    ID_f1, OOD_MEAN_f1, OOD_WORST_f1, OOD_STD_f1,
                    FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                    repeat_i, epoch, PATIENCE,
                    train_losses, val_losses, train_accs, val_accs,
                    train_f1s, val_f1s,
                    train_grad_terms_sum, date,
                    benchmark=BENCHMARK,
                    artifact_dataset=artifact_dataset,
                )
            if PLOT_SHAP:
                plot_shap_values(
                    dataset, TRAIN_VAL_GROUP, TEST_GROUPS,
                    FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                    shape_values, repeat_i, date,
                    benchmark=BENCHMARK,
                    artifact_dataset=artifact_dataset,
                )
        
        # logging
        ID_accs.append(ID_acc)
        OOD_MEAN_accs.append(OOD_MEAN_acc)
        OOD_WORST_accs.append(OOD_WORST_acc)
        OOD_STD_accs.append(OOD_STD_acc)
        ID_f1s.append(ID_f1)
        OOD_MEAN_f1s.append(OOD_MEAN_f1)
        OOD_WORST_f1s.append(OOD_WORST_f1)
        OOD_STD_f1s.append(OOD_STD_f1)
        repeat_details.append({
            "repeat": repeat_i + 1,
            "model_seed": model_seed,
            "epochs": epoch,
            "best_epoch": best_epoch_idx + 1,
            "accuracy": {
                "id": float(ID_acc),
                "ood_mean": float(OOD_MEAN_acc),
                "ood_worst": float(OOD_WORST_acc),
                "ood_std": float(OOD_STD_acc),
                "test_states": {
                    str(state): float(value)
                    for state, value in zip(TEST_GROUPS, test_state_accs)
                },
            },
            "f1": {
                "id": float(ID_f1),
                "ood_mean": float(OOD_MEAN_f1),
                "ood_worst": float(OOD_WORST_f1),
                "ood_std": float(OOD_STD_f1),
                "test_states": {
                    str(state): float(value)
                    for state, value in zip(TEST_GROUPS, test_state_f1s)
                },
            },
        })
        if VERBOSE:
            print(
                f"ACC: {ID_acc:.4f}, {OOD_MEAN_acc:.4f}, {OOD_WORST_acc:.4f}, {OOD_STD_acc:.4f}; "
                f"F1: {ID_f1:.4f}, {OOD_MEAN_f1:.4f}, {OOD_WORST_f1:.4f}, {OOD_STD_f1:.4f}"
            )

    metrics = (
        _mean_confidence_interval(ID_accs),
        _mean_confidence_interval(OOD_MEAN_accs),
        _mean_confidence_interval(OOD_WORST_accs),
        _mean_confidence_interval(OOD_STD_accs),
        _mean_confidence_interval(ID_f1s),
        _mean_confidence_interval(OOD_MEAN_f1s),
        _mean_confidence_interval(OOD_WORST_f1s),
        _mean_confidence_interval(OOD_STD_f1s),
    )

    if RESULT_PATH is not None:
        save_json_result(RESULT_PATH, {
            "status": "ok",
            "completed_at": datetime.datetime.now(),
            "metadata": RESULT_METADATA or {},
            "benchmark": BENCHMARK,
            "dataset": dataset,
            "artifact_dataset": artifact_dataset,
            "train_val_group": TRAIN_VAL_GROUP,
            "test_groups": TEST_GROUPS,
            "data_cache_dir": data_dir,
            "data_cache_hit": cache_hit,
            "feature_index": FEATURE_INDEX,
            "removed_feature_indices": REMOVED_FEATURE_INDICES,
            "feature_loss_weights": FEATURE_LOSS_WEIGHTS,
            "dataset_config": DATASET_CONFIG,
            "model": {
                "name": MODEL_NAME,
                "hidden_size": HIDDEN_SIZE,
                "seeds": MODEL_SEEDS,
            },
            "training": {
                "device": device,
                "train_batch": TRAIN_BATCH,
                "eval_batch": EVAL_BATCH,
                "val_rate": VAL_RATE,
                "lr": LR,
                "patience": PATIENCE,
                "repeat": REPEAT,
                "max_epochs": MAX_EPOCHS,
            },
            "loss": {
                "name": LOSS_NAME,
                "reg_scale": REG_SCALE,
                "kwargs": LOSS_KWARGS or {},
                "resolved_kwargs": resolved_loss_kwargs,
            },
            "metrics": experiment_metrics_to_dict(metrics),
            "repeats": repeat_details,
        })

    return metrics
