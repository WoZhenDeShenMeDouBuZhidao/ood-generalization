import datetime
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from typing import Any, Dict, List, Optional, Tuple
from src.mlp import MLP, Linear
from src.loss import FeatureGradCELoss, FirstLayerWeightCELoss, FeatureImportanceTargetCELoss
from src.trainer import Trainer
from src.utils import (
    set_seeds, 
    plot_training_curves, plot_shap_values
)
from acsincome.dataset import acsincome_load_train_val, acsincome_load_tests
from synthetic_ood.dataset import synthetic_ood_load_train_val, synthetic_ood_load_tests

data_loading_wrapper = {
    "acsincome": {
        "load_train_val": lambda removed_feature_indices, group, val_rate, dataset_config: acsincome_load_train_val(
            removed_feature_indices, group, val_rate
        ),
        "load_tests": lambda removed_feature_indices, groups, dataset_config: acsincome_load_tests(
            removed_feature_indices, groups
        ),
    },
    "synthetic_ood": {
        "load_train_val": synthetic_ood_load_train_val,
        "load_tests": synthetic_ood_load_tests,
    }
}


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

    if loss_name == "feature_grad_ce":
        return FeatureGradCELoss(**common_kwargs, **criterion_kwargs)
    if loss_name == "first_layer_weight_ce":
        return FirstLayerWeightCELoss(**common_kwargs, **criterion_kwargs)
    if loss_name == "feature_importance_target_ce":
        return FeatureImportanceTargetCELoss(**common_kwargs, **criterion_kwargs)
    raise ValueError(f"Unsupported loss_name: {loss_name}")

def main(
    dataset: str, TRAIN_VAL_GROUP: str, TEST_GROUPS: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int],
    FEATURE_LOSS_WEIGHTS: Dict[str, float],
    TRAIN_BATCH: int=256, EVAL_BATCH: int=1024,
    VAL_RATE: float=0.2, LR: int=1e-4, REG_SCALE: int=1.0,
    PATIENCE: int=50, REPEAT: int=10, MAX_EPOCHS: int=5000,
    DATASET_CONFIG: Optional[Any]=None, PLOT_TEST_SHAP: bool=False,
    MODEL_NAME: str="mlp", HIDDEN_SIZE: int=64,
    LOSS_NAME: str="feature_grad_ce", LOSS_KWARGS: Optional[Dict[str, Any]]=None,
    MODEL_SEEDS: Optional[List[int]]=None,
    device: str="cuda"
) -> Tuple[float, float, float]:
    print({
        "model_name": MODEL_NAME,
        "loss_name": LOSS_NAME,
        "feature_loss_weights": FEATURE_LOSS_WEIGHTS,
        "loss_kwargs": LOSS_KWARGS or {},
    })
    # a fixed seed for data split
    set_seeds(67)

    # dataset, dataloader
    train, val = data_loading_wrapper[dataset]["load_train_val"](
        REMOVED_FEATURE_INDICES, TRAIN_VAL_GROUP, VAL_RATE, DATASET_CONFIG
    )
    tests = data_loading_wrapper[dataset]["load_tests"](
        REMOVED_FEATURE_INDICES, TEST_GROUPS, DATASET_CONFIG
    )
    train_loader = DataLoader(train, batch_size=TRAIN_BATCH, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True)
    test_loaders = [DataLoader(test, batch_size=EVAL_BATCH, shuffle=False, pin_memory=True) for test in tests]

    # loss
    criterion = _build_criterion(
        LOSS_NAME,
        FEATURE_INDEX,
        REMOVED_FEATURE_INDICES,
        FEATURE_LOSS_WEIGHTS,
        REG_SCALE,
        device,
        LOSS_KWARGS,
    )

    # repeat experiments
    ID_accs = []
    OOD_MEAN_accs = []
    OOD_WORST_accs = []
    for repeat_i in range(REPEAT):
        # random seeds for model initialization
        model_seed = MODEL_SEEDS[repeat_i] if MODEL_SEEDS is not None else random.randint(0, 100000)
        set_seeds(model_seed)
        print(f"repeat {repeat_i + 1} model_seed={model_seed}")

        # model, loss, optimizer
        model = _build_model(
            MODEL_NAME,
            num_features=(len(FEATURE_INDEX) - len(REMOVED_FEATURE_INDICES)),
            hidden_size=HIDDEN_SIZE,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # training loop
        trainer = Trainer(
            device, PATIENCE, MAX_EPOCHS, PLOT_TEST_SHAP,
            train, val, tests,
            train_loader, val_loader, test_loaders,
            model, criterion, optimizer
        )
        epoch, train_losses, val_losses, train_accs, val_accs, test_state_accs, train_grad_terms_sum, shape_values = trainer.run_training(repeat_i)
        ID_acc, OOD_MEAN_acc, OOD_WORST_acc = max(val_accs), np.array(test_state_accs).mean(), min(test_state_accs)
        
        # check convergence & SHAP
        if repeat_i < 3:
            date = datetime.datetime.now()
            plot_training_curves(
                dataset, ID_acc, OOD_MEAN_acc, OOD_WORST_acc,
                FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                repeat_i, epoch, PATIENCE,
                train_losses, val_losses, train_accs, val_accs,
                train_grad_terms_sum, date
            )
            plot_shap_values(
                dataset, TRAIN_VAL_GROUP, TEST_GROUPS,
                FEATURE_INDEX, REMOVED_FEATURE_INDICES,
                shape_values, repeat_i, date
            )
        
        # logging
        ID_accs.append(ID_acc)
        OOD_MEAN_accs.append(OOD_MEAN_acc)
        OOD_WORST_accs.append(OOD_WORST_acc)
        print(f"{ID_acc:.4f}, {OOD_MEAN_acc:.4f}, {OOD_WORST_acc:.4f}")

    mean_ID_acc, mean_OOD_MEAN_acc, mean_OOD_WORST_acc = np.array(ID_accs).mean(), np.array(OOD_MEAN_accs).mean(), np.array(OOD_WORST_accs).mean()
    return mean_ID_acc, mean_OOD_MEAN_acc, mean_OOD_WORST_acc
