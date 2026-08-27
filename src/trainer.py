import copy
import shap
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset
from src.mlp import MLP
from src.metrics import (
    binary_balanced_accuracy_macro_f1_from_confusion,
    binary_confusion_from_logits,
)
from src.loss import (
    diagnostic_gradient_terms,
    diagnostic_weight_terms,
    logit_margin_gradients,
)
from tqdm import tqdm


def margin_grad_l2(
    model: nn.Module,
    train: Dataset,
    device: str,
    batch_size: int,
    class_reweighting: bool,
) -> List[float]:
    """Measure raw margin Grad-L2 at one fixed checkpoint."""
    was_training = model.training
    model.eval()
    class_sums: Dict[int, torch.Tensor] = {}
    class_counts: Dict[int, int] = {}
    try:
        for start in range(0, len(train), batch_size):
            Xs = (
                train.X[start:start + batch_size]
                .to(device, non_blocking=True)
                .detach()
                .requires_grad_(True)
            )
            Ys = train.Y[start:start + batch_size].to(device, non_blocking=True)
            grads = logit_margin_gradients(
                model(Xs),
                Xs,
                create_graph=False,
                retain_graph=False,
            )
            grad_l2 = grads.pow(2)
            for class_value in torch.unique(Ys.detach(), sorted=True):
                class_id = int(class_value.item())
                mask = Ys == class_value
                contribution = grad_l2[mask].sum(dim=0).detach()
                if class_id in class_sums:
                    class_sums[class_id] += contribution
                else:
                    class_sums[class_id] = contribution
                class_counts[class_id] = (
                    class_counts.get(class_id, 0) + int(mask.sum().item())
                )
    finally:
        model.train(was_training)

    if not class_sums:
        return []
    if class_reweighting:
        grad_l2 = torch.stack([
            class_sums[class_id] / class_counts[class_id]
            for class_id in sorted(class_sums)
        ]).mean(dim=0)
    else:
        grad_l2 = sum(class_sums.values()) / sum(class_counts.values())
    return grad_l2.cpu().tolist()


class Trainer():
    def __init__(
        self, device: str, PATIENCE: int, MAX_EPOCHS: int, SHAP_ON_VAL: bool, SHAP_ON_TESTS: bool,
        train: Dataset, val: Dataset, tests: List[Dataset], train_loader: DataLoader, val_loader: DataLoader, test_loaders: List[DataLoader],
        model: MLP, criterion: nn.Module, optimizer: optim.Adam,
        REG_SCALE: float = 0.0,
        REG_WARMUP_EPOCHS: int = 0,
        SHAP_SAMPLE_SIZE: int = 500,
        SHOW_PROGRESS: bool = False,
        TRACK_DIAGNOSTICS: bool = False,
        RECORD_BEST_GRAD_L2: bool = False,
        FEATURE_NAMES: List[str] | None = None,
    ):
        self.device = device
        self.PATIENCE = PATIENCE
        self.MAX_EPOCHS = MAX_EPOCHS
        self.SHAP_ON_VAL = SHAP_ON_VAL
        self.SHAP_ON_TESTS = SHAP_ON_TESTS
        self.SHAP_SAMPLE_SIZE = SHAP_SAMPLE_SIZE
        self.train, self.val, self.tests = train, val, tests
        self.train_loader, self.val_loader, self.test_loaders = train_loader, val_loader, test_loaders
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.REG_SCALE = REG_SCALE
        self.REG_WARMUP_EPOCHS = REG_WARMUP_EPOCHS
        self.SHOW_PROGRESS = SHOW_PROGRESS
        self.TRACK_DIAGNOSTICS = TRACK_DIAGNOSTICS
        self.RECORD_BEST_GRAD_L2 = RECORD_BEST_GRAD_L2
        self.FEATURE_NAMES = FEATURE_NAMES or [f"feature_{idx}" for idx in range(train.X.shape[1])]

    def run_training(
        self, repeat_i
    ) -> Tuple[
        int,
        Dict[str, List[float]],
        Dict[str, List[float]],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
        Dict[str, List[float]],
        List[float] | None,
        List[np.ndarray],
    ]:
        epoch = 0
        no_improve_epoch = 0
        best_val_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        train_losses = {}
        val_losses = {}
        train_balanced_accuracies, val_balanced_accuracies = [], []
        train_macro_f1s, val_macro_f1s = [], []
        train_grads = {}
        for _ in tqdm(
            range(self.MAX_EPOCHS),
            desc=f"training repeat {repeat_i + 1}",
            disable=not self.SHOW_PROGRESS,
        ):
            if self.REG_WARMUP_EPOCHS:
                self.criterion.reg_scale = self.REG_SCALE * min(
                    epoch / self.REG_WARMUP_EPOCHS,
                    1.0,
                )

            # training
            self.model.train()
            train_tp = 0
            train_tn = 0
            train_fp = 0
            train_fn = 0
            train_loss_terms_sum = {}
            train_grad_terms_sum = {}
            for Xs, Ys in self.train_loader:
                Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits, loss, loss_terms, grad_terms = self.criterion(self.model, Xs, Ys)
                loss.backward()
                self.optimizer.step()

                batch_tp, batch_tn, batch_fp, batch_fn = binary_confusion_from_logits(logits, Ys)
                train_tp += batch_tp.item()
                train_tn += batch_tn.item()
                train_fp += batch_fp.item()
                train_fn += batch_fn.item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in train_loss_terms_sum:
                        train_loss_terms_sum[loss_name] += loss_value.detach() * Xs.size(0)
                    else:
                        train_loss_terms_sum[loss_name] = loss_value.detach() * Xs.size(0)
                if self.TRACK_DIAGNOSTICS:
                    if hasattr(self.criterion, "diagnostic_gradient_terms"):
                        grad_terms = self.criterion.diagnostic_gradient_terms(
                            self.model,
                            Xs,
                        )
                    else:
                        grad_terms = diagnostic_gradient_terms(
                            self.model,
                            Xs,
                            self.FEATURE_NAMES,
                            getattr(self.criterion, "grad_prob_temperature", 1.0),
                        )
                for grad_name, grad_value in grad_terms.items():
                    if grad_name in train_grad_terms_sum:
                        train_grad_terms_sum[grad_name] += grad_value.detach() * Xs.size(0)
                    else:
                        train_grad_terms_sum[grad_name] = grad_value.detach() * Xs.size(0)

            train_balanced_accuracy, train_macro_f1 = (
                binary_balanced_accuracy_macro_f1_from_confusion(
                    train_tp,
                    train_tn,
                    train_fp,
                    train_fn,
                )
            )
            if self.TRACK_DIAGNOSTICS:
                weight_terms = diagnostic_weight_terms(self.model, self.FEATURE_NAMES)
                for grad_name, grad_value in weight_terms.items():
                    train_grad_terms_sum[grad_name] = grad_value.detach() * len(self.train)
            

            # validation
            self.model.eval()
            val_tp = 0
            val_tn = 0
            val_fp = 0
            val_fn = 0
            val_loss_terms_sum = {}
            for Xs, Ys in self.val_loader:
                Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                logits, loss, loss_terms, _ = self.criterion(self.model, Xs, Ys)

                batch_tp, batch_tn, batch_fp, batch_fn = binary_confusion_from_logits(logits, Ys)
                val_tp += batch_tp.item()
                val_tn += batch_tn.item()
                val_fp += batch_fp.item()
                val_fn += batch_fn.item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in val_loss_terms_sum:
                        val_loss_terms_sum[loss_name] += loss_value.detach() * Xs.size(0)
                    else:
                        val_loss_terms_sum[loss_name] = loss_value.detach() * Xs.size(0)

            val_balanced_accuracy, val_macro_f1 = (
                binary_balanced_accuracy_macro_f1_from_confusion(
                    val_tp,
                    val_tn,
                    val_fp,
                    val_fn,
                )
            )
            val_score = val_balanced_accuracy + val_macro_f1


            # track loss and metrics
            epoch += 1
            def _append_epoch_terms(history, term_sums, denom):
                if not term_sums:
                    return

                names = list(term_sums.keys())
                values = torch.stack([
                    term_sums[name] / denom
                    for name in names
                ]).detach().cpu().tolist()

                for name, value in zip(names, values):
                    history.setdefault(name, []).append(float(value))
            _append_epoch_terms(train_losses, train_loss_terms_sum, len(self.train))
            _append_epoch_terms(val_losses, val_loss_terms_sum, len(self.val))
            _append_epoch_terms(train_grads, train_grad_terms_sum, len(self.train))
            train_balanced_accuracies.append(train_balanced_accuracy)
            val_balanced_accuracies.append(val_balanced_accuracy)
            train_macro_f1s.append(train_macro_f1)
            val_macro_f1s.append(val_macro_f1)


            # save best model
            if val_score > best_val_score:
                no_improve_epoch = 0
                best_val_score = val_score
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= self.PATIENCE:
                    break

        if self.REG_WARMUP_EPOCHS:
            self.criterion.reg_scale = self.REG_SCALE

        # testing
        with torch.inference_mode():
            self.model.load_state_dict(best_state)
            best_model = self.model
            test_state_balanced_accuracies = []
            test_state_macro_f1s = []

            batch_size = self.test_loaders[0].batch_size if self.test_loaders else 2048
            test_tensors = [
                (
                    test.X.to(self.device, non_blocking=True),
                    test.Y.to(self.device, non_blocking=True),
                )
                for test in self.tests
            ]

            for X_test, Y_test in test_tensors:
                test_tp = 0
                test_tn = 0
                test_fp = 0
                test_fn = 0
                for start in range(0, X_test.size(0), batch_size):
                    Xs = X_test[start:start + batch_size]
                    Ys = Y_test[start:start + batch_size]
                    logits = best_model(Xs)
                    batch_tp, batch_tn, batch_fp, batch_fn = binary_confusion_from_logits(logits, Ys)
                    test_tp += batch_tp.item()
                    test_tn += batch_tn.item()
                    test_fp += batch_fp.item()
                    test_fn += batch_fn.item()

                balanced_accuracy, macro_f1 = (
                    binary_balanced_accuracy_macro_f1_from_confusion(
                        test_tp,
                        test_tn,
                        test_fp,
                        test_fn,
                    )
                )
                test_state_balanced_accuracies.append(balanced_accuracy)
                test_state_macro_f1s.append(macro_f1)

        best_grad_l2 = None
        if self.RECORD_BEST_GRAD_L2:
            best_grad_l2 = margin_grad_l2(
                self.model,
                self.train,
                self.device,
                self.train_loader.batch_size or 256,
                class_reweighting=bool(
                    getattr(self.criterion, "reweighting", False)
                ),
            )

        # shap
        def predict_fn(X_np):
            X_tensor = torch.as_tensor(X_np, dtype=torch.float32)
            with torch.inference_mode():
                logits = self.model(X_tensor)
                return (logits[:, 1] - logits[:, 0]).detach().numpy()

        shap_values = []
        if self.SHAP_ON_VAL and repeat_i < 3:
            self.model.to("cpu")
            X_background = self.train.X.detach().numpy()
            explainer = shap.Explainer(predict_fn, X_background, algorithm="permutation")

            X_explain = shap.sample(self.val.X.detach().numpy(), min(self.SHAP_SAMPLE_SIZE, len(self.val)))
            max_evals = max(500, 2 * X_explain.shape[1] + 1)
            shap_values.append(explainer(X_explain, max_evals=max_evals, silent=True).values)
            if self.SHAP_ON_TESTS:
                for test in self.tests:
                    X_explain = shap.sample(test.X.detach().numpy(), min(self.SHAP_SAMPLE_SIZE, len(test)))
                    max_evals = max(500, 2 * X_explain.shape[1] + 1)
                    shap_values.append(explainer(X_explain, max_evals=max_evals, silent=True).values)

        return (
            epoch,
            train_losses,
            val_losses,
            train_balanced_accuracies,
            val_balanced_accuracies,
            train_macro_f1s,
            val_macro_f1s,
            test_state_balanced_accuracies,
            test_state_macro_f1s,
            train_grads,
            best_grad_l2,
            shap_values,
        )
