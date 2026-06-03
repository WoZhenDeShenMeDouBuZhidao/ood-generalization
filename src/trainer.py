import copy
import shap
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset
from src.mlp import MLP
from src.loss import FeatureGradCELoss
from src.metrics import binary_counts_from_logits, binary_f1_from_counts
from tqdm import tqdm


class Trainer():
    def __init__(
        self, device: str, PATIENCE: int, MAX_EPOCHS: int, SHAP_ON_VAL: bool, SHAP_ON_TESTS: bool,
        train: Dataset, val: Dataset, tests: List[Dataset], train_loader: DataLoader, val_loader: DataLoader, test_loaders: List[DataLoader],
        model: MLP, criterion: FeatureGradCELoss, optimizer: optim.Adam,
        SHOW_PROGRESS: bool = False,
    ):
        self.device = device
        self.PATIENCE = PATIENCE
        self.MAX_EPOCHS = MAX_EPOCHS
        self.SHAP_ON_VAL = SHAP_ON_VAL
        self.SHAP_ON_TESTS = SHAP_ON_TESTS
        self.train, self.val, self.tests = train, val, tests
        self.train_loader, self.val_loader, self.test_loaders = train_loader, val_loader, test_loaders
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.SHOW_PROGRESS = SHOW_PROGRESS
    
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
        List[np.ndarray],
    ]:
        epoch = 0
        no_improve_epoch = 0
        best_val_acc = 0
        best_state = copy.deepcopy(self.model.state_dict())
        train_losses = {}
        val_losses = {}
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        train_grads = {}
        for _ in tqdm(
            range(self.MAX_EPOCHS),
            desc=f"training repeat {repeat_i + 1}",
            disable=not self.SHOW_PROGRESS,
        ):
            # training
            self.model.train()
            train_correct = 0
            train_tp = 0
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

                batch_correct, batch_tp, batch_fp, batch_fn = binary_counts_from_logits(logits, Ys)
                train_correct += batch_correct
                train_tp += batch_tp.item()
                train_fp += batch_fp.item()
                train_fn += batch_fn.item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in train_loss_terms_sum:
                        train_loss_terms_sum[loss_name] += loss_value.detach() * Xs.size(0)
                    else:
                        train_loss_terms_sum[loss_name] = loss_value.detach() * Xs.size(0)
                for grad_name, grad_value in grad_terms.items():
                    if grad_name in train_grad_terms_sum:
                        train_grad_terms_sum[grad_name] += grad_value.detach() * Xs.size(0)
                    else:
                        train_grad_terms_sum[grad_name] = grad_value.detach() * Xs.size(0)

            train_acc = train_correct.item() / len(self.train)
            train_f1 = binary_f1_from_counts(train_tp, train_fp, train_fn)
            

            # validation
            self.model.eval()
            val_correct = 0
            val_tp = 0
            val_fp = 0
            val_fn = 0
            val_loss_terms_sum = {}
            for Xs, Ys in self.val_loader:
                Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                logits, loss, loss_terms, _ = self.criterion(self.model, Xs, Ys)

                batch_correct, batch_tp, batch_fp, batch_fn = binary_counts_from_logits(logits, Ys)
                val_correct += batch_correct
                val_tp += batch_tp.item()
                val_fp += batch_fp.item()
                val_fn += batch_fn.item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in val_loss_terms_sum:
                        val_loss_terms_sum[loss_name] += loss_value.detach() * Xs.size(0)
                    else:
                        val_loss_terms_sum[loss_name] = loss_value.detach() * Xs.size(0)

            val_acc = val_correct.item() / len(self.val)
            val_f1 = binary_f1_from_counts(val_tp, val_fp, val_fn)


            # track loss, acc
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
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)


            # save best model
            if val_acc > best_val_acc:
                no_improve_epoch = 0
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= self.PATIENCE:
                    break


        # testing
        with torch.inference_mode():
            self.model.load_state_dict(best_state)
            best_model = self.model
            test_state_accs = []
            test_state_f1s = []

            batch_size = self.test_loaders[0].batch_size if self.test_loaders else 2048
            test_tensors = [
                (
                    test.X.to(self.device, non_blocking=True),
                    test.Y.to(self.device, non_blocking=True),
                )
                for test in self.tests
            ]

            for X_test, Y_test in test_tensors:
                test_correct = torch.zeros((), device=self.device, dtype=torch.long)
                test_tp = 0
                test_fp = 0
                test_fn = 0
                for start in range(0, X_test.size(0), batch_size):
                    Xs = X_test[start:start + batch_size]
                    Ys = Y_test[start:start + batch_size]
                    logits = best_model(Xs)
                    batch_correct, batch_tp, batch_fp, batch_fn = binary_counts_from_logits(logits, Ys)
                    test_correct += batch_correct
                    test_tp += batch_tp.item()
                    test_fp += batch_fp.item()
                    test_fn += batch_fn.item()

                test_state_accs.append(test_correct.item() / Y_test.numel())
                test_state_f1s.append(binary_f1_from_counts(test_tp, test_fp, test_fn))


        # shap
        def predict_fn(X_np):
            X_tensor = torch.as_tensor(X_np, dtype=torch.float32)
            with torch.inference_mode():
                return self.model(X_tensor).detach().numpy()

        shap_values = []
        if self.SHAP_ON_VAL and repeat_i < 3:
            self.model.to("cpu")
            X_background = self.train.X.detach().numpy()
            explainer = shap.Explainer(predict_fn, X_background)

            X_explain = self.val.X.detach().numpy()
            shap_values.append(explainer(X_explain).values) # ndarray shape: (#data, #feature, #class)
            if self.SHAP_ON_TESTS:
                for test in self.tests:
                    X_explain = shap.sample(test.X.detach().numpy(), min(2000, len(test)))
                    shap_values.append(explainer(X_explain).values)

        return (
            epoch,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            train_f1s,
            val_f1s,
            test_state_accs,
            test_state_f1s,
            train_grads,
            shap_values,
        )
