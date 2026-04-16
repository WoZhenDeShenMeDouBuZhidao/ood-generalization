import copy
import shap
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset
from src.mlp import MLP
from src.loss import FeatureGradCELoss
from tqdm import tqdm


class Trainer():
    def __init__(
        self, device: str, PATIENCE: int, MAX_EPOCHS: int, SHAP_ON_TESTS: bool,
        train: Dataset, val: Dataset, tests: List[Dataset],
        train_loader: DataLoader, val_loader: DataLoader, test_loaders: List[DataLoader],
        model: MLP, criterion: FeatureGradCELoss, optimizer: optim.Adam
    ):
        self.device = device
        self.PATIENCE = PATIENCE
        self.MAX_EPOCHS = MAX_EPOCHS
        self.SHAP_ON_TESTS = SHAP_ON_TESTS
        self.train, self.val, self.tests = train, val, tests
        self.train_loader, self.val_loader, self.test_loaders = train_loader, val_loader, test_loaders
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
    
    def run_training(
        self, repeat_i
    ) -> Tuple[int, Dict[str, List[float]], Dict[str, List[float]], List[float], List[float], List[float], Dict[str, List[float]], List[np.ndarray]]:
        epoch = 0
        no_improve_epoch = 0
        best_val_acc = 0
        best_state = copy.deepcopy(self.model.state_dict())
        train_losses = {}
        val_losses = {}
        train_accs, val_accs = [], []
        train_grads = {}
        for _ in tqdm(range(self.MAX_EPOCHS), desc=f"training repeat {repeat_i + 1}"):
            # training
            self.model.train()
            train_correct = 0
            train_loss_terms_sum = {}
            train_grad_terms_sum = {}
            for Xs, Ys in self.train_loader:
                Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits, loss, loss_terms, grad_terms = self.criterion(self.model, Xs, Ys)
                loss.backward()
                self.optimizer.step()

                train_correct += (logits.argmax(dim=1) == Ys).sum().item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in train_loss_terms_sum:
                        train_loss_terms_sum[loss_name] += loss_value * Xs.size(0)
                    else:
                        train_loss_terms_sum[loss_name] = loss_value * Xs.size(0)
                for grad_name, grad_value in grad_terms.items():
                    if grad_name in train_grad_terms_sum:
                        train_grad_terms_sum[grad_name] += grad_value * Xs.size(0)
                    else:
                        train_grad_terms_sum[grad_name] = grad_value * Xs.size(0)

            train_acc = train_correct / len(self.train)
            

            # validation
            self.model.eval()
            val_correct = 0
            val_loss_terms_sum = {}
            for Xs, Ys in self.val_loader:
                Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                logits, loss, loss_terms, _ = self.criterion(self.model, Xs, Ys)

                val_correct += (logits.argmax(dim=1) == Ys).sum().item()
                for loss_name, loss_value in loss_terms.items():
                    if loss_name in val_loss_terms_sum:
                        val_loss_terms_sum[loss_name] += loss_value * Xs.size(0)
                    else:
                        val_loss_terms_sum[loss_name] = loss_value * Xs.size(0)

            val_acc = val_correct / len(self.val)


            # track loss, acc
            epoch += 1
            for loss_name, loss_sum in train_loss_terms_sum.items():
                if loss_name in train_losses:
                    train_losses[loss_name].append(float(loss_sum.detach() / len(self.train)))
                else:
                    train_losses[loss_name] = [float(loss_sum.detach() / len(self.train))]
            for loss_name, loss_sum in val_loss_terms_sum.items():
                if loss_name in val_losses:
                    val_losses[loss_name].append(float(loss_sum.detach() / len(self.val)))
                else:
                    val_losses[loss_name] = [float(loss_sum.detach() / len(self.val))]
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            for grad_name, grad_sum in train_grad_terms_sum.items():
                if grad_name in train_grads:
                    train_grads[grad_name].append(float(grad_sum.detach() / len(self.train)))
                else:
                    train_grads[grad_name] = [float(grad_sum.detach() / len(self.train))]


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
            for test, test_loader in tqdm(zip(self.tests, self.test_loaders), total=len(self.tests), desc=f"testing repeat {repeat_i + 1}"):
                test_correct = 0
                for Xs, Ys in test_loader:
                    Xs, Ys = Xs.to(self.device, non_blocking=True), Ys.to(self.device, non_blocking=True)
                    logits = best_model(Xs)
                    test_correct += (logits.argmax(dim=1) == Ys).sum().item()
                test_state_accs.append(test_correct / len(test))
        

        # shap
        def predict_fn(X_np):
            X_tensor = torch.as_tensor(X_np, dtype=torch.float32)
            with torch.inference_mode():
                return self.model(X_tensor).detach().numpy()

        shap_values = []
        if repeat_i < 3:
            self.model.to("cpu")
            X_background = self.train.X.detach().numpy()
            explainer = shap.Explainer(predict_fn, X_background)

            X_explain = self.val.X.detach().numpy()
            shap_values.append(explainer(X_explain).values) # ndarray shape: (#data, #feature, #class)
            if self.SHAP_ON_TESTS:
                for test in self.tests:
                    X_explain = shap.sample(test.X.detach().numpy(), min(2000, len(test)))
                    shap_values.append(explainer(X_explain).values)

        return epoch, train_losses, val_losses, train_accs, val_accs, test_state_accs, train_grads, shap_values
