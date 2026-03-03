import copy
import torch
from typing import List, Tuple
from tqdm import tqdm


class Trainer():
    def __init__(
        self, device, PATIENCE,
        train, val, tests,
        train_loader, val_loader, test_loaders,
        model, criterion, optimizer
    ):
        self.device = device
        self.PATIENCE = PATIENCE
        self.train, self.val, self.tests = train, val, tests
        self.train_loader, self.val_loader, self.test_loaders = train_loader, val_loader, test_loaders
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
    
    def run_training(self, repeat_i) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        epoch = 0
        no_improve_epoch = 0
        best_val_acc = 0
        best_state = copy.deepcopy(self.model.state_dict())
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        for _ in tqdm(range(1000), desc=f"training repeat {repeat_i + 1}"):
            # training
            self.model.train()

            train_loss = 0
            train_correct = 0

            for Xs, Ys in self.train_loader:
                Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                self.optimizer.zero_grad()
                use_input_grads = getattr(self.criterion, "requires_inputs", False)
                if use_input_grads:
                    Xs = Xs.requires_grad_(True)
                preds = self.model(Xs)
                loss = self.criterion(preds, Ys, Xs) if use_input_grads else self.criterion(preds, Ys)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += (preds.argmax(dim=1) == Ys).sum().item()

            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = train_correct / len(self.train)
            
            # validation
            self.model.eval()

            val_loss = 0
            val_correct = 0

            with torch.no_grad():
                for Xs, Ys in self.val_loader:
                    Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                    preds = self.model(Xs)
                    loss = self.criterion(preds, Ys)

                    val_loss += loss.item()
                    val_correct += (preds.argmax(dim=1) == Ys).sum().item()

            avg_val_loss = val_loss / len(self.val_loader)
            val_acc = val_correct / len(self.val)

            # track loss, acc
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            epoch += 1

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
        self.model.load_state_dict(best_state)
        best_model = self.model
        test_state_accs = []
        for test, test_loader in tqdm(zip(self.tests, self.test_loaders), total=len(self.tests), desc=f"testing repeat {repeat_i + 1}"):
            best_model.eval()

            test_correct = 0

            with torch.no_grad():
                for Xs, Ys in test_loader:
                    Xs, Ys = Xs.to(self.device), Ys.to(self.device)
                    preds = best_model(Xs)
                    loss = self.criterion(preds, Ys)

                    test_correct += (preds.argmax(dim=1) == Ys).sum().item()

            test_state_accs.append(test_correct / len(test))

        return epoch, train_losses, val_losses, train_accs, val_accs, test_state_accs
