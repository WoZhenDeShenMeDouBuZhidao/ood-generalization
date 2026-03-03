import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from dataset import load_train_val, load_tests
from mlp import MLP
from trainer import Trainer
from utils import set_seeds, plot_training_curves, plot_accdelta_bars


def main(
    TRAIN_VAL_STATE: str, TEST_STATES: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int],
    BATCH_SIZE: int=256, VAL_RATE: float=0.2, LR: int=1e-4, 
    PATIENCE: int=50, REPEAT: int=10, device: str="cuda"
) -> Tuple[float, float, float]:
    # a fixed seed for data split
    set_seeds(67)

    # dataset, dataloader
    train, val = load_train_val(REMOVED_FEATURE_INDICES, TRAIN_VAL_STATE, VAL_RATE)
    tests = load_tests(REMOVED_FEATURE_INDICES, TEST_STATES)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
    test_loaders = [DataLoader(test, batch_size=BATCH_SIZE, shuffle=False) for test in tests]

    # repeat experiments
    ID_accs = []
    OOD_MEAN_accs = []
    OOD_WORST_accs = []
    for repeat_i in range(REPEAT):
        set_seeds(random.randint(0, 100000))

        # model, loss, optimizer
        model = MLP(
            num_features=(len(FEATURE_INDEX) - len(REMOVED_FEATURE_INDICES)),
            hidden_size=64,
            num_classes=2
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # training loop
        trainer = Trainer(
            device, PATIENCE, 
            train, val, tests,
            train_loader, val_loader, test_loaders,
            model, criterion, optimizer
        )
        epoch, train_losses, val_losses, train_accs, val_accs, test_state_accs = trainer.run_training(repeat_i)

        # logging
        ID_accs.append(max(val_accs))
        OOD_MEAN_accs.append(np.array(test_state_accs).mean())
        OOD_WORST_accs.append(min(test_state_accs))
        print(f"{(max(val_accs)):.4f}, {(np.array(test_state_accs).mean()):.4f}, {(min(test_state_accs)):.4f}")

        # check convergence
        plot_training_curves(repeat_i, FEATURE_INDEX, REMOVED_FEATURE_INDICES, epoch, train_losses, val_losses, train_accs, val_accs)

    ID_acc, OOD_MEAN_acc, OOD_WORST_acc = np.array(ID_accs).mean(), np.array(OOD_MEAN_accs).mean(), np.array(OOD_WORST_accs).mean()
    return ID_acc, OOD_MEAN_acc, OOD_WORST_acc


if __name__ == "__main__":
    # data config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    TRAIN_VAL_STATE = "PR"
    TEST_STATES = [ 
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IA", "KS", "ME", "MD", "MA", "MI", "MN",
        "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
        "ND", "OH", "OK", "OR", "PA", "RI", "SD", "TN", "TX", "UT",
        "VT", "VA", "WA", "WV", "WI", "WY",  # States list (known corrupted states excluded)
    ]
    FEATURE_INDEX = {
        0: 'AGEP',
        1: 'COW',
        2: 'SCHL',
        3: 'MAR',
        4: 'OCCP',
        5: 'POBP',
        6: 'RELP',
        7: 'WKHP',
        8: 'SEX',
        9: 'RAC1P'
    }

    # hyperparameter config
    BATCH_SIZE = 256
    LR = 1e-4
    PATIENCE = 50
    REPEAT = 3

    # use all features
    REMOVED_FEATURE_INDICES = []
    ID_base, OOD_MEAN_base, OOD_WORST_base = main(
        TRAIN_VAL_STATE, TEST_STATES,
        FEATURE_INDEX, REMOVED_FEATURE_INDICES,
        BATCH_SIZE=BATCH_SIZE, LR=LR,
        PATIENCE=PATIENCE, REPEAT=REPEAT,
        device=device,
    )
    print(f"### Use all features:\n- ID: {ID_base:.4f}\n- OOD MEAN: {OOD_MEAN_base:.4f}\n- OOD WORST: {OOD_WORST_base:.4f}")

    # leave one feature out
    rmfeature_accdelta = {}
    for feat_i, feat in FEATURE_INDEX.items():
        REMOVED_FEATURE_INDICES = [feat_i]
        ID_rm, OOD_MEAN_rm, OOD_WORST_rm = main(
            TRAIN_VAL_STATE, TEST_STATES,
            FEATURE_INDEX, REMOVED_FEATURE_INDICES,
            BATCH_SIZE=BATCH_SIZE, LR=LR,
            PATIENCE=PATIENCE, REPEAT=REPEAT,
            device=device,
        )
        print(f"### Remove {feat}:\n- ID: {ID_rm:.4f}\n- OOD MEAN: {OOD_MEAN_rm:.4f}\n- OOD WORST: {OOD_WORST_rm:.4f}")

        rmfeature_accdelta[feat] = {
            "ID": (ID_rm - ID_base),
            "OOD MEAN": (OOD_MEAN_rm - OOD_MEAN_base),
            "OOD WORST": (OOD_WORST_rm - OOD_WORST_base)
        }
    

    # check distributional shift, causal features
    plot_accdelta_bars(rmfeature_accdelta, ID_base, OOD_MEAN_base, OOD_WORST_base)
