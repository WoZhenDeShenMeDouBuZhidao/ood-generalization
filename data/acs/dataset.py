import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSEmploymentFiltered,
    ACSHealthInsurance,
    ACSIncome,
    ACSIncomePovertyRatio,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
from src.paths import acs_raw_data_root


ACS_SURVEY_YEAR = "2018"
ACS_HORIZON = "1-Year"
ACS_SURVEY = "person"
ACS_RAW_DATA_ROOT = str(acs_raw_data_root())

ACS_TASKS = {
    "acsincome": ACSIncome,
    "acsemployment": ACSEmployment,
    "acsemploymentfiltered": ACSEmploymentFiltered,
    "acshealthinsurance": ACSHealthInsurance,
    "acsincomepovertyratio": ACSIncomePovertyRatio,
    "acsmobility": ACSMobility,
    "acspubliccoverage": ACSPublicCoverage,
    "acstraveltime": ACSTravelTime,
}


class ACSTaskDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).to(dtype=torch.float)
        self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def remove_feature(removed_feature_indices: List[int], X: np.ndarray) -> np.ndarray:
    removed = set(removed_feature_indices)
    keep_indices = [i for i in range(X.shape[1]) if i not in removed]
    return X[:, keep_indices]


def _label_distribution(Y: np.ndarray) -> str:
    labels, counts = np.unique(Y.astype(int), return_counts=True)
    total = len(Y)
    return ", ".join(
        f"{label}: {count} ({count / total:.4f})"
        for label, count in zip(labels, counts)
    )


def _balanced_oversample(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(Y, return_counts=True)
    if len(labels) < 2:
        return X, Y

    max_count = counts.max()
    resampled_indices = []
    for label, count in zip(labels, counts):
        label_indices = np.flatnonzero(Y == label)
        extra_indices = np.random.choice(label_indices, size=max_count - count, replace=True)
        resampled_indices.append(np.concatenate([label_indices, extra_indices]))

    indices = np.concatenate(resampled_indices)
    np.random.shuffle(indices)
    return X[indices], Y[indices]


def _fit_standardization_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}


def _standardize(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X - stats["mean"]) / stats["std"]).astype(np.float32)


def _get_config_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def acs_task_load_data(
    task_name: str,
    removed_feature_indices: List[int],
    train_val_state: str,
    test_states: List[str],
    val_rate: float,
    config: Optional[Any] = None,
) -> Tuple[ACSTaskDataset, ACSTaskDataset, List[ACSTaskDataset]]:
    if task_name not in ACS_TASKS:
        raise ValueError(f"Unsupported ACS task_name: {task_name}")

    task = ACS_TASKS[task_name]
    resampling = _get_config_value(config, "resampling", False)
    standardize = _get_config_value(config, "standardize", False)
    data_source = ACSDataSource(
        survey_year=ACS_SURVEY_YEAR,
        horizon=ACS_HORIZON,
        survey=ACS_SURVEY,
        root_dir=ACS_RAW_DATA_ROOT,
    )

    acs_data = data_source.get_data(states=[train_val_state], download=True)
    X_train_val, Y_train_val, _ = task.df_to_numpy(acs_data)
    X_train_val = remove_feature(removed_feature_indices, X_train_val)

    indices = np.arange(X_train_val.shape[0])
    np.random.shuffle(indices)
    X_train_val, Y_train_val = X_train_val[indices], Y_train_val[indices]

    val_idx = int(val_rate * len(indices))
    X_train, X_val = X_train_val[val_idx:], X_train_val[:val_idx]
    Y_train, Y_val = Y_train_val[val_idx:], Y_train_val[:val_idx]

    stats = None
    if standardize:
        stats = _fit_standardization_stats(X_train)
        X_train = _standardize(X_train, stats)
        X_val = _standardize(X_val, stats)

    if resampling:
        print(f"{task_name} train Y distribution before resampling: {_label_distribution(Y_train)}")
        X_train, Y_train = _balanced_oversample(X_train, Y_train)
        print(f"{task_name} train Y distribution after resampling: {_label_distribution(Y_train)}")

    train = ACSTaskDataset(X_train, Y_train)
    val = ACSTaskDataset(X_val, Y_val)

    tests = []
    for state in test_states:
        acs_data = data_source.get_data(states=[state], download=True)
        X_test, Y_test, _ = task.df_to_numpy(acs_data)
        X_test = remove_feature(removed_feature_indices, X_test)
        if standardize:
            X_test = _standardize(X_test, stats)
        tests.append(ACSTaskDataset(X_test, Y_test))

    return train, val, tests


def acsincome_load_data_acc_upperbound_test(
    removed_feature_indices: List[int],
    train_val_state: str,
    test_states: List[str],
    val_rate: float,
    config: Optional[Any] = None,
) -> Tuple[ACSTaskDataset, ACSTaskDataset, List[ACSTaskDataset]]:
    del train_val_state
    del val_rate
    train_data_per_state = int(_get_config_value(config, "train_data_per_state", 1000))
    val_data_per_state = int(_get_config_value(config, "val_data_per_state", 200))
    resampling = _get_config_value(config, "resampling", False)
    standardize = _get_config_value(config, "standardize", False)
    print_state_counts = _get_config_value(config, "print_state_counts", False)

    if train_data_per_state <= 0:
        raise ValueError(f"train_data_per_state must be positive, got {train_data_per_state}.")
    if val_data_per_state <= 0:
        raise ValueError(f"val_data_per_state must be positive, got {val_data_per_state}.")
    if not test_states:
        raise ValueError("test_states must contain at least one state.")

    data_source = ACSDataSource(
        survey_year=ACS_SURVEY_YEAR,
        horizon=ACS_HORIZON,
        survey=ACS_SURVEY,
        root_dir=ACS_RAW_DATA_ROOT,
    )
    task = ACS_TASKS["acsincome"]

    X_train_parts = []
    Y_train_parts = []
    X_val_parts = []
    Y_val_parts = []
    per_state_test_parts = []

    for state in test_states:
        acs_data = data_source.get_data(states=[state], download=True)
        X_state, Y_state, _ = task.df_to_numpy(acs_data)
        X_state = remove_feature(removed_feature_indices, X_state)

        used_data_per_state = train_data_per_state + val_data_per_state
        if X_state.shape[0] <= used_data_per_state:
            raise ValueError(
                f"State {state} has {X_state.shape[0]} rows, "
                f"which is not enough for train_data_per_state={train_data_per_state}, "
                f"val_data_per_state={val_data_per_state}, and a non-empty test split."
            )

        indices = np.arange(X_state.shape[0])
        np.random.shuffle(indices)
        train_indices = indices[:train_data_per_state]
        val_indices = indices[train_data_per_state:used_data_per_state]
        test_indices = indices[used_data_per_state:]

        X_state_train, Y_state_train = X_state[train_indices], Y_state[train_indices]
        X_state_val, Y_state_val = X_state[val_indices], Y_state[val_indices]
        X_state_test, Y_state_test = X_state[test_indices], Y_state[test_indices]

        X_train_parts.append(X_state_train)
        Y_train_parts.append(Y_state_train)
        X_val_parts.append(X_state_val)
        Y_val_parts.append(Y_state_val)
        per_state_test_parts.append((X_state_test, Y_state_test))

        if print_state_counts:
            print(
                f"ACSIncome acc upperbound {state}: "
                f"train={len(Y_state_train)}, valid={len(Y_state_val)}, test={len(Y_state_test)}"
            )

    X_train = np.concatenate(X_train_parts, axis=0)
    Y_train = np.concatenate(Y_train_parts, axis=0)
    X_val = np.concatenate(X_val_parts, axis=0)
    Y_val = np.concatenate(Y_val_parts, axis=0)

    stats = None
    if standardize:
        stats = _fit_standardization_stats(X_train)
        X_train = _standardize(X_train, stats)
        X_val = _standardize(X_val, stats)
        per_state_test_parts = [
            (_standardize(X_state_test, stats), Y_state_test)
            for X_state_test, Y_state_test in per_state_test_parts
        ]

    if resampling:
        print(f"ACSIncome acc upperbound train Y distribution before resampling: {_label_distribution(Y_train)}")
        X_train, Y_train = _balanced_oversample(X_train, Y_train)
        print(f"ACSIncome acc upperbound train Y distribution after resampling: {_label_distribution(Y_train)}")

    print(
        "ACSIncome acc upperbound split: "
        f"states={len(test_states)}, "
        f"train_data_per_state={train_data_per_state}, "
        f"val_data_per_state={val_data_per_state}, "
        f"train={len(Y_train)}, valid={len(Y_val)}, test_states={len(per_state_test_parts)}"
    )
    print(f"ACSIncome acc upperbound train Y distribution: {_label_distribution(Y_train)}")
    print(f"ACSIncome acc upperbound valid Y distribution: {_label_distribution(Y_val)}")
    Y_test_all = np.concatenate([Y_state_test for _, Y_state_test in per_state_test_parts], axis=0)
    print(f"ACSIncome acc upperbound test Y distribution: {_label_distribution(Y_test_all)}")

    train = ACSTaskDataset(X_train, Y_train)
    val = ACSTaskDataset(X_val, Y_val)
    tests = [
        ACSTaskDataset(X_state_test, Y_state_test)
        for X_state_test, Y_state_test in per_state_test_parts
    ]
    return train, val, tests
