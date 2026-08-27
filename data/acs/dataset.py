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

# Folktables exposes ACS category codes as numeric values. These five fields are
# genuinely continuous; the remaining task predictors are coded categories.
ACS_CONTINUOUS_FEATURES = {"AGEP", "WKHP", "JWMNP", "PINCP", "POVPIP"}


class ACSTaskDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, feature_names: Optional[List[str]] = None):
        self.X = torch.from_numpy(X).to(dtype=torch.float)
        self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        self.n_samples = X.shape[0]
        if feature_names is not None:
            self.feature_names = list(feature_names)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def remove_feature(removed_feature_indices: List[int], X: np.ndarray) -> np.ndarray:
    removed = set(removed_feature_indices)
    keep_indices = [i for i in range(X.shape[1]) if i not in removed]
    return X[:, keep_indices]


def _kept_feature_names(feature_names: List[str], removed_feature_indices: List[int]) -> List[str]:
    removed = set(removed_feature_indices)
    return [name for idx, name in enumerate(feature_names) if idx not in removed]


def _category_token(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _fit_one_hot_schema(
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    categories = {}
    encoded_names = []
    for idx, name in enumerate(feature_names):
        if name in ACS_CONTINUOUS_FEATURES:
            encoded_names.append(name)
            continue
        values = np.unique(X[:, idx])
        values = values[~np.isnan(values)]
        categories[idx] = values
        encoded_names.extend(f"{name}_{_category_token(value)}" for value in values)
    return categories, encoded_names


def _apply_one_hot_schema(X: np.ndarray, categories: Dict[int, np.ndarray]) -> np.ndarray:
    columns = []
    for idx in range(X.shape[1]):
        if idx not in categories:
            columns.append(X[:, idx:idx + 1])
            continue
        columns.append((X[:, idx:idx + 1] == categories[idx][None, :]).astype(np.float32))
    return np.concatenate(columns, axis=1).astype(np.float32)


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


def _subsample_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_size <= 0 or X.shape[0] <= max_size:
        return X, Y
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    indices = indices[:max_size]
    return X[indices], Y[indices]


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
    categorical_encoding = _get_config_value(config, "categorical_encoding", "auto")
    if categorical_encoding not in {"auto", "integer", "one_hot"}:
        raise ValueError(f"Unsupported categorical_encoding: {categorical_encoding}")
    max_train_val_size = int(_get_config_value(config, "max_train_val_size", 0))
    max_per_test_size = int(_get_config_value(config, "max_per_test_size", 0))
    data_source = ACSDataSource(
        survey_year=ACS_SURVEY_YEAR,
        horizon=ACS_HORIZON,
        survey=ACS_SURVEY,
        root_dir=ACS_RAW_DATA_ROOT,
    )

    acs_data = data_source.get_data(states=[train_val_state], download=True)
    X_train_val, Y_train_val, _ = task.df_to_numpy(acs_data)
    X_train_val = remove_feature(removed_feature_indices, X_train_val)
    feature_names = _kept_feature_names(list(task.features), removed_feature_indices)

    indices = np.arange(X_train_val.shape[0])
    np.random.shuffle(indices)
    X_train_val, Y_train_val = X_train_val[indices], Y_train_val[indices]
    X_train_val, Y_train_val = _subsample_arrays(
        X_train_val,
        Y_train_val,
        max_train_val_size,
    )

    val_idx = int(val_rate * X_train_val.shape[0])
    X_train, X_val = X_train_val[val_idx:], X_train_val[:val_idx]
    Y_train, Y_val = Y_train_val[val_idx:], Y_train_val[:val_idx]

    one_hot_categories = None
    if categorical_encoding == "one_hot":
        one_hot_categories, feature_names = _fit_one_hot_schema(X_train, feature_names)
        X_train = _apply_one_hot_schema(X_train, one_hot_categories)
        X_val = _apply_one_hot_schema(X_val, one_hot_categories)

    stats = None
    if standardize:
        stats = _fit_standardization_stats(X_train)
        X_train = _standardize(X_train, stats)
        X_val = _standardize(X_val, stats)

    if resampling:
        print(f"{task_name} train Y distribution before resampling: {_label_distribution(Y_train)}")
        X_train, Y_train = _balanced_oversample(X_train, Y_train)
        print(f"{task_name} train Y distribution after resampling: {_label_distribution(Y_train)}")

    train = ACSTaskDataset(X_train, Y_train, feature_names)
    val = ACSTaskDataset(X_val, Y_val, feature_names)

    tests = []
    for state in test_states:
        acs_data = data_source.get_data(states=[state], download=True)
        X_test, Y_test, _ = task.df_to_numpy(acs_data)
        X_test = remove_feature(removed_feature_indices, X_test)
        X_test, Y_test = _subsample_arrays(X_test, Y_test, max_per_test_size)
        if one_hot_categories is not None:
            X_test = _apply_one_hot_schema(X_test, one_hot_categories)
        if standardize:
            X_test = _standardize(X_test, stats)
        tests.append(ACSTaskDataset(X_test, Y_test, feature_names))

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
    max_train_val_size = int(_get_config_value(config, "max_train_val_size", 0))
    max_per_test_size = int(_get_config_value(config, "max_per_test_size", 0))

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
        X_state_test, Y_state_test = _subsample_arrays(
            X_state_test,
            Y_state_test,
            max_per_test_size,
        )

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
    train_val_size = len(Y_train) + len(Y_val)
    if max_train_val_size > 0 and train_val_size > max_train_val_size:
        val_ratio = len(Y_val) / train_val_size
        X_train_val = np.concatenate([X_train, X_val], axis=0)
        Y_train_val = np.concatenate([Y_train, Y_val], axis=0)
        X_train_val, Y_train_val = _subsample_arrays(
            X_train_val,
            Y_train_val,
            max_train_val_size,
        )
        val_idx = int(val_ratio * X_train_val.shape[0])
        X_train, X_val = X_train_val[val_idx:], X_train_val[:val_idx]
        Y_train, Y_val = Y_train_val[val_idx:], Y_train_val[:val_idx]

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
