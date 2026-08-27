from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SYNTHETIC_DATASET_ORDER = (
    "simple",
    "range",
    "categorical_integer",
    "categorical_onehot",
    "multi_spurious",
)

SYNTHETIC_TEST_GROUPS = (
    "id_test",
    "ood_weak",
    "ood_independent",
    "ood_heterogeneous",
    "ood_reverse",
)
SYNTHETIC_REPORT_ID_GROUPS = ("id_test",)
SYNTHETIC_REPORT_OOD_GROUPS = SYNTHETIC_TEST_GROUPS[1:]

_SIMPLE_FEATURES = (
    "causal_continuous_1p25",
    "causal_continuous_0p75",
    "spurious_continuous_2p25",
    "noise_continuous_1",
    "noise_continuous_2",
    "noise_continuous_3",
)
_INTEGER_CATEGORY_FEATURES = (
    "causal_binary",
    "causal_ordinal",
    "causal_nominal",
    "noise_categorical_1",
)
_ONEHOT_CATEGORY_GROUPS = {
    "causal_binary": 2,
    "causal_ordinal": 4,
    "causal_nominal": 4,
    "noise_categorical_1": 4,
}
_MULTI_SPURIOUS_FEATURES = (
    "causal_continuous_1p25",
    "causal_continuous_0p75",
    "spurious_continuous_2p25",
    "spurious_continuous_1p75",
    "spurious_continuous_1p25",
    "spurious_continuous_0p75",
    "noise_continuous_1",
    "noise_continuous_2",
    "noise_continuous_3",
)


def _onehot_feature_names() -> Tuple[str, ...]:
    names = list(_SIMPLE_FEATURES)
    for group, levels in _ONEHOT_CATEGORY_GROUPS.items():
        names.extend(f"{group}_category_{level}" for level in range(levels))
    return tuple(names)


_FEATURE_NAMES = {
    "simple": _SIMPLE_FEATURES,
    "range": _SIMPLE_FEATURES,
    "categorical_integer": _SIMPLE_FEATURES + _INTEGER_CATEGORY_FEATURES,
    "categorical_onehot": _onehot_feature_names(),
    "multi_spurious": _MULTI_SPURIOUS_FEATURES,
}

_SIMPLE_ORACLE = {
    "causal_continuous_1p25": 2.0,
    "causal_continuous_0p75": 1.0,
    "spurious_continuous_2p25": 0.0,
    "noise_continuous_1": 0.0,
    "noise_continuous_2": 0.0,
    "noise_continuous_3": 0.0,
}
_CATEGORY_GROUP_ORACLE = {
    "causal_binary": 1.5,
    "causal_ordinal": 1.0,
    "causal_nominal": 1.0,
    "noise_categorical_1": 0.0,
}


def _onehot_oracle() -> Dict[str, float]:
    weights = dict(_SIMPLE_ORACLE)
    for group, levels in _ONEHOT_CATEGORY_GROUPS.items():
        group_weight = _CATEGORY_GROUP_ORACLE[group]
        weights.update({
            f"{group}_category_{level}": group_weight
            for level in range(levels)
        })
    return weights


_FEATURE_LOSS_WEIGHTS = {
    "simple": _SIMPLE_ORACLE,
    "range": _SIMPLE_ORACLE,
    "categorical_integer": {**_SIMPLE_ORACLE, **_CATEGORY_GROUP_ORACLE},
    "categorical_onehot": _onehot_oracle(),
    "multi_spurious": {
        "causal_continuous_1p25": 2.0,
        "causal_continuous_0p75": 1.0,
        "spurious_continuous_2p25": 0.0,
        "spurious_continuous_1p75": 0.0,
        "spurious_continuous_1p25": 0.0,
        "spurious_continuous_0p75": 0.0,
        "noise_continuous_1": 0.0,
        "noise_continuous_2": 0.0,
        "noise_continuous_3": 0.0,
    },
}


def feature_index_for(dataset: str) -> Dict[int, str]:
    return {index: name for index, name in enumerate(_FEATURE_NAMES[dataset])}


def feature_loss_weights_for(dataset: str) -> Dict[str, float]:
    return dict(_FEATURE_LOSS_WEIGHTS[dataset])


@dataclass(frozen=True)
class SyntheticConfig:
    dataset: str = "simple"
    train_size: int = 6000
    val_size: int = 2000
    test_size: int = 4000
    max_train_val_size: int = 0
    max_per_test_size: int = 0
    cache_suffix: str = ""
    dataset_seed: int = 20260401
    causal_1p25_strength: float = 1.25
    causal_0p75_strength: float = 0.75
    feature_noise: float = 1.0


class SyntheticOODDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).to(dtype=torch.float32)
        self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def remove_feature(removed_feature_indices: List[int], X: np.ndarray) -> np.ndarray:
    removed = set(removed_feature_indices)
    return X[:, [index for index in range(X.shape[1]) if index not in removed]]


def _balanced_labels(rng: np.random.Generator, size: int) -> np.ndarray:
    labels = np.arange(size, dtype=np.int64) % 2
    rng.shuffle(labels)
    return labels


def _continuous_signal(
    rng: np.random.Generator,
    label_sign: np.ndarray,
    strength: float,
    correlation: float,
    feature_noise: float,
) -> np.ndarray:
    match_probability = (1.0 + correlation) * 0.5
    signal = np.where(
        rng.random(label_sign.size) < match_probability,
        label_sign,
        -label_sign,
    )
    return strength * signal + rng.normal(0.0, feature_noise, label_sign.size)


def _sample_categories(
    rng: np.random.Generator,
    labels: np.ndarray,
    probabilities: Sequence[Sequence[float]],
) -> np.ndarray:
    categories = np.empty(labels.size, dtype=np.int64)
    for label, probs in enumerate(probabilities):
        indices = np.flatnonzero(labels == label)
        categories[indices] = rng.choice(len(probs), size=len(indices), p=probs)
    return categories


def _categorical_features(
    rng: np.random.Generator,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "causal_binary": _sample_categories(
            rng,
            labels,
            ((0.8, 0.2), (0.2, 0.8)),
        ),
        "causal_ordinal": _sample_categories(
            rng,
            labels,
            ((0.55, 0.30, 0.10, 0.05), (0.05, 0.10, 0.30, 0.55)),
        ),
        "causal_nominal": _sample_categories(
            rng,
            labels,
            ((0.40, 0.10, 0.10, 0.40), (0.10, 0.40, 0.40, 0.10)),
        ),
        "noise_categorical_1": rng.integers(0, 4, size=labels.size),
    }


def _spurious_correlations(dataset: str, environment: str) -> Tuple[float, ...]:
    train = (0.95, 0.85, 0.75, 0.65) if dataset == "multi_spurious" else (0.95,)
    if environment in {"train", "id_test"}:
        return train
    if environment == "ood_weak":
        return tuple(0.30 for _ in train)
    if environment == "ood_independent":
        return tuple(0.0 for _ in train)
    if environment == "ood_heterogeneous":
        return (0.30, 0.0, -0.50, -0.95) if len(train) == 4 else (-0.50,)
    if environment == "ood_reverse":
        return tuple(-correlation for correlation in train)
    raise ValueError(f"Unknown synthetic environment: {environment}")


def _build_environment(
    rng: np.random.Generator,
    size: int,
    config: SyntheticConfig,
    environment: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = config.dataset
    y = _balanced_labels(rng, size)
    label_sign = (2 * y - 1).astype(np.float32)
    columns = [
        config.causal_1p25_strength * label_sign
        + rng.normal(0.0, config.feature_noise, size),
        config.causal_0p75_strength * label_sign
        + rng.normal(0.0, config.feature_noise, size),
    ]

    correlations = _spurious_correlations(dataset, environment)
    if dataset == "multi_spurious":
        strengths = (2.25, 1.75, 1.25, 0.75)
        columns.extend(
            _continuous_signal(
                rng,
                label_sign,
                strength,
                correlation,
                config.feature_noise,
            )
            for strength, correlation in zip(strengths, correlations)
        )
    else:
        columns.append(
            _continuous_signal(
                rng,
                label_sign,
                2.25,
                correlations[0],
                config.feature_noise,
            )
        )

    columns.extend(rng.normal(0.0, 1.0, size=(size, 3)).T)

    if dataset in {"categorical_integer", "categorical_onehot"}:
        categories = _categorical_features(rng, y)
        if dataset == "categorical_integer":
            columns.extend(categories[name] for name in _INTEGER_CATEGORY_FEATURES)
        else:
            for group, levels in _ONEHOT_CATEGORY_GROUPS.items():
                encoded = np.eye(levels, dtype=np.float32)[categories[group]]
                columns.extend(encoded[:, level] for level in range(levels))

    X = np.column_stack(columns).astype(np.float32)
    if dataset == "range":
        X *= np.asarray((0.01, 100.0, 10.0, 0.1, 1.0, 100.0), dtype=np.float32)
    return X, y


def _standardize_with_train_stats(
    X_train: np.ndarray,
    X_other: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((X_train - mean) / std).astype(np.float32),
        [((X - mean) / std).astype(np.float32) for X in X_other],
    )


def _build_synthetic_arrays(
    config: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(config.dataset_seed)
    X_train, y_train = _build_environment(rng, config.train_size, config, "train")
    X_val, y_val = _build_environment(rng, config.val_size, config, "train")
    tests = {
        group: _build_environment(rng, config.test_size, config, group)
        for group in SYNTHETIC_TEST_GROUPS
    }

    if config.dataset != "range":
        test_arrays = [tests[group][0] for group in SYNTHETIC_TEST_GROUPS]
        X_train, standardized = _standardize_with_train_stats(
            X_train,
            [X_val, *test_arrays],
        )
        X_val, *standardized_tests = standardized
        tests = {
            group: (X, tests[group][1])
            for group, X in zip(SYNTHETIC_TEST_GROUPS, standardized_tests)
        }
    return X_train, y_train, X_val, y_val, tests


def _subsample_arrays(
    X: np.ndarray,
    y: np.ndarray,
    max_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_size <= 0 or X.shape[0] <= max_size:
        return X, y
    indices = np.random.default_rng(67).permutation(X.shape[0])[:max_size]
    return X[indices], y[indices]


def synthetic_ood_load_data(
    removed_feature_indices: List[int],
    train_val_group: str,
    test_groups: List[str],
    val_rate: float,
    config: Optional[SyntheticConfig] = None,
) -> Tuple[SyntheticOODDataset, SyntheticOODDataset, List[SyntheticOODDataset]]:
    del train_val_group
    config = config or SyntheticConfig()
    X_train, y_train, X_val, y_val, test_arrays = _build_synthetic_arrays(config)

    if config.max_train_val_size > 0:
        X_train_val = np.concatenate([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        X_train_val, y_train_val = _subsample_arrays(
            X_train_val,
            y_train_val,
            config.max_train_val_size,
        )
        val_size = int(val_rate * len(y_train_val))
        X_train, X_val = X_train_val[val_size:], X_train_val[:val_size]
        y_train, y_val = y_train_val[val_size:], y_train_val[:val_size]

    X_train = remove_feature(removed_feature_indices, X_train)
    X_val = remove_feature(removed_feature_indices, X_val)
    tests = []
    for group in test_groups:
        X_test, y_test = _subsample_arrays(
            *test_arrays[group],
            config.max_per_test_size,
        )
        tests.append(SyntheticOODDataset(
            remove_feature(removed_feature_indices, X_test),
            y_test,
        ))

    return (
        SyntheticOODDataset(X_train, y_train),
        SyntheticOODDataset(X_val, y_val),
        tests,
    )
