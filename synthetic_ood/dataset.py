import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_INDEX = {
    0: "causal_main",
    1: "causal_aux",
    2: "spurious_main",
    3: "noise_1",
    4: "noise_2",
    5: "noise_3",
}


@dataclass(frozen=True)
class SyntheticConfig:
    train_size: int = 6000
    val_size: int = 2000
    test_size: int = 4000
    dataset_seed: int = 20260401
    causal_main_strength: float = 1.25
    causal_aux_strength: float = 0.75
    spurious_strength: float = 2.25
    feature_noise: float = 1.0
    train_spurious_corr: float = 0.95
    ood_spurious_corr: float = -0.95


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
    keep_indices = [i for i in range(X.shape[1]) if i not in removed]
    return X[:, keep_indices]


def _build_environment(
    rng: np.random.Generator,
    size: int,
    *,
    causal_main_strength: float,
    causal_aux_strength: float,
    spurious_strength: float,
    spurious_corr: float,
    feature_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=size, dtype=np.int64)
    label_sign = (2 * y - 1).astype(np.float32)

    causal_main = causal_main_strength * label_sign + rng.normal(0.0, feature_noise, size)
    causal_aux = causal_aux_strength * label_sign + rng.normal(0.0, feature_noise, size)

    spurious_signal = label_sign.copy()
    flip_mask = rng.random(size) > abs(spurious_corr)
    spurious_signal[flip_mask] *= -1.0
    if spurious_corr < 0:
        spurious_signal *= -1.0
    spurious_main = spurious_strength * spurious_signal + rng.normal(0.0, feature_noise, size)

    noise = rng.normal(0.0, 1.0, size=(size, 3))

    X = np.column_stack(
        [causal_main, causal_aux, spurious_main, noise[:, 0], noise[:, 1], noise[:, 2]]
    ).astype(np.float32)
    return X, y


def _standardize_with_train_stats(
    X_train: np.ndarray, X_other: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_std = (X_train - mean) / std
    X_other_std = [(X - mean) / std for X in X_other]
    return X_train_std, X_other_std


def _build_synthetic_arrays(
    config: SyntheticConfig = SyntheticConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(config.dataset_seed)
    X_train, y_train = _build_environment(
        rng,
        config.train_size,
        causal_main_strength=config.causal_main_strength,
        causal_aux_strength=config.causal_aux_strength,
        spurious_strength=config.spurious_strength,
        spurious_corr=config.train_spurious_corr,
        feature_noise=config.feature_noise,
    )
    X_val, y_val = _build_environment(
        rng,
        config.val_size,
        causal_main_strength=config.causal_main_strength,
        causal_aux_strength=config.causal_aux_strength,
        spurious_strength=config.spurious_strength,
        spurious_corr=config.train_spurious_corr,
        feature_noise=config.feature_noise,
    )
    X_test_ood, y_test_ood = _build_environment(
        rng,
        config.test_size,
        causal_main_strength=config.causal_main_strength,
        causal_aux_strength=config.causal_aux_strength,
        spurious_strength=config.spurious_strength,
        spurious_corr=config.ood_spurious_corr,
        feature_noise=config.feature_noise,
    )

    X_train, (X_val, X_test_ood) = _standardize_with_train_stats(X_train, [X_val, X_test_ood])
    return X_train, y_train, X_val, y_val, X_test_ood, y_test_ood


def synthetic_ood_load_train_val(
    removed_feature_indices: List[int],
    group: str,
    val_rate: float,
    config: SyntheticConfig = SyntheticConfig(),
) -> Tuple[SyntheticOODDataset, SyntheticOODDataset]:
    del group
    del val_rate

    X_train, y_train, X_val, y_val, _, _ = _build_synthetic_arrays(config)

    X_train = remove_feature(removed_feature_indices, X_train)
    X_val = remove_feature(removed_feature_indices, X_val)

    train = SyntheticOODDataset(X_train, y_train)
    val = SyntheticOODDataset(X_val, y_val)
    return train, val


def synthetic_ood_load_tests(
    removed_feature_indices: List[int],
    groups: List[str],
    config: SyntheticConfig = SyntheticConfig(),
) -> List[SyntheticOODDataset]:
    del groups

    _, _, _, _, X_test_ood, y_test_ood = _build_synthetic_arrays(config)
    X_test_ood = remove_feature(removed_feature_indices, X_test_ood)
    return [SyntheticOODDataset(X_test_ood, y_test_ood)]
