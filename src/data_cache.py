import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from src.paths import dataset_cache_dir


def get_config_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def removed_features_cache_token(removed_feature_indices: List[int]) -> str:
    removed = sorted(set(removed_feature_indices))
    if not removed:
        return "rm_none"
    return "rm_" + "-".join(str(idx) for idx in removed)


def data_cache_dir(
    benchmark: str,
    dataset: str,
    removed_feature_indices: List[int],
    dataset_config: Optional[Any],
) -> Path:
    data_dir = dataset_cache_dir(benchmark, dataset)
    removed_token = removed_features_cache_token(removed_feature_indices)
    cache_suffix = get_config_value(dataset_config, "cache_suffix", "")
    if cache_suffix:
        return data_dir / f"{removed_token}__{cache_suffix}"

    if benchmark != "acs":
        return data_dir / removed_token

    resampling = int(bool(get_config_value(dataset_config, "resampling", False)))
    standardize = int(bool(get_config_value(dataset_config, "standardize", False)))
    max_train_val_size = int(get_config_value(dataset_config, "max_train_val_size", 0))
    max_per_test_size = int(get_config_value(dataset_config, "max_per_test_size", 0))
    cache_name = "__".join([
        removed_token,
        f"rs{resampling}",
        f"std{standardize}",
        f"trv{max_train_val_size}",
        f"test{max_per_test_size}",
    ])
    return data_dir / cache_name


def load_or_build_dataset_cache(
    benchmark: str,
    dataset: str,
    removed_feature_indices: List[int],
    dataset_config: Optional[Any],
    build_fn: Callable[[], Tuple[Any, Any, list]],
) -> Tuple[Any, Any, list, Path, bool]:
    data_dir = data_cache_dir(benchmark, dataset, removed_feature_indices, dataset_config)
    cache_hit = (data_dir / "train.pkl").is_file()
    if cache_hit:
        try:
            with open(data_dir / "train.pkl", "rb") as fp:
                train = pickle.load(fp)
            with open(data_dir / "val.pkl", "rb") as fp:
                val = pickle.load(fp)
            with open(data_dir / "tests.pkl", "rb") as fp:
                tests = pickle.load(fp)
            return train, val, tests, data_dir, cache_hit
        except (AttributeError, ImportError, ModuleNotFoundError):
            cache_hit = False

    train, val, tests = build_fn()
    os.makedirs(data_dir, exist_ok=True)
    with open(data_dir / "train.pkl", "wb") as fp:
        pickle.dump(train, fp)
    with open(data_dir / "val.pkl", "wb") as fp:
        pickle.dump(val, fp)
    with open(data_dir / "tests.pkl", "wb") as fp:
        pickle.dump(tests, fp)
    return train, val, tests, data_dir, cache_hit
