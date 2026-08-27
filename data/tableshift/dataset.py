from dataclasses import replace
import os
from pathlib import Path
import zipfile
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from data.tableshift.config import TABLESHIFT_DATASET_ORDER, raw_dir_for
from src.data_cache import get_config_value


TABLESHIFT_TASKS = set(TABLESHIFT_DATASET_ORDER)


class TableShiftDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, feature_names: Sequence[str]):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.int64))
        self.feature_names = list(feature_names)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def _get_tableshift_dataset(dataset: str, categorical_encoding: str):
    try:
        from tableshift import get_dataset
        from tableshift.configs.benchmark_configs import BENCHMARK_CONFIGS
    except ImportError as exc:
        raise ImportError(
            "TableShift support requires the optional `tableshift` package. "
            "Install it from the official repository before running "
            "`--benchmark tableshift`."
        ) from exc

    _patch_tableshift_kbins_discretizer()
    _patch_tableshift_xpt_reader()
    if dataset in {"nhanes_cholesterol", "nhanes_lead"}:
        _patch_nhanes_downloads()

    benchmark_config = BENCHMARK_CONFIGS[dataset]
    preprocessor_config = benchmark_config.preprocessor_config
    if categorical_encoding != "auto":
        tableshift_encoding = {
            "integer": "label_encode",
            "one_hot": "one_hot",
        }[categorical_encoding]
        preprocessor_config = replace(
            preprocessor_config,
            categorical_features=tableshift_encoding,
        )
    domain_split_col = getattr(benchmark_config.splitter, "domain_split_varname", None)
    passthrough_columns = preprocessor_config.passthrough_columns
    if domain_split_col and passthrough_columns != "all":
        passthrough_columns = list(passthrough_columns or ())
        if domain_split_col not in passthrough_columns:
            passthrough_columns.append(domain_split_col)
        preprocessor_config = replace(
            preprocessor_config,
            passthrough_columns=passthrough_columns,
        )

    raw_dir = raw_dir_for(dataset)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "college_scorecard":
        _prepare_college_scorecard_manual_cache(raw_dir)
    if dataset == "anes":
        _prepare_anes_manual_cache(raw_dir)
    return get_dataset(
        dataset,
        cache_dir=str(raw_dir),
        preprocessor_config=preprocessor_config,
    )


def _patch_tableshift_kbins_discretizer():
    from sklearn.utils.validation import check_array
    from tableshift.core.discretization import KBinsDiscretizer

    if hasattr(KBinsDiscretizer, "_validate_data"):
        return

    def _validate_data(
        self,
        X,
        dtype=None,
        force_all_finite=True,
        reset=True,
        copy=False,
        **kwargs,
    ):
        del kwargs
        try:
            X_checked = check_array(
                X,
                dtype=dtype,
                ensure_all_finite=force_all_finite,
                copy=copy,
            )
        except TypeError:
            X_checked = check_array(
                X,
                dtype=dtype,
                force_all_finite=force_all_finite,
                copy=copy,
            )

        if reset:
            self.n_features_in_ = X_checked.shape[1]
        elif hasattr(self, "n_features_in_") and X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but KBinsDiscretizer "
                f"is expecting {self.n_features_in_} features."
            )
        return X_checked

    KBinsDiscretizer._validate_data = _validate_data


def _patch_tableshift_xpt_reader():
    from tableshift.core import utils

    if getattr(utils.read_xpt, "_ood_patch", False):
        return

    def _read_xpt(fp):
        df = pd.read_sas(fp, format="xport")
        for column in df.select_dtypes(include="object").columns:
            df[column] = df[column].map(
                lambda value: value.decode("utf-8", errors="ignore").strip()
                if isinstance(value, (bytes, bytearray))
                else value
            )
        return df

    _read_xpt._ood_patch = True
    utils.read_xpt = _read_xpt


def _patch_nhanes_downloads():
    from tableshift.core.data_source import NHANESDataSource
    from tableshift.datasets.nhanes import get_nhanes_data_sources

    if getattr(NHANESDataSource._download_if_not_cached, "_ood_patch", False):
        return

    def _add_year_suffix(filename: str, year: str) -> str:
        stem, extension = filename.rsplit(".", 1)
        return f"{stem}{year}.{extension}"

    def _is_valid_xpt(path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as handle:
            return handle.read(13) == b"HEADER RECORD"

    def _download_if_not_cached(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        sources = get_nhanes_data_sources(self.nhanes_task, self.years)
        for year, urls in sources.items():
            for old_url in urls:
                filename = os.path.basename(old_url)
                destfile = _add_year_suffix(filename, str(year))
                dest_path = Path(self.cache_dir) / destfile
                if _is_valid_xpt(dest_path):
                    continue

                url = (
                    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
                    f"{year}/DataFiles/{filename}"
                )
                response = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=120,
                )
                response.raise_for_status()
                if not response.content.startswith(b"HEADER RECORD"):
                    raise RuntimeError(f"NHANES download did not return XPT data: {url}")
                dest_path.write_bytes(response.content)

    _download_if_not_cached._ood_patch = True
    NHANESDataSource._download_if_not_cached = _download_if_not_cached


def _prepare_college_scorecard_manual_cache(raw_dir):
    expected_csv = raw_dir / "kaggle" / "college-scorecard" / "Scorecard.csv"
    root_csv = raw_dir / "Scorecard.csv"
    if root_csv.exists() and not expected_csv.exists():
        expected_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            expected_csv.symlink_to(root_csv.resolve())
        except OSError:
            _patch_college_scorecard_load_path(root_csv)
            _skip_college_scorecard_download_if_cached(root_csv)
            return

    if expected_csv.exists():
        _skip_college_scorecard_download_if_cached(expected_csv)
        return

    zip_path = raw_dir / "college-scorecard.zip"
    if zip_path.exists():
        expected_csv.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(expected_csv.parent)
        if expected_csv.exists():
            _skip_college_scorecard_download_if_cached(expected_csv)


def _skip_college_scorecard_download_if_cached(expected_csv):
    from tableshift.core.data_source import CollegeScorecardDataSource

    original_download = CollegeScorecardDataSource._download_if_not_cached

    def _download_if_not_cached(self):
        if expected_csv.exists():
            return
        return original_download(self)

    CollegeScorecardDataSource._download_if_not_cached = _download_if_not_cached


def _patch_college_scorecard_load_path(csv_path):
    import pandas as pd
    from tableshift.core.data_source import CollegeScorecardDataSource

    def _load_data(self):
        return pd.read_csv(csv_path)

    CollegeScorecardDataSource._load_data = _load_data


def _prepare_anes_manual_cache(raw_dir):
    expected_csv = (
        raw_dir
        / "anes_timeseries_cdf_csv_20220916"
        / "anes_timeseries_cdf_csv_20220916.csv"
    )
    if expected_csv.exists():
        return

    for zip_path in sorted(raw_dir.glob("anes_timeseries_cdf_csv_*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    candidates = sorted(raw_dir.glob("anes_timeseries_cdf_csv_*.csv"))
    candidates += sorted(raw_dir.glob("anes_timeseries_cdf_csv_*/*.csv"))
    if not candidates:
        return

    csv_path = candidates[-1]
    expected_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        expected_csv.symlink_to(csv_path.resolve())
    except OSError:
        _patch_anes_load_path(csv_path)


def _patch_anes_load_path(csv_path):
    import pandas as pd
    from tableshift.core.data_source import ANESDataSource

    def _load_data(self):
        df = pd.read_csv(csv_path, low_memory=False, na_values=(" "))
        if self.years:
            df = df[df["VCF0004"].isin(self.years)]
        return df

    ANESDataSource._load_data = _load_data


def _subsample_frame(
    X: pd.DataFrame,
    y: pd.Series,
    max_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if max_size <= 0 or len(X) <= max_size:
        return X, y
    indices = np.random.default_rng(random_state).permutation(len(X))[:max_size]
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def _load_split_frame(
    tableshift_dataset,
    split: str,
    max_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    X, y, _, _ = tableshift_dataset.get_pandas(split)
    X = X.copy()
    X.columns = [str(column) for column in X.columns]
    y = pd.Series(y).reset_index(drop=True)
    return _subsample_frame(X.reset_index(drop=True), y, max_size, random_state)


def _frame_to_dataset(
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    removed_feature_indices: List[int],
) -> TableShiftDataset:
    missing = [feature for feature in feature_names if feature not in X.columns]
    if missing:
        raise ValueError(f"{split} is missing TableShift features: {missing[:5]}")

    X = X.loc[:, list(feature_names)]
    labels = sorted(set(y.dropna().astype(int).tolist()))
    if not set(labels).issubset({0, 1}):
        raise ValueError(f"Expected binary labels for TableShift split `{split}`, got {labels}.")

    kept_indices = [
        idx for idx in range(len(feature_names))
        if idx not in set(removed_feature_indices)
    ]
    kept_feature_names = [feature_names[idx] for idx in kept_indices]
    X_np = X.iloc[:, kept_indices].to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.int64)
    return TableShiftDataset(X_np, y_np, kept_feature_names)


def tableshift_load_data(
    task_name: str,
    removed_feature_indices: List[int],
    train_val_group: str,
    test_groups: List[str],
    val_rate: float,
    config: Optional[Any] = None,
) -> Tuple[TableShiftDataset, TableShiftDataset, List[TableShiftDataset]]:
    del train_val_group, val_rate
    max_train_val_size = int(get_config_value(config, "max_train_val_size", 0))
    max_per_test_size = int(get_config_value(config, "max_per_test_size", 0))
    random_state = int(get_config_value(config, "random_state", 42))
    categorical_encoding = str(get_config_value(config, "categorical_encoding", "auto"))
    if categorical_encoding not in {"auto", "integer", "one_hot"}:
        raise ValueError(f"Unsupported categorical_encoding: {categorical_encoding}")
    tableshift_dataset = _get_tableshift_dataset(task_name, categorical_encoding)

    X_train, y_train = _load_split_frame(
        tableshift_dataset,
        "train",
        max_train_val_size,
        random_state,
    )
    feature_names = tuple(sorted(X_train.columns))
    train = _frame_to_dataset(
        "train",
        X_train,
        y_train,
        feature_names,
        removed_feature_indices,
    )

    X_val, y_val = _load_split_frame(
        tableshift_dataset,
        "validation",
        0,
        random_state + 1,
    )
    val = _frame_to_dataset(
        "validation",
        X_val,
        y_val,
        feature_names,
        removed_feature_indices,
    )

    tests = []
    for idx, split in enumerate(test_groups):
        X_test, y_test = _load_split_frame(
            tableshift_dataset,
            split,
            max_per_test_size,
            random_state + 2 + idx,
        )
        tests.append(
            _frame_to_dataset(
                split,
                X_test,
                y_test,
                feature_names,
                removed_feature_indices,
            )
        )
    return train, val, tests
