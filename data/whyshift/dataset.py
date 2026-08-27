from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from data.whyshift.config import (
    ACCIDENT_FEATURES,
    TAXI_FEATURES,
    WHYSHIFT_DATASET_ORDER,
    raw_dir_for,
    taxi_domain_from_csv,
)
from src.data_cache import get_config_value


WHYSHIFT_TASKS = set(WHYSHIFT_DATASET_ORDER)

TAXI_CATEGORIES = {
    "month": tuple(range(1, 13)),
    "week": tuple(range(1, 54)),
    "weekday": tuple(range(7)),
    "hour": tuple(range(24)),
}


class WhyShiftDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, feature_names: Sequence[str]):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.int64))
        self.feature_names = list(feature_names)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def _csv_by_domain(dataset: str, domain: str):
    for path in sorted(raw_dir_for(dataset).glob("*.csv")):
        if taxi_domain_from_csv(path) == domain:
            return path
    raise FileNotFoundError(f"No {dataset} CSV for domain {domain} in {raw_dir_for(dataset)}.")


def _single_csv(dataset: str):
    paths = sorted(raw_dir_for(dataset).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV found in {raw_dir_for(dataset)}.")
    return paths[0]


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X).astype(np.float32)


def _remove_features(
    X: np.ndarray,
    feature_names: Sequence[str],
    removed_feature_indices: List[int],
) -> Tuple[np.ndarray, List[str]]:
    keep_indices = [
        idx for idx in range(X.shape[1])
        if idx not in set(removed_feature_indices)
    ]
    return X[:, keep_indices], [feature_names[idx] for idx in keep_indices]


def _split_train_val(
    X: np.ndarray,
    Y: np.ndarray,
    val_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    val_idx = int(val_rate * len(indices))
    return X[val_idx:], Y[val_idx:], X[:val_idx], Y[:val_idx]


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


def _haversine_distance(
    pickup_latitude: pd.Series,
    pickup_longitude: pd.Series,
    dropoff_latitude: pd.Series,
    dropoff_longitude: pd.Series,
) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(
        np.radians,
        (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude),
    )
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    distance = (
        np.sin(lat_delta * 0.5) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(distance))


def _direction(
    pickup_latitude: pd.Series,
    pickup_longitude: pd.Series,
    dropoff_latitude: pd.Series,
    dropoff_longitude: pd.Series,
) -> np.ndarray:
    lng_delta_rad = np.radians(dropoff_longitude - pickup_longitude)
    lat1, lng1, lat2, lng2 = map(
        np.radians,
        (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude),
    )
    del lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def _load_taxi_domain(
    domain: str,
    need_preprocess: bool,
    categorical_encoding: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(_csv_by_domain("taxi", domain))
    df = df[(df["trip_duration"] < 5900)]
    df = df[(df["pickup_longitude"] > -110)]
    df = df[(df["pickup_latitude"] < 50)]
    df = df.drop(
        columns=["store_and_fwd_flag", "vendor_id", "dropoff_datetime"],
        errors="ignore",
    )

    pickup_datetime = pd.to_datetime(df["pickup_datetime"])
    df["month"] = pickup_datetime.dt.month
    df["week"] = pickup_datetime.dt.isocalendar().week.astype(int)
    df["weekday"] = pickup_datetime.dt.weekday
    df["hour"] = pickup_datetime.dt.hour
    df["minute_oftheday"] = df["hour"] * 60 + pickup_datetime.dt.minute
    df["distance"] = _haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["direction"] = _direction(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df = df[(df["distance"] < 200)]
    df["speed"] = df["distance"] / df["trip_duration"]
    df = df[(df["speed"] < 30)]

    y = (df["trip_duration"] >= 900).to_numpy(dtype=np.int64)
    X_columns = []
    feature_names = []
    for feature_name in TAXI_FEATURES:
        values = df[feature_name].to_numpy()
        if categorical_encoding == "one_hot" and feature_name in TAXI_CATEGORIES:
            categories = np.asarray(TAXI_CATEGORIES[feature_name])
            X_columns.append((values[:, None] == categories[None, :]).astype(np.float32))
            feature_names.extend(f"{feature_name}_{category}" for category in categories)
        else:
            X_columns.append(values[:, None].astype(np.float32))
            feature_names.append(feature_name)
    X = np.concatenate(X_columns, axis=1)
    if need_preprocess:
        X = _standardize(X)
    return X, y, feature_names


def _simplify_accident_weather(df: pd.DataFrame) -> None:
    df.loc[df["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
    df.loc[df["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
    df.loc[df["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
    df.loc[df["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
    df.loc[df["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
    df.loc[df["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
    df.loc[df["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
    df.loc[df["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
    df.loc[df["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
    df.loc[df["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
    df.loc[df["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan


def _preprocess_accident(random_state: int, categorical_encoding: str) -> pd.DataFrame:
    df = pd.read_csv(_single_csv("accident"), low_memory=False)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Start_Time"])
    df["Year"] = df["Start_Time"].dt.year
    df["Month"] = df["Start_Time"].dt.month
    df["Weekday"] = df["Start_Time"].dt.weekday
    df["Day"] = df["Start_Time"].dt.day
    df["Hour"] = df["Start_Time"].dt.hour
    df["Minute"] = df["Start_Time"].dt.minute
    df = df.drop(
        columns=[
            "ID",
            "Source",
            "Start_Time",
            "End_Time",
            "End_Lat",
            "End_Lng",
            "Description",
            "Number",
            "Street",
            "County",
            "Zipcode",
            "City",
            "Country",
            "Timezone",
            "Airport_Code",
            "Weather_Timestamp",
            "Wind_Chill(F)",
            "Turning_Loop",
            "Sunrise_Sunset",
            "Nautical_Twilight",
            "Astronomical_Twilight",
        ],
        errors="ignore",
    )
    df = df.drop_duplicates()
    if "Side" in df.columns:
        df = df[(df["Side"] != " ")]
    df = df[(df["Pressure(in)"] != 0)]
    df = df[(df["Visibility(mi)"] != 0)]

    _simplify_accident_weather(df)
    df.loc[df["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
    df.loc[df["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
    df.loc[df["Wind_Direction"] == "East", "Wind_Direction"] = "E"
    df.loc[df["Wind_Direction"] == "North", "Wind_Direction"] = "N"
    df.loc[df["Wind_Direction"] == "South", "Wind_Direction"] = "S"
    df.loc[df["Wind_Direction"] == "West", "Wind_Direction"] = "W"

    features_to_fill = [
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
    ]
    df[features_to_fill] = df[features_to_fill].fillna(df[features_to_fill].mean())
    df = df.dropna()
    df["Wind_Direction"] = df["Wind_Direction"].map(
        lambda value: value if len(value) != 3 else value[1:],
        na_action="ignore",
    )

    df = df[df["Severity"].isin([2, 3])]
    min_class_size = min(len(df[df["Severity"] == 2]), len(df[df["Severity"] == 3]))
    df = pd.concat([
        df[df["Severity"] == severity].sample(min_class_size, random_state=random_state)
        for severity in [2, 3]
    ])
    df["Severity"] -= 2

    minmax_features = [
        "Temperature(F)",
        "Distance(mi)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Precipitation(in)",
        "Start_Lng",
        "Start_Lat",
        "Year",
        "Month",
        "Weekday",
        "Day",
        "Hour",
        "Minute",
    ]
    df[minmax_features] = MinMaxScaler().fit_transform(df[minmax_features])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    categorical_features = [
        column
        for column in ["Side", "Wind_Direction", "Weather_Condition", "Civil_Twilight"]
        if column in df.columns
    ]
    if categorical_encoding == "integer":
        for column in categorical_features:
            categories = sorted(df[column].unique(), key=str)
            if len(categories) <= 1:
                df = df.drop(columns=[column])
                continue
            df[column] = pd.Categorical(df[column], categories=categories).codes
        return df
    return pd.get_dummies(df, columns=categorical_features, drop_first=True)


def _load_accident_domain(
    frame: pd.DataFrame,
    domain: str,
    need_preprocess: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    domain_frame = frame[frame["State"] == domain]
    y = domain_frame["Severity"].to_numpy(dtype=np.int64)
    X_frame = domain_frame.drop(columns=["Severity", "State", "Start_Lat", "Start_Lng"])
    feature_names = list(X_frame.columns)
    X = X_frame.to_numpy(dtype=np.float32)
    if need_preprocess:
        X = _standardize(X)
    return X, y, feature_names


def _load_domain(
    task_name: str,
    domain: str,
    need_preprocess: bool,
    categorical_encoding: str,
    accident_frame: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if task_name == "taxi":
        return _load_taxi_domain(domain, need_preprocess, categorical_encoding)
    return _load_accident_domain(accident_frame, domain, need_preprocess)


def _expected_feature_names(task_name: str) -> List[str]:
    if task_name == "taxi":
        return list(TAXI_FEATURES)
    return list(ACCIDENT_FEATURES)


def whyshift_load_data(
    task_name: str,
    removed_feature_indices: List[int],
    train_val_domain: str,
    test_domains: List[str],
    val_rate: float,
    config: Optional[Any] = None,
) -> Tuple[WhyShiftDataset, WhyShiftDataset, List[WhyShiftDataset]]:
    need_preprocess = bool(get_config_value(config, "need_preprocess", True))
    max_train_val_size = int(get_config_value(config, "max_train_val_size", 0))
    max_per_test_size = int(get_config_value(config, "max_per_test_size", 0))
    random_state = int(get_config_value(config, "random_state", 42))
    categorical_encoding = str(get_config_value(config, "categorical_encoding", "auto"))
    if categorical_encoding not in {"auto", "integer", "one_hot"}:
        raise ValueError(f"Unsupported categorical_encoding: {categorical_encoding}")
    accident_frame = (
        _preprocess_accident(random_state, categorical_encoding)
        if task_name == "accident"
        else None
    )

    X_train_val, y_train_val, feature_names = _load_domain(
        task_name,
        train_val_domain,
        need_preprocess,
        categorical_encoding,
        accident_frame,
    )
    expected_feature_names = _expected_feature_names(task_name)
    if categorical_encoding == "auto" and feature_names != expected_feature_names:
        raise ValueError(
            f"{task_name} feature names do not match data/whyshift/config.py. "
            "Update the explicit FEATURE_INDEX reference before running GPT ranking or training."
        )
    X_train_val, feature_names = _remove_features(
        X_train_val,
        feature_names,
        removed_feature_indices,
    )
    X_train_val, y_train_val = _subsample_arrays(
        X_train_val,
        y_train_val,
        max_train_val_size,
    )
    X_train, y_train, X_val, y_val = _split_train_val(X_train_val, y_train_val, val_rate)

    tests = []
    for domain in test_domains:
        X_test, y_test, test_feature_names = _load_domain(
            task_name,
            domain,
            need_preprocess,
            categorical_encoding,
            accident_frame,
        )
        X_test, test_feature_names = _remove_features(
            X_test,
            test_feature_names,
            removed_feature_indices,
        )
        X_test, y_test = _subsample_arrays(X_test, y_test, max_per_test_size)
        if test_feature_names != feature_names:
            raise ValueError(f"{task_name} feature mismatch between train and test domains.")
        tests.append(WhyShiftDataset(X_test, y_test, feature_names))

    train = WhyShiftDataset(X_train, y_train, feature_names)
    val = WhyShiftDataset(X_val, y_val, feature_names)
    return train, val, tests
