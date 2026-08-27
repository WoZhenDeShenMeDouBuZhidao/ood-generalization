from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from src.paths import dataset_data_dir


WHYSHIFT_DATASET_ORDER = (
    "taxi",
    "accident",
)

WHYSHIFT_TAXI_DOMAINS = (
    "nyc",
    "bog",
    "uio",
    "mex",
)

WHYSHIFT_ACCIDENT_DOMAINS = (
    "CA",
    "TX",
    "FL",
    "OR",
    "MN",
    "VA",
    "SC",
    "NY",
    "PA",
    "NC",
    "TN",
    "MI",
    "MO",
)

TAXI_FEATURES = (
    "month",
    "week",
    "weekday",
    "hour",
    "minute_oftheday",
    "distance",
    "direction",
)

ACCIDENT_FEATURES = (
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Year",
    "Month",
    "Weekday",
    "Day",
    "Hour",
    "Minute",
    "Wind_Direction_E",
    "Wind_Direction_N",
    "Wind_Direction_NE",
    "Wind_Direction_NW",
    "Wind_Direction_S",
    "Wind_Direction_SE",
    "Wind_Direction_SW",
    "Wind_Direction_Variable",
    "Wind_Direction_W",
    "Weather_Condition_Cloudy",
    "Weather_Condition_Fog",
    "Weather_Condition_Hail",
    "Weather_Condition_Rain",
    "Weather_Condition_Sand",
    "Weather_Condition_Smoke",
    "Weather_Condition_Snow",
    "Weather_Condition_Thunderstorm",
    "Weather_Condition_Tornado",
    "Weather_Condition_Windy",
    "Civil_Twilight_Night",
)


@dataclass(frozen=True)
class WhyShiftTaskConfig:
    train_val_domain: str
    removed_feature_indices: Tuple[int, ...] = ()


WHYSHIFT_TASK_CONFIGS = {
    "taxi": WhyShiftTaskConfig("bog"),
    "accident": WhyShiftTaskConfig("CA"),
}

# FEATURE_INDEX reference.
# Runtime FEATURE_INDEX values are loaded by feature_index_for().
#
# taxi:
#   0: "month"
#   1: "week"
#   2: "weekday"
#   3: "hour"
#   4: "minute_oftheday"
#   5: "distance"
#   6: "direction"
#
# accident:
#   0: "Distance(mi)"
#   1: "Temperature(F)"
#   2: "Humidity(%)"
#   3: "Pressure(in)"
#   4: "Visibility(mi)"
#   5: "Wind_Speed(mph)"
#   6: "Precipitation(in)"
#   7: "Amenity"
#   8: "Bump"
#   9: "Crossing"
#   10: "Give_Way"
#   11: "Junction"
#   12: "No_Exit"
#   13: "Railway"
#   14: "Roundabout"
#   15: "Station"
#   16: "Stop"
#   17: "Traffic_Calming"
#   18: "Traffic_Signal"
#   19: "Year"
#   20: "Month"
#   21: "Weekday"
#   22: "Day"
#   23: "Hour"
#   24: "Minute"
#   25: "Wind_Direction_E"
#   26: "Wind_Direction_N"
#   27: "Wind_Direction_NE"
#   28: "Wind_Direction_NW"
#   29: "Wind_Direction_S"
#   30: "Wind_Direction_SE"
#   31: "Wind_Direction_SW"
#   32: "Wind_Direction_Variable"
#   33: "Wind_Direction_W"
#   34: "Weather_Condition_Cloudy"
#   35: "Weather_Condition_Fog"
#   36: "Weather_Condition_Hail"
#   37: "Weather_Condition_Rain"
#   38: "Weather_Condition_Sand"
#   39: "Weather_Condition_Smoke"
#   40: "Weather_Condition_Snow"
#   41: "Weather_Condition_Thunderstorm"
#   42: "Weather_Condition_Tornado"
#   43: "Weather_Condition_Windy"
#   44: "Civil_Twilight_Night"


def raw_dir_for(dataset: str) -> Path:
    return dataset_data_dir("whyshift", dataset) / "raw"


def taxi_domain_from_csv(path: Path) -> str:
    if path.stem == "train":
        return "nyc"
    return path.stem.removesuffix("_clean")


def taxi_domains_from_raw() -> Tuple[str, ...]:
    domains = {
        taxi_domain_from_csv(path)
        for path in raw_dir_for("taxi").glob("*.csv")
    }
    if not domains:
        return WHYSHIFT_TAXI_DOMAINS
    return tuple(domain for domain in WHYSHIFT_TAXI_DOMAINS if domain in domains)


def feature_index_for(dataset: str) -> Dict[int, str]:
    if dataset == "taxi":
        return {idx: feature for idx, feature in enumerate(TAXI_FEATURES)}
    if dataset == "accident":
        return {idx: feature for idx, feature in enumerate(ACCIDENT_FEATURES)}
    raise ValueError(f"Unknown WHYSHIFT dataset: {dataset}")


def test_domains_for(dataset: str) -> Tuple[str, ...]:
    config = WHYSHIFT_TASK_CONFIGS[dataset]
    if dataset == "taxi":
        base_domains = taxi_domains_from_raw()
    else:
        base_domains = WHYSHIFT_ACCIDENT_DOMAINS
    return tuple(domain for domain in base_domains if domain != config.train_val_domain)
