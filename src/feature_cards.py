import re
from typing import Any, Dict, List, Mapping

from src.semantic_features import longest_prefix_feature, whyshift_semantic_name


MAX_VALUE_MAPPING_ITEMS = 12
MAX_TEXT_LENGTH = 260


ACS_FEATURE_CARDS = {
    "AGEP": {
        "type": "integer",
        "description": "Age of the person in years.",
        "values": "ACS age code; older ages may be topcoded.",
    },
    "COW": {
        "type": "categorical",
        "description": "Class of worker.",
        "values": "ACS employment class code, such as private, government, self-employed, or unpaid family worker.",
    },
    "SCHL": {
        "type": "ordinal_categorical",
        "description": "Educational attainment.",
        "values": "ACS education level code; larger codes generally indicate more education.",
    },
    "MAR": {
        "type": "categorical",
        "description": "Marital status.",
        "values": "ACS marital status code.",
    },
    "OCCP": {
        "type": "categorical",
        "description": "Occupation code.",
        "values": "Detailed ACS occupation category; many possible occupations.",
    },
    "POBP": {
        "type": "categorical",
        "description": "Place of birth.",
        "values": "Geographic birthplace recode.",
    },
    "RELP": {
        "type": "categorical",
        "description": "Relationship to householder.",
        "values": "ACS household relationship code.",
    },
    "WKHP": {
        "type": "integer",
        "description": "Usual hours worked per week during the past 12 months.",
        "values": "0 or positive weekly work hours.",
    },
    "SEX": {
        "type": "categorical",
        "description": "Sex reported in ACS.",
        "values": "1=male, 2=female.",
    },
    "RAC1P": {
        "type": "categorical",
        "description": "Detailed race recode.",
        "values": "ACS race category code.",
    },
    "DIS": {
        "type": "boolean_categorical",
        "description": "Disability recode.",
        "values": "1=with a disability, 2=without a disability.",
    },
    "ESP": {
        "type": "categorical",
        "description": "Employment status of parents.",
        "values": "ACS family/parent employment code, mainly defined for children.",
    },
    "CIT": {
        "type": "categorical",
        "description": "Citizenship status.",
        "values": "US-born, US territory-born, born abroad to American parents, naturalized citizen, or not a citizen.",
    },
    "MIG": {
        "type": "categorical",
        "description": "Mobility status one year ago.",
        "values": "Whether and where the person lived one year ago.",
    },
    "MIL": {
        "type": "categorical",
        "description": "Military service status.",
        "values": "ACS military service category.",
    },
    "ANC": {
        "type": "categorical",
        "description": "Ancestry recode.",
        "values": "ACS ancestry category.",
    },
    "NATIVITY": {
        "type": "categorical",
        "description": "Nativity recode.",
        "values": "Native-born or foreign-born.",
    },
    "DEAR": {
        "type": "boolean_categorical",
        "description": "Hearing difficulty.",
        "values": "1=yes, 2=no.",
    },
    "DEYE": {
        "type": "boolean_categorical",
        "description": "Vision difficulty.",
        "values": "1=yes, 2=no.",
    },
    "DREM": {
        "type": "boolean_categorical",
        "description": "Cognitive difficulty remembering, concentrating, or making decisions.",
        "values": "1=yes, 2=no.",
    },
    "GCL": {
        "type": "categorical",
        "description": "Grandparents living with grandchildren and responsibility status.",
        "values": "ACS grandparent care category.",
    },
    "RACAIAN": {
        "type": "boolean_categorical",
        "description": "American Indian or Alaska Native race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACASN": {
        "type": "boolean_categorical",
        "description": "Asian race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACBLK": {
        "type": "boolean_categorical",
        "description": "Black or African American race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACNH": {
        "type": "boolean_categorical",
        "description": "Native Hawaiian race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACPI": {
        "type": "boolean_categorical",
        "description": "Pacific Islander race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACSOR": {
        "type": "boolean_categorical",
        "description": "Some other race recode indicator.",
        "values": "Race indicator code.",
    },
    "RACWHT": {
        "type": "boolean_categorical",
        "description": "White race recode indicator.",
        "values": "Race indicator code.",
    },
    "PINCP": {
        "type": "continuous",
        "description": "Total person's income.",
        "values": "Dollar income amount; may be negative, zero, or positive.",
    },
    "ESR": {
        "type": "categorical",
        "description": "Employment status recode.",
        "values": "Employed, unemployed, armed forces, or not in labor force categories.",
    },
    "ST": {
        "type": "categorical",
        "description": "State code.",
        "values": "Geographic state identifier; likely encodes train/test domain.",
    },
    "FER": {
        "type": "categorical",
        "description": "Gave birth to a child within the past 12 months.",
        "values": "ACS fertility code; includes not applicable categories.",
    },
    "PUMA": {
        "type": "categorical",
        "description": "Public Use Microdata Area code.",
        "values": "Fine geographic identifier within state; likely encodes local domain.",
    },
    "JWMNP": {
        "type": "integer",
        "description": "Travel time to work in minutes.",
        "values": "Number of commuting minutes.",
    },
    "JWTR": {
        "type": "categorical",
        "description": "Means of transportation to work.",
        "values": "ACS commute mode category.",
    },
    "POWPUMA": {
        "type": "categorical",
        "description": "Place-of-work PUMA code.",
        "values": "Workplace geographic identifier; likely encodes local domain.",
    },
    "POVPIP": {
        "type": "continuous",
        "description": "Income-to-poverty ratio.",
        "values": "Poverty ratio percentage/code; larger values indicate higher income relative to poverty threshold.",
    },
}


WHYSHIFT_FEATURE_CARDS = {
    "taxi": {
        "month": {
            "type": "ordinal_categorical",
            "description": "Pickup month.",
            "values": "1-12.",
        },
        "week": {
            "type": "ordinal_categorical",
            "description": "Pickup week of year.",
            "values": "1-53.",
        },
        "weekday": {
            "type": "ordinal_categorical",
            "description": "Pickup day of week.",
            "values": "0-6.",
        },
        "hour": {
            "type": "ordinal_categorical",
            "description": "Pickup hour of day.",
            "values": "0-23.",
        },
        "minute_oftheday": {
            "type": "integer",
            "description": "Pickup time represented as minutes since midnight.",
            "values": "0-1439.",
        },
        "distance": {
            "type": "continuous",
            "description": "Trip distance derived from pickup/dropoff locations.",
            "values": "Non-negative distance.",
        },
        "direction": {
            "type": "continuous",
            "description": "Trip travel direction or bearing.",
            "values": "Angular direction in degrees.",
        },
    },
    "accident": {
        "Distance(mi)": {
            "type": "continuous",
            "description": "Distance affected by the accident.",
            "values": "Miles.",
        },
        "Temperature(F)": {
            "type": "continuous",
            "description": "Weather temperature near the accident.",
            "values": "Degrees Fahrenheit.",
        },
        "Humidity(%)": {
            "type": "continuous",
            "description": "Weather humidity near the accident.",
            "values": "Percentage.",
        },
        "Pressure(in)": {
            "type": "continuous",
            "description": "Atmospheric pressure near the accident.",
            "values": "Inches.",
        },
        "Visibility(mi)": {
            "type": "continuous",
            "description": "Weather visibility near the accident.",
            "values": "Miles.",
        },
        "Wind_Speed(mph)": {
            "type": "continuous",
            "description": "Wind speed near the accident.",
            "values": "Miles per hour.",
        },
        "Precipitation(in)": {
            "type": "continuous",
            "description": "Precipitation near the accident.",
            "values": "Inches.",
        },
        "Amenity": {
            "type": "boolean",
            "description": "Whether an amenity point of interest is nearby.",
            "values": "0=false, 1=true.",
        },
        "Bump": {
            "type": "boolean",
            "description": "Whether a road bump is nearby.",
            "values": "0=false, 1=true.",
        },
        "Crossing": {
            "type": "boolean",
            "description": "Whether a crossing is nearby.",
            "values": "0=false, 1=true.",
        },
        "Give_Way": {
            "type": "boolean",
            "description": "Whether a give-way sign is nearby.",
            "values": "0=false, 1=true.",
        },
        "Junction": {
            "type": "boolean",
            "description": "Whether a road junction is nearby.",
            "values": "0=false, 1=true.",
        },
        "No_Exit": {
            "type": "boolean",
            "description": "Whether a no-exit road feature is nearby.",
            "values": "0=false, 1=true.",
        },
        "Railway": {
            "type": "boolean",
            "description": "Whether a railway feature is nearby.",
            "values": "0=false, 1=true.",
        },
        "Roundabout": {
            "type": "boolean",
            "description": "Whether a roundabout is nearby.",
            "values": "0=false, 1=true.",
        },
        "Station": {
            "type": "boolean",
            "description": "Whether a station point of interest is nearby.",
            "values": "0=false, 1=true.",
        },
        "Stop": {
            "type": "boolean",
            "description": "Whether a stop sign is nearby.",
            "values": "0=false, 1=true.",
        },
        "Traffic_Calming": {
            "type": "boolean",
            "description": "Whether a traffic-calming feature is nearby.",
            "values": "0=false, 1=true.",
        },
        "Traffic_Signal": {
            "type": "boolean",
            "description": "Whether a traffic signal is nearby.",
            "values": "0=false, 1=true.",
        },
        "Year": {
            "type": "ordinal_categorical",
            "description": "Accident year.",
        },
        "Month": {
            "type": "ordinal_categorical",
            "description": "Accident month.",
            "values": "1-12.",
        },
        "Weekday": {
            "type": "ordinal_categorical",
            "description": "Accident day of week.",
        },
        "Day": {
            "type": "ordinal_categorical",
            "description": "Accident day of month.",
            "values": "1-31.",
        },
        "Hour": {
            "type": "ordinal_categorical",
            "description": "Accident hour of day.",
            "values": "0-23.",
        },
        "Minute": {
            "type": "integer",
            "description": "Accident minute within the hour.",
            "values": "0-59.",
        },
        "Wind_Direction": {
            "type": "categorical",
            "description": "Wind direction category near the accident.",
            "values": "Grouped one-hot model columns: E, N, NE, NW, S, SE, SW, Variable, W.",
        },
        "Weather_Condition": {
            "type": "categorical",
            "description": "Weather condition category near the accident.",
            "values": "Grouped one-hot model columns such as Cloudy, Fog, Rain, Snow, Thunderstorm, Windy.",
        },
        "Civil_Twilight": {
            "type": "categorical",
            "description": "Whether the accident occurred during civil twilight day or night.",
            "values": "Grouped one-hot model columns; current model column indicates Night.",
        },
    },
}


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    return text[: MAX_TEXT_LENGTH - 3].rstrip() + "..."


def _kind_name(kind: Any) -> str:
    if kind is None:
        return "unknown"
    if isinstance(kind, str):
        return kind.lower()
    return getattr(kind, "__name__", str(kind)).lower()


def _feature_type(kind: Any, value_mapping: Any = None) -> str:
    name = _kind_name(kind)
    if "categorical" in name or value_mapping:
        if isinstance(value_mapping, Mapping) and set(value_mapping.values()) <= {"yes", "no", "Yes", "No"}:
            return "boolean_categorical"
        return "categorical"
    if name in {"bool", "boolean"}:
        return "boolean"
    if name in {"int", "int64", "integer"}:
        return "integer"
    if name in {"float", "float64", "double"}:
        return "continuous"
    return name


def _format_value_mapping(value_mapping: Any) -> str:
    if not isinstance(value_mapping, Mapping) or not value_mapping:
        return ""
    items = list(value_mapping.items())
    shown = items[:MAX_VALUE_MAPPING_ITEMS]
    text = "; ".join(f"{key}={_clean_text(value)}" for key, value in shown)
    remaining = len(items) - len(shown)
    if remaining > 0:
        text += f"; ... (+{remaining} more)"
    return text


def _tableshift_feature_lookup(dataset: str) -> Dict[str, Any]:
    try:
        from tableshift.core.tasks import _TASK_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "TableShift feature cards require the optional `tableshift` package."
        ) from exc
    feature_list = _TASK_REGISTRY[dataset].feature_list
    return {str(name): feature_list[name] for name in feature_list.predictors}


def _tableshift_card(dataset: str, feature_name: str) -> Dict[str, Any]:
    lookup = _tableshift_feature_lookup(dataset)
    semantic_feature_name = feature_name
    if feature_name not in lookup:
        semantic_feature_name = longest_prefix_feature(feature_name, tuple(lookup))
    if semantic_feature_name not in lookup:
        return {}
    feature = lookup[semantic_feature_name]
    value_mapping = getattr(feature, "value_mapping", None)
    card = {
        "type": _feature_type(getattr(feature, "kind", None), value_mapping),
        "description": _clean_text(
            getattr(feature, "name_extended", None)
            or getattr(feature, "description", None)
            or semantic_feature_name
        ),
    }
    if semantic_feature_name != feature_name:
        card["type"] = "encoded_model_feature"
        card["description"] = (
            f"Model feature `{feature_name}` derived from semantic feature "
            f"`{semantic_feature_name}`. Interpret this column as the encoded "
            f"model input for that derived category/value. {card['description']}"
        )
    values = _format_value_mapping(value_mapping)
    if values:
        card["values"] = values
    note = getattr(feature, "note", None)
    if note:
        card["note"] = _clean_text(note)
    return card


def _static_card(benchmark: str, dataset: str, feature_name: str) -> Dict[str, Any]:
    if benchmark == "acs":
        return dict(ACS_FEATURE_CARDS.get(feature_name, {}))
    if benchmark == "whyshift":
        cards = WHYSHIFT_FEATURE_CARDS.get(dataset, {})
        if feature_name in cards:
            return dict(cards[feature_name])
        semantic_feature_name = whyshift_semantic_name(dataset, feature_name)
        if semantic_feature_name in cards:
            card = dict(cards[semantic_feature_name])
            card["type"] = "encoded_model_feature"
            description = card.get("description", semantic_feature_name)
            card["description"] = (
                f"Model feature `{feature_name}` derived from semantic feature "
                f"`{semantic_feature_name}`. Interpret this column as the encoded "
                f"model input for that derived category/value. {description}"
            )
            return card
        return {}
    return {}


def feature_card_for_task(
    benchmark: str,
    dataset: str,
    feature_id: int,
    feature_name: str,
) -> Dict[str, Any]:
    if benchmark == "tableshift":
        metadata = _tableshift_card(dataset, feature_name)
    else:
        metadata = _static_card(benchmark, dataset, feature_name)

    card = {"feature_id": int(feature_id), "feature": feature_name}
    for key in ("type", "description", "values", "note"):
        value = metadata.get(key)
        if value:
            card[key] = _clean_text(value)
    if "description" not in card:
        card["description"] = feature_name
    return card


def feature_cards_for_task(
    benchmark: str,
    dataset: str,
    feature_index: Dict[int, str],
) -> List[Dict[str, Any]]:
    return [
        feature_card_for_task(benchmark, dataset, idx, feature_index[idx])
        for idx in sorted(feature_index)
    ]
