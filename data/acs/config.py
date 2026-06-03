from dataclasses import dataclass
from typing import Dict, Tuple

from data.acs.dataset import ACS_TASKS


ACS_DATASET_ORDER = (
    "acsincome",
    "acsemployment",
    "acsemploymentfiltered",
    "acshealthinsurance",
    "acsincomepovertyratio",
    "acsmobility",
    "acspubliccoverage",
    "acstraveltime",
)

ACS_BASE_TEST_STATES = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IA", "KS", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
)


@dataclass(frozen=True)
class ACSTaskConfig:
    dataset: str
    train_val_state: str
    removed_feature_indices: Tuple[int, ...] = ()


ACS_TASK_CONFIGS = {
    "acsincome": ACSTaskConfig("acsincome", "PR"),
    "acsemployment": ACSTaskConfig("acsemployment", "SD"),
    "acsemploymentfiltered": ACSTaskConfig("acsemploymentfiltered", "SD"),
    "acshealthinsurance": ACSTaskConfig("acshealthinsurance", "MN", (23,)),
    "acsincomepovertyratio": ACSTaskConfig("acsincomepovertyratio", "HI"),
    "acsmobility": ACSTaskConfig("acsmobility", "AK"),
    "acspubliccoverage": ACSTaskConfig("acspubliccoverage", "CA", (16,)),
    "acstraveltime": ACSTaskConfig("acstraveltime", "AZ", (9, 10, 14)),
}

# FEATURE_INDEX reference.
# Runtime FEATURE_INDEX values are loaded from ACS_TASKS by feature_index_for().
#
# acsincome:
#   0: "AGEP"
#   1: "COW"
#   2: "SCHL"
#   3: "MAR"
#   4: "OCCP"
#   5: "POBP"
#   6: "RELP"
#   7: "WKHP"
#   8: "SEX"
#   9: "RAC1P"
#
# acsemployment:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "RELP"
#   4: "DIS"
#   5: "ESP"
#   6: "CIT"
#   7: "MIG"
#   8: "MIL"
#   9: "ANC"
#   10: "NATIVITY"
#   11: "DEAR"
#   12: "DEYE"
#   13: "DREM"
#   14: "SEX"
#   15: "RAC1P"
#
# acsemploymentfiltered:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "MIG"
#   7: "CIT"
#   8: "MIL"
#   9: "ANC"
#   10: "NATIVITY"
#   11: "RELP"
#   12: "DEAR"
#   13: "DEYE"
#   14: "DREM"
#   15: "RAC1P"
#   16: "GCL"
#
# acshealthinsurance:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "CIT"
#   7: "MIG"
#   8: "MIL"
#   9: "ANC"
#   10: "NATIVITY"
#   11: "DEAR"
#   12: "DEYE"
#   13: "DREM"
#   14: "RACAIAN"
#   15: "RACASN"
#   16: "RACBLK"
#   17: "RACNH"
#   18: "RACPI"
#   19: "RACSOR"
#   20: "RACWHT"
#   21: "PINCP"
#   22: "ESR"
#   23: "ST"
#   24: "FER"
#
# acsincomepovertyratio:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "MIG"
#   7: "CIT"
#   8: "MIL"
#   9: "ANC"
#   10: "NATIVITY"
#   11: "RELP"
#   12: "DEAR"
#   13: "DEYE"
#   14: "DREM"
#   15: "RAC1P"
#   16: "GCL"
#   17: "ESR"
#   18: "OCCP"
#   19: "WKHP"
#
# acsmobility:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "CIT"
#   7: "MIL"
#   8: "ANC"
#   9: "NATIVITY"
#   10: "RELP"
#   11: "DEAR"
#   12: "DEYE"
#   13: "DREM"
#   14: "RAC1P"
#   15: "GCL"
#   16: "COW"
#   17: "ESR"
#   18: "WKHP"
#   19: "JWMNP"
#   20: "PINCP"
#
# acspubliccoverage:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "CIT"
#   7: "MIG"
#   8: "MIL"
#   9: "ANC"
#   10: "NATIVITY"
#   11: "DEAR"
#   12: "DEYE"
#   13: "DREM"
#   14: "PINCP"
#   15: "ESR"
#   16: "ST"
#   17: "FER"
#   18: "RAC1P"
#
# acstraveltime:
#   0: "AGEP"
#   1: "SCHL"
#   2: "MAR"
#   3: "SEX"
#   4: "DIS"
#   5: "ESP"
#   6: "MIG"
#   7: "RELP"
#   8: "RAC1P"
#   9: "PUMA"
#   10: "ST"
#   11: "CIT"
#   12: "OCCP"
#   13: "JWTR"
#   14: "POWPUMA"
#   15: "POVPIP"


def feature_index_for(dataset: str) -> Dict[int, str]:
    if dataset not in ACS_TASKS:
        raise ValueError(f"Unknown ACS dataset: {dataset}")
    return {idx: feature for idx, feature in enumerate(ACS_TASKS[dataset].features)}


def test_states_for(dataset: str) -> Tuple[str, ...]:
    config = ACS_TASK_CONFIGS[dataset]
    return tuple(state for state in ACS_BASE_TEST_STATES if state != config.train_val_state)
