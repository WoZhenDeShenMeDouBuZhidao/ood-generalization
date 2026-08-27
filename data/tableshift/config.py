from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from src.paths import dataset_data_dir


TABLESHIFT_DATASET_ORDER = (
    "college_scorecard",
    "diabetes_readmission",
    "nhanes_lead",
    "nhanes_cholesterol",
    "anes",
    "acsfoodstamps",
    "brfss_diabetes",
    "brfss_blood_pressure",
)

TABLESHIFT_TEST_GROUPS = ("ood_test",)
TABLESHIFT_REPORT_ID_GROUPS = ()
TABLESHIFT_REPORT_OOD_GROUPS = ("ood_test",)


@dataclass(frozen=True)
class TableShiftTaskConfig:
    dataset: str
    removed_feature_indices: Tuple[int, ...] = ()


TABLESHIFT_TASK_CONFIGS = {
    dataset: TableShiftTaskConfig(dataset)
    for dataset in TABLESHIFT_DATASET_ORDER
}


def raw_dir_for(dataset: str) -> Path:
    return dataset_data_dir("tableshift", dataset) / "raw"
