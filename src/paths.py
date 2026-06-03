from pathlib import Path


ACS_DATASETS = {
    "acsincome",
    "acsincome_acc_upperbound_test",
    "acsemployment",
    "acsemploymentfiltered",
    "acshealthinsurance",
    "acsincomepovertyratio",
    "acsmobility",
    "acspubliccoverage",
    "acstraveltime",
}


def dataset_artifact_dir(dataset: str) -> Path:
    if dataset in ACS_DATASETS:
        return Path("acs_tasks") / dataset
    return Path(dataset)
