from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def benchmark_data_dir(benchmark: str) -> Path:
    return DATA_ROOT / benchmark


def dataset_data_dir(benchmark: str, dataset: str) -> Path:
    return benchmark_data_dir(benchmark) / dataset


def dataset_cache_dir(benchmark: str, dataset: str) -> Path:
    return dataset_data_dir(benchmark, dataset) / "cache"


def ranking_artifact_dir(benchmark: str, dataset: str) -> Path:
    return dataset_data_dir(benchmark, dataset) / "rankings"


def dataset_output_dir(benchmark: str, dataset: str) -> Path:
    return OUTPUT_ROOT / benchmark / dataset


def dataset_plot_dir(benchmark: str, dataset: str) -> Path:
    return dataset_output_dir(benchmark, dataset) / "plots"

def result_output_path(benchmark: str, method: str, dataset: str, filename: str = "result.json") -> Path:
    return OUTPUT_ROOT / benchmark / method / dataset / filename


def acs_raw_data_root() -> Path:
    return benchmark_data_dir("acs") / "raw"
