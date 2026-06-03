import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acs_tasks.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from acs_tasks.dataset import acs_task_load_data
from src.main import _data_cache_dir
from src.metrics import binary_accuracy_f1
from src.utils import ExperimentMetrics, metric_summary, print_results, set_seeds

REWEIGHTING_BY_DATASET = {
    "acsincome": True,
    "acsemployment": False,
    "acsemploymentfiltered": False,
    "acshealthinsurance": False,
    "acsincomepovertyratio": True,
    "acsmobility": False,
    "acspubliccoverage": False,
    "acstraveltime": False,
}


def parse_c_grid(value: str) -> tuple[float, ...]:
    c_values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not c_values:
        raise argparse.ArgumentTypeError("C grid must contain at least one value.")
    if any(c <= 0 for c in c_values):
        raise argparse.ArgumentTypeError("All C values must be positive.")
    return c_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACS sklearn LogisticRegression baseline.")
    parser.add_argument("--dataset", choices=ACS_DATASET_ORDER, help="Run one ACS task. Default: run all tasks.")
    parser.add_argument("--c-grid", type=parse_c_grid, default=parse_c_grid(os.environ.get("C_GRID", "0.01,0.1,1,10,100")))
    parser.add_argument("--max-iter", type=int, default=int(os.environ.get("MAX_ITER", "1000")))
    return parser


def _load_acs_data(dataset: str):
    task_config = ACS_TASK_CONFIGS[dataset]
    dataset_config = {
        "resampling": False,
        "standardize": False,
    }
    removed_feature_indices = list(task_config.removed_feature_indices)
    data_dir = _data_cache_dir(dataset, removed_feature_indices, dataset_config)
    print({"data_cache_dir": str(data_dir)})

    if Path(data_dir / "train.pkl").is_file():
        with open(data_dir / "train.pkl", "rb") as fp:
            train = pickle.load(fp)
        with open(data_dir / "val.pkl", "rb") as fp:
            val = pickle.load(fp)
        with open(data_dir / "tests.pkl", "rb") as fp:
            tests = pickle.load(fp)
        return train, val, tests

    set_seeds(67)
    train, val, tests = acs_task_load_data(
        dataset,
        removed_feature_indices,
        task_config.train_val_state,
        list(test_states_for(dataset)),
        0.2,
        dataset_config,
    )
    os.makedirs(data_dir, exist_ok=True)
    with open(data_dir / "train.pkl", "wb") as fp:
        pickle.dump(train, fp)
    with open(data_dir / "val.pkl", "wb") as fp:
        pickle.dump(val, fp)
    with open(data_dir / "tests.pkl", "wb") as fp:
        pickle.dump(tests, fp)
    return train, val, tests


def _as_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    return dataset.X.detach().cpu().numpy(), dataset.Y.detach().cpu().numpy()


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    preds = model.predict(X)
    return binary_accuracy_f1(y, preds)


def run_dataset(args: argparse.Namespace) -> ExperimentMetrics:
    feature_index = feature_index_for(args.dataset)
    removed_feature_indices = set(ACS_TASK_CONFIGS[args.dataset].removed_feature_indices)
    kept_feature_indices = [
        idx for idx in sorted(feature_index)
        if idx not in removed_feature_indices
    ]
    if not kept_feature_indices:
        print({
            "all_features_removed": True,
            "removed_feature_indices": sorted(removed_feature_indices),
            "message": "No input features remain; recording zero ACC/F1 metrics.",
        })
        return tuple((0.0, 0.0) for _ in range(8))

    train, val, tests = _load_acs_data(args.dataset)
    X_train, y_train = _as_numpy(train)
    X_val, y_val = _as_numpy(val)
    class_weight = "balanced" if REWEIGHTING_BY_DATASET[args.dataset] else None

    best_model = None
    best_c = None
    best_val_acc = -np.inf
    best_val_f1 = 0.0
    for c_value in args.c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=args.max_iter,
                penalty="l2",
                solver="lbfgs",
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        val_acc, val_f1 = _evaluate(model, X_val, y_val)
        print(f"C={c_value:g}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        if val_acc > best_val_acc:
            best_model = model
            best_c = c_value
            best_val_acc = val_acc
            best_val_f1 = val_f1

    test_accs = []
    test_f1s = []
    for test in tests:
        X_test, y_test = _as_numpy(test)
        test_acc, test_f1 = _evaluate(best_model, X_test, y_test)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    test_accs_np = np.array(test_accs)
    test_f1s_np = np.array(test_f1s)
    print({
        "best_C": best_c,
        "class_weight": class_weight,
        "standard_scaler": "fit_on_train_only",
    })
    return (
        metric_summary(best_val_acc),
        metric_summary(float(test_accs_np.mean())),
        metric_summary(float(test_accs_np.min())),
        metric_summary(float(test_accs_np.std())),
        metric_summary(best_val_f1),
        metric_summary(float(test_f1s_np.mean())),
        metric_summary(float(test_f1s_np.min())),
        metric_summary(float(test_f1s_np.std())),
    )


def main_cli() -> None:
    args = build_parser().parse_args()
    datasets = [args.dataset] if args.dataset else ACS_DATASET_ORDER
    for dataset in datasets:
        task_args = argparse.Namespace(**vars(args))
        task_args.dataset = dataset
        print(f"Dataset: {dataset}")
        print_results(run_dataset(task_args))


if __name__ == "__main__":
    main_cli()
