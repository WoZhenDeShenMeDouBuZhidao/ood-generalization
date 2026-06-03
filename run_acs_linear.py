import argparse
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.acs.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from data.acs.dataset import acs_task_load_data
from src.data_cache import load_or_build_dataset_cache
from src.metrics import binary_accuracy_f1
from src.paths import result_output_path
from src.utils import (
    ExperimentMetrics,
    experiment_metrics_to_dict,
    metric_summary,
    save_json_result,
    set_seeds,
)

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

    def _build_dataset():
        set_seeds(67)
        return acs_task_load_data(
            dataset,
            removed_feature_indices,
            task_config.train_val_state,
            list(test_states_for(dataset)),
            0.2,
            dataset_config,
        )

    return load_or_build_dataset_cache(
        "acs",
        dataset,
        removed_feature_indices,
        dataset_config,
        _build_dataset,
    )


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
        metrics = tuple((0.0, 0.0) for _ in range(8))
        save_json_result(result_output_path("acs", "linear", args.dataset), {
            "status": "all_features_removed",
            "metadata": {"method": "linear"},
            "dataset": args.dataset,
            "removed_feature_indices": sorted(removed_feature_indices),
            "metrics": experiment_metrics_to_dict(metrics),
            "message": "No input features remain; recording zero ACC/F1 metrics.",
        })
        return metrics

    train, val, tests, data_dir, cache_hit = _load_acs_data(args.dataset)
    X_train, y_train = _as_numpy(train)
    X_val, y_val = _as_numpy(val)
    class_weight = "balanced" if REWEIGHTING_BY_DATASET[args.dataset] else None

    best_model = None
    best_c = None
    best_val_acc = -np.inf
    best_val_f1 = 0.0
    search_trace = []
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
        search_trace.append({
            "C": c_value,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        })
        if val_acc > best_val_acc:
            best_model = model
            best_c = c_value
            best_val_acc = val_acc
            best_val_f1 = val_f1

    test_accs = []
    test_f1s = []
    test_state_metrics = []
    for state, test in zip(test_states_for(args.dataset), tests):
        X_test, y_test = _as_numpy(test)
        test_acc, test_f1 = _evaluate(best_model, X_test, y_test)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_state_metrics.append({
            "state": state,
            "accuracy": test_acc,
            "f1": test_f1,
        })

    test_accs_np = np.array(test_accs)
    test_f1s_np = np.array(test_f1s)
    metrics = (
        metric_summary(best_val_acc),
        metric_summary(float(test_accs_np.mean())),
        metric_summary(float(test_accs_np.min())),
        metric_summary(float(test_accs_np.std())),
        metric_summary(best_val_f1),
        metric_summary(float(test_f1s_np.mean())),
        metric_summary(float(test_f1s_np.min())),
        metric_summary(float(test_f1s_np.std())),
    )
    save_json_result(result_output_path("acs", "linear", args.dataset), {
        "status": "ok",
        "metadata": {"method": "linear"},
        "dataset": args.dataset,
        "train_val_group": ACS_TASK_CONFIGS[args.dataset].train_val_state,
        "test_groups": list(test_states_for(args.dataset)),
        "feature_index": feature_index,
        "removed_feature_indices": sorted(removed_feature_indices),
        "data_cache_dir": data_dir,
        "data_cache_hit": cache_hit,
        "model": {
            "name": "logistic_regression",
            "c_grid": args.c_grid,
            "max_iter": args.max_iter,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": class_weight,
            "standard_scaler": "fit_on_train_only",
        },
        "best_hyperparameters": {
            "C": best_c,
        },
        "search_trace": search_trace,
        "test_states": test_state_metrics,
        "metrics": experiment_metrics_to_dict(metrics),
    })
    return metrics


def main_cli() -> None:
    args = build_parser().parse_args()
    datasets = [args.dataset] if args.dataset else ACS_DATASET_ORDER
    for dataset in datasets:
        task_args = argparse.Namespace(**vars(args))
        task_args.dataset = dataset
        run_dataset(task_args)
        print(f"saved {result_output_path('acs', 'linear', dataset)}")


if __name__ == "__main__":
    main_cli()
