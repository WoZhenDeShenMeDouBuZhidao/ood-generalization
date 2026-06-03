import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from acs_tasks.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from acs_tasks.dataset import acs_task_load_data
from src.main import _data_cache_dir
from src.metrics import binary_accuracy_f1
from src.ranking import SCORE_WEIGHT_METHODS, llm_lasso_penalty_factors_from_ranking
from src.utils import ExperimentMetrics, metric_summary, print_results, set_seeds


def parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACS LLM-Lasso weighted L1 logistic baseline.")
    parser.add_argument("--dataset", choices=ACS_DATASET_ORDER, help="Run one ACS task. Default: run all tasks.")
    parser.add_argument(
        "--ranking-method",
        choices=SCORE_WEIGHT_METHODS,
        default=os.environ.get("RANKING_METHOD", "score_all"),
        help="LLM-Lasso requires score-based ranking artifacts.",
    )
    parser.add_argument("--eta-grid", type=parse_float_grid, default=parse_float_grid(os.environ.get("ETA_GRID", "0,1,2,3,4")))
    parser.add_argument("--c-grid", type=parse_float_grid, default=parse_float_grid(os.environ.get("C_GRID", "0.01,0.1,1,10,100")))
    parser.add_argument("--max-iter", type=int, default=int(os.environ.get("MAX_ITER", "5000")))
    parser.add_argument("--penalty-floor", type=float, default=float(os.environ.get("PENALTY_FLOOR", "0.1")))
    parser.add_argument("--class-weight", choices=("none", "balanced"), default=os.environ.get("CLASS_WEIGHT", "none"))
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


def _weighted_l1_features(X: np.ndarray, penalty_factors: np.ndarray) -> np.ndarray:
    return X / np.clip(penalty_factors, 1e-6, None)


def _evaluate(model: LogisticRegression, scaler: StandardScaler, penalty_factors: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X_scaled = _weighted_l1_features(scaler.transform(X), penalty_factors)
    preds = model.predict(X_scaled)
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
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    class_weight = None if args.class_weight == "none" else args.class_weight

    best_model = None
    best_penalty_factors = None
    best_eta = None
    best_c = None
    best_val_acc = -np.inf
    best_val_f1 = 0.0
    for eta in args.eta_grid:
        penalties_by_feature = llm_lasso_penalty_factors_from_ranking(
            args.dataset,
            feature_index,
            method=args.ranking_method,
            eta=eta,
            penalty_floor=args.penalty_floor,
        )
        penalty_factors = np.array([
            penalties_by_feature[feature_index[idx]]
            for idx in kept_feature_indices
        ], dtype=float)
        X_train_weighted = _weighted_l1_features(X_train_scaled, penalty_factors)

        for c_value in args.c_grid:
            model = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=args.max_iter,
                penalty="l1",
                random_state=67,
                solver="saga",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train_weighted, y_train)
            val_acc, val_f1 = _evaluate(model, scaler, penalty_factors, X_val, y_val)
            print(f"eta={eta:g}, C={c_value:g}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
            if val_acc > best_val_acc:
                best_model = model
                best_penalty_factors = penalty_factors
                best_eta = eta
                best_c = c_value
                best_val_acc = val_acc
                best_val_f1 = val_f1

    test_accs = []
    test_f1s = []
    for test in tests:
        X_test, y_test = _as_numpy(test)
        test_acc, test_f1 = _evaluate(best_model, scaler, best_penalty_factors, X_test, y_test)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    test_accs_np = np.array(test_accs)
    test_f1s_np = np.array(test_f1s)
    print({
        "best_eta": best_eta,
        "best_C": best_c,
        "ranking_method": args.ranking_method,
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
