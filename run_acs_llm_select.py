import argparse
import os

import torch

from data.acs.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from src.main import main
from src.paths import result_output_path
from src.ranking import (
    RANKING_METHODS,
    SCORE_WEIGHT_METHODS,
    SELECTION_MODES,
    removed_feature_indices_from_selected_names,
    selected_feature_names_from_ranking,
)
from src.utils import ExperimentMetrics, experiment_metrics_to_dict, save_json_result

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

MODEL_SEEDS = [9803, 38224, 8113, 4854, 98825]


def parse_float_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACS LLM-Select MLP baseline.")
    parser.add_argument("--dataset", choices=ACS_DATASET_ORDER, help="Run one ACS task. Default: run all tasks.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=os.environ.get("PATIENCE", "500"))
    parser.add_argument("--repeat", type=int, default=os.environ.get("REPEAT", "5"))
    parser.add_argument("--max-epochs", type=int, default=os.environ.get("MAX_EPOCHS", "5000"))
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default=os.environ.get("RANKING_METHOD", "score_all"),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=os.environ.get("SELECTION_MODE", "top_p"),
        help="Use top_p for all ranking methods; score_threshold only supports score/score_all.",
    )
    parser.add_argument("--top-p-grid", type=parse_float_grid, default=parse_float_grid(os.environ.get("TOP_P_GRID", "1.0,0.75,0.5,0.25")))
    parser.add_argument("--score-threshold-grid", type=parse_float_grid, default=parse_float_grid(os.environ.get("SCORE_THRESHOLD_GRID", "0.0,0.25,0.5,0.75")))
    return parser


def _candidate_values(args: argparse.Namespace) -> tuple[float, ...]:
    if args.selection_mode == "score_threshold":
        if args.ranking_method not in SCORE_WEIGHT_METHODS:
            valid = ", ".join(SCORE_WEIGHT_METHODS)
            raise ValueError(f"score_threshold only supports ranking methods: {valid}")
        return args.score_threshold_grid
    return args.top_p_grid


def _value_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _candidate_result_path(args: argparse.Namespace, value: float):
    filename = f"candidate_{args.selection_mode}_{_value_token(value)}.json"
    return result_output_path("acs", "llm_select", args.dataset, filename=filename)


def _run_candidate(args: argparse.Namespace, value: float) -> ExperimentMetrics:
    task_config = ACS_TASK_CONFIGS[args.dataset]
    feature_index = feature_index_for(args.dataset)
    if args.selection_mode == "score_threshold":
        selected_features = selected_feature_names_from_ranking(
            args.dataset,
            feature_index,
            method=args.ranking_method,
            selection_mode=args.selection_mode,
            score_threshold=value,
        )
    else:
        selected_features = selected_feature_names_from_ranking(
            args.dataset,
            feature_index,
            method=args.ranking_method,
            selection_mode=args.selection_mode,
            top_p=value,
        )
    removed_feature_indices = removed_feature_indices_from_selected_names(
        feature_index,
        selected_features,
        task_config.removed_feature_indices,
    )
    dataset_config = {
        "resampling": False,
        "standardize": False,
    }
    loss_kwargs = {
        "reweighting": REWEIGHTING_BY_DATASET[args.dataset],
    }
    return main(
        args.dataset,
        task_config.train_val_state,
        list(test_states_for(args.dataset)),
        feature_index,
        removed_feature_indices,
        {},
        TRAIN_BATCH=args.train_batch,
        EVAL_BATCH=args.eval_batch,
        LR=args.lr,
        REG_SCALE=0.0,
        PATIENCE=args.patience,
        REPEAT=args.repeat,
        MAX_EPOCHS=args.max_epochs,
        DATASET_CONFIG=dataset_config,
        PLOT_CURVE=False,
        PLOT_SHAP=False,
        PLOT_TEST_SHAP=False,
        MODEL_NAME="mlp",
        LOSS_NAME="cross_entropy",
        LOSS_KWARGS=loss_kwargs,
        device=args.device,
        MODEL_SEEDS=MODEL_SEEDS,
        RESULT_PATH=_candidate_result_path(args, value),
        RESULT_METADATA={
            "method": "llm_select",
            "ranking_method": args.ranking_method,
            "selection_mode": args.selection_mode,
            "selection_value": value,
            "selected_features": selected_features,
            "removed_feature_indices": removed_feature_indices,
        },
    )


def run_dataset(args: argparse.Namespace) -> ExperimentMetrics:
    best_value = None
    best_metrics = None
    best_val_acc = float("-inf")
    candidates = []
    for value in _candidate_values(args):
        metrics = _run_candidate(args, value)
        val_acc = metrics[0][0]
        candidates.append({
            "selection_mode": args.selection_mode,
            "selection_value": value,
            "result_path": _candidate_result_path(args, value),
            "metrics": experiment_metrics_to_dict(metrics),
        })
        if val_acc > best_val_acc:
            best_value = value
            best_metrics = metrics
            best_val_acc = val_acc
    save_json_result(result_output_path("acs", "llm_select", args.dataset), {
        "status": "ok",
        "metadata": {
            "method": "llm_select",
            "ranking_method": args.ranking_method,
            "selection_mode": args.selection_mode,
            "top_p_grid": args.top_p_grid,
            "score_threshold_grid": args.score_threshold_grid,
        },
        "dataset": args.dataset,
        "best_selection_mode": args.selection_mode,
        "best_selection_value": best_value,
        "metrics": experiment_metrics_to_dict(best_metrics),
        "candidates": candidates,
    })
    return best_metrics


def main_cli() -> None:
    args = build_parser().parse_args()
    datasets = [args.dataset] if args.dataset else ACS_DATASET_ORDER
    for dataset in datasets:
        task_args = argparse.Namespace(**vars(args))
        task_args.dataset = dataset
        run_dataset(task_args)
        print(f"saved {result_output_path('acs', 'llm_select', dataset)}")


if __name__ == "__main__":
    main_cli()
