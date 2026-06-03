import argparse
import os

import torch

from acs_tasks.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from src.main import main
from src.ranking import FEATURE_WEIGHT_MODES, RANKING_METHODS, feature_loss_weights_from_ranking
from src.utils import ExperimentMetrics, print_results

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACS FeatureImportanceTargetCELoss baseline.")
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
        help=f"Ranking artifact method for RANKING_METHOD. Options: {', '.join(RANKING_METHODS)}.",
    )
    parser.add_argument(
        "--feature-weight-mode",
        choices=FEATURE_WEIGHT_MODES,
        default=os.environ.get("FEATURE_WEIGHT_MODE", "score"),
        help=f"Feature weight conversion mode for FEATURE_WEIGHT_MODE. Options: {', '.join(FEATURE_WEIGHT_MODES)}.",
    )
    parser.add_argument("--suppress-bound", type=float, default=os.environ.get("SUPPRESS_BOUND", "0.0"))
    parser.add_argument("--reg-scale", type=float, default=os.environ.get("REG_SCALE", "2.0"))
    parser.add_argument("--loss-lambda", type=float, default=os.environ.get("LOSS_LAMBDA", "16.0"))
    parser.add_argument("--loss-alpha", type=float, default=os.environ.get("LOSS_ALPHA", "0.75"))
    parser.add_argument("--target-power", type=float, default=os.environ.get("TARGET_POWER", "1.0"))
    return parser


def run_dataset(args: argparse.Namespace) -> ExperimentMetrics:
    task_config = ACS_TASK_CONFIGS[args.dataset]
    feature_index = feature_index_for(args.dataset)
    feature_loss_weights = feature_loss_weights_from_ranking(
        args.dataset,
        feature_index,
        method=args.ranking_method,
        weight_mode=args.feature_weight_mode,
        suppress_bound=args.suppress_bound,
    )
    loss_kwargs = {
        "reweighting": REWEIGHTING_BY_DATASET[args.dataset],
        "grad_scale": args.loss_lambda * args.loss_alpha,
        "weight_scale": args.loss_lambda * (1.0 - args.loss_alpha),
        "suppress_scale": 1.0,
        "target_power": args.target_power,
        "importance_scale": "train_std",
    }
    dataset_config = {
        "resampling": False,
        "standardize": False,
    }
    return main(
        args.dataset,
        task_config.train_val_state,
        list(test_states_for(args.dataset)),
        feature_index,
        list(task_config.removed_feature_indices),
        feature_loss_weights,
        TRAIN_BATCH=args.train_batch,
        EVAL_BATCH=args.eval_batch,
        LR=args.lr,
        REG_SCALE=args.reg_scale,
        PATIENCE=args.patience,
        REPEAT=args.repeat,
        MAX_EPOCHS=args.max_epochs,
        DATASET_CONFIG=dataset_config,
        PLOT_CURVE=False,
        PLOT_SHAP=False,
        PLOT_TEST_SHAP=False,
        MODEL_NAME="mlp",
        LOSS_NAME="feature_importance_target_ce",
        LOSS_KWARGS=loss_kwargs,
        device=args.device,
        MODEL_SEEDS=[9803, 38224, 8113, 4854, 98825],
    )


def main_cli() -> None:
    args = build_parser().parse_args()
    print(f"Device: {args.device}")
    datasets = [args.dataset] if args.dataset else ACS_DATASET_ORDER
    for dataset in datasets:
        task_args = argparse.Namespace(**vars(args))
        task_args.dataset = dataset
        print(f"Dataset: {dataset}")
        print_results(run_dataset(task_args))


if __name__ == "__main__":
    main_cli()
