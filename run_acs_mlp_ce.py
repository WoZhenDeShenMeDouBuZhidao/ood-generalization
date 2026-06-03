import argparse
import os

import torch

from acs_tasks.config import ACS_DATASET_ORDER, ACS_TASK_CONFIGS, feature_index_for, test_states_for
from src.main import main
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
    parser = argparse.ArgumentParser(description="Run ACS MLP CrossEntropyLoss baseline.")
    parser.add_argument("--dataset", choices=ACS_DATASET_ORDER, help="Run one ACS task. Default: run all tasks.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=os.environ.get("PATIENCE", "500"))
    parser.add_argument("--repeat", type=int, default=os.environ.get("REPEAT", "5"))
    parser.add_argument("--max-epochs", type=int, default=os.environ.get("MAX_EPOCHS", "5000"))
    return parser


def run_dataset(args: argparse.Namespace) -> ExperimentMetrics:
    task_config = ACS_TASK_CONFIGS[args.dataset]
    feature_index = feature_index_for(args.dataset)
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
        list(task_config.removed_feature_indices),
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
