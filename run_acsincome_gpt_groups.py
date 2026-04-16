import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

import src.main as main_module
import src.trainer as trainer_module


FEATURE_INDEX = {
    0: "AGEP",
    1: "COW",
    2: "SCHL",
    3: "MAR",
    4: "OCCP",
    5: "POBP",
    6: "RELP",
    7: "WKHP",
    8: "SEX",
    9: "RAC1P",
}

TRAIN_VAL_STATE = "PR"
TEST_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IA", "KS", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
]
MODEL_SEEDS = [9803, 38224, 8113, 2011, 74927]


class _DummyShapOutput:
    def __init__(self) -> None:
        self.values = None


class _DummyExplainer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs) -> _DummyShapOutput:
        return _DummyShapOutput()


def _empty_weights() -> Dict[str, float]:
    return {feat_name: 0.0 for feat_name in FEATURE_INDEX.values()}


def _load_gpt_ranking(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _weights_from_ranking(ranking: List[str], rank_to_weight: Dict[int, float]) -> Dict[str, float]:
    weights = _empty_weights()
    for rank, feat_name in enumerate(ranking, start=1):
        weights[feat_name] = float(rank_to_weight.get(rank, 0.0))
    return weights


def _weights_from_groups(group_definition: Dict[str, List[str]], group_to_weight: Dict[str, float]) -> Dict[str, float]:
    weights = _empty_weights()
    for group_name, feat_names in group_definition.items():
        for feat_name in feat_names:
            weights[feat_name] = float(group_to_weight.get(group_name, 0.0))
    return weights


def build_named_configs(gpt_spec: Dict) -> Dict[str, Dict[str, float]]:
    ranking = gpt_spec["ranking"]
    group_definition = gpt_spec["group_definition"]
    spurious_start_rank = int(gpt_spec["spurious_start_rank"])

    two_group_weights = {
        rank: 1.0 if rank < spurious_start_rank else 0.0
        for rank in range(1, len(ranking) + 1)
    }
    four_group_4321 = {
        1: 4.0, 2: 4.0,
        3: 3.0, 4: 3.0,
        5: 2.0, 6: 2.0, 7: 2.0,
        8: 1.0, 9: 1.0, 10: 1.0,
    }
    four_group_4320 = {
        1: 4.0, 2: 4.0,
        3: 3.0, 4: 3.0,
        5: 2.0, 6: 2.0, 7: 2.0,
        8: 0.0, 9: 0.0, 10: 0.0,
    }

    return {
        "gpt_3group_321": _weights_from_groups(
            group_definition,
            {"group_1": 3.0, "group_2": 2.0, "group_3": 1.0},
        ),
        "gpt_3group_320": _weights_from_groups(
            group_definition,
            {"group_1": 3.0, "group_2": 2.0, "group_3": 0.0},
        ),
        "gpt_3group_310": _weights_from_groups(
            group_definition,
            {"group_1": 3.0, "group_2": 1.0, "group_3": 0.0},
        ),
        "gpt_2group_110": _weights_from_ranking(ranking, two_group_weights),
        "gpt_4group_4321": _weights_from_ranking(ranking, four_group_4321),
        "gpt_4group_4320": _weights_from_ranking(ranking, four_group_4320),
    }


def run_config(
    name: str,
    feature_loss_weights: Dict[str, float],
    repeat: int,
    device: str,
) -> Dict[str, float]:
    print(f"=== Running {name} on {device}")
    print(json.dumps(feature_loss_weights, ensure_ascii=False, sort_keys=True))

    id_result, ood_mean_result, ood_worst_result = main_module.main(
        dataset="acsincome",
        TRAIN_VAL_GROUP=TRAIN_VAL_STATE,
        TEST_GROUPS=TEST_STATES,
        FEATURE_INDEX=FEATURE_INDEX,
        REMOVED_FEATURE_INDICES=[],
        FEATURE_LOSS_WEIGHTS=feature_loss_weights,
        TRAIN_BATCH=256,
        EVAL_BATCH=2048,
        LR=1e-4,
        REG_SCALE=0.5,
        PATIENCE=500,
        REPEAT=repeat,
        MAX_EPOCHS=5000,
        DATASET_CONFIG=None,
        PLOT_TEST_SHAP=False,
        MODEL_NAME="mlp",
        LOSS_NAME="feature_importance_target_ce",
        LOSS_KWARGS={
            "grad_scale": 2.0,
            "weight_scale": 3.0,
            "suppress_scale": 6.0,
            "target_power": 1.0,
        },
        MODEL_SEEDS=MODEL_SEEDS[:repeat],
        device=device,
    )

    result = {
        "ID": float(id_result),
        "OOD_MEAN": float(ood_mean_result),
        "OOD_WORST": float(ood_worst_result),
    }
    print(f"RESULT {name} {result}")
    return result


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-json", default="gpt_acsincome.json")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["gpt_3group_321", "gpt_3group_320", "gpt_3group_310", "gpt_2group_110"],
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-path", default="acsincome/results/gpt_group_runs.jsonl")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    if args.skip_analysis:
        main_module.plot_training_curves = lambda *args, **kwargs: None
        main_module.plot_shap_values = lambda *args, **kwargs: None
        trainer_module.shap.Explainer = _DummyExplainer

    gpt_spec = _load_gpt_ranking(Path(args.gpt_json))
    configs = build_named_configs(gpt_spec)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for config_name in args.configs:
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}")
        result = run_config(config_name, configs[config_name], repeat=args.repeat, device=args.device)
        record = {
            "config": config_name,
            "repeat": args.repeat,
            "device": args.device,
            "feature_loss_weights": configs[config_name],
            **result,
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main_cli()
