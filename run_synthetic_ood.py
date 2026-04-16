import torch

from synthetic_ood.dataset import FEATURE_INDEX, SyntheticConfig
from src.main import main


if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    DATASET = "synthetic_ood"
    TRAIN_VAL_GROUP = "train_env"
    TEST_GROUPS = ["ood_env"]
    MODEL_NAME = "mlp"
    LOSS_NAME = "feature_importance_target_ce"
    FEATURE_INDEX = {
        0: 'causal_main',
        1: 'causal_aux',
        2: 'spurious_main',
        3: 'noise_1',
        4: 'noise_2',
        5: 'noise_3',
    }
    REMOVED_FEATURE_INDICES = []

    TRAIN_BATCH = 256
    EVAL_BATCH = 2048
    LR = 1e-4
    PATIENCE = 40
    REPEAT = 5
    MAX_EPOCHS = 400

    DATASET_CONFIG = SyntheticConfig(
        train_size=6000,
        val_size=2000,
        test_size=4000,
        dataset_seed=20260401,
        causal_main_strength=1.25,
        causal_aux_strength=0.75,
        spurious_strength=2.25,
        feature_noise=1.0,
        train_spurious_corr=0.95,
        ood_spurious_corr=-0.95,
    )

    FEATURE_LOSS_WEIGHTS = {
        "causal_main": 3.0,
        "causal_aux": 2.0,
        "spurious_main": 0.0,
        "noise_1": 1.0,
        "noise_2": 1.0,
        "noise_3": 1.0,
    }
    REG_SCALE = 0.0
    LOSS_KWARGS = {
        "grad_scale":2.0,
        "weight_scale": 3.0,
        "suppress_scale": 6.0,
        "target_power": 1.0,
    }

    ID_result, OOD_MEAN_result, OOD_WORST_result = main(
        DATASET, TRAIN_VAL_GROUP, TEST_GROUPS,
        FEATURE_INDEX, REMOVED_FEATURE_INDICES, FEATURE_LOSS_WEIGHTS,
        TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
        PATIENCE=PATIENCE, REPEAT=REPEAT, MAX_EPOCHS=MAX_EPOCHS,
        DATASET_CONFIG=DATASET_CONFIG, PLOT_TEST_SHAP=False,
        MODEL_NAME=MODEL_NAME, LOSS_NAME=LOSS_NAME, LOSS_KWARGS=LOSS_KWARGS,
        device=device, MODEL_SEEDS=[9803, 38224, 8113, 4854, 98825]
    )
    print(f"### Results:\n- ID: {ID_result:.4f}\n- OOD MEAN: {OOD_MEAN_result:.4f}\n- OOD WORST: {OOD_WORST_result:.4f}")
