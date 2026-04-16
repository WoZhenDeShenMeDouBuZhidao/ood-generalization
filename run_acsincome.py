import torch
from src.main import main
from src.utils import plot_accdelta_bars

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    DATASET = "acsincome"
    TRAIN_VAL_STATE = "PR"
    TEST_STATES = [ 
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IA", "KS", "ME", "MD", "MA", "MI", "MN",
        "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
        "ND", "OH", "OK", "OR", "PA", "RI", "SD", "TN", "TX", "UT",
        "VT", "VA", "WA", "WV", "WI", "WY",  # States list (known corrupted states excluded)
    ]
    MODEL_NAME = "mlp"
    LOSS_NAME = "feature_importance_target_ce"
    FEATURE_INDEX = {
        0: 'AGEP',
        1: 'COW',
        2: 'SCHL',
        3: 'MAR',
        4: 'OCCP',
        5: 'POBP',
        6: 'RELP',
        7: 'WKHP',
        8: 'SEX',
        9: 'RAC1P'
    }
    REMOVED_FEATURE_INDICES = []

    TRAIN_BATCH = 256
    EVAL_BATCH = 2048
    LR = 1e-4
    PATIENCE = 500
    REPEAT = 5
    MAX_EPOCHS = 5000

    FEATURE_LOSS_WEIGHTS = {
        'AGEP': 7.0,
        'COW': 2.0,
        'SCHL': 10.0,
        'MAR': 5.0,
        'OCCP': 3.0,
        'POBP': 6.0,
        'RELP': 4.0,
        'WKHP': 9.0,
        'SEX': 8.0,
        'RAC1P': 1.0,
    }
    REG_SCALE = 0.5
    LOSS_KWARGS = {
        "grad_scale":2.0,
        "weight_scale": 3.0,
        "suppress_scale": 6.0,
        "target_power": 1.0,
    }

    ID_result, OOD_MEAN_result, OOD_WORST_result = main(
        DATASET, TRAIN_VAL_STATE, TEST_STATES,
        FEATURE_INDEX, REMOVED_FEATURE_INDICES, FEATURE_LOSS_WEIGHTS,
        TRAIN_BATCH=TRAIN_BATCH, EVAL_BATCH=EVAL_BATCH, LR=LR, REG_SCALE=REG_SCALE,
        PATIENCE=PATIENCE, REPEAT=REPEAT, MAX_EPOCHS=MAX_EPOCHS,
        DATASET_CONFIG=None, PLOT_TEST_SHAP=False, # too many test states
        MODEL_NAME=MODEL_NAME, LOSS_NAME=LOSS_NAME, LOSS_KWARGS=LOSS_KWARGS,
        device=device, MODEL_SEEDS=[9803, 38224, 8113, 4854, 98825],
    )
    print(f"### Results:\n- ID: {ID_result:.4f}\n- OOD MEAN: {OOD_MEAN_result:.4f}\n- OOD WORST: {OOD_WORST_result:.4f}")
