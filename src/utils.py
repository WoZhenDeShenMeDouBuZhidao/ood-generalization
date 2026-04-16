import os
import random
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_training_curves(
    dataset: str, ID_acc: float, OOD_MEAN_acc: float, OOD_WORST_acc: float,
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[str],
    repeat_i: int, EPOCH: int, PATIENCE: int,
    train_losses: Dict[str, List[float]], val_losses: Dict[str, List[float]],
    train_accs: List[float], val_accs: List[float],
    train_grads: Dict[str, List[float]], date: str,
) -> None:
    plot_path = f"./{dataset}/plots/curves"
    os.makedirs(plot_path, exist_ok=True)

    epochs = range(1, EPOCH + 1)
    best_epoch = EPOCH - PATIENCE
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"ID: {ID_acc:.4f}, OOD MEAN: {OOD_MEAN_acc:.4f}, OOD WORST: {OOD_WORST_acc:.4f}")
    active_feature_names = [
        FEATURE_INDEX[idx]
        for idx in sorted(FEATURE_INDEX)
        if idx not in REMOVED_FEATURE_INDICES
    ]

    def _plot_metric_group(ax, title: str, suffix: str, ylabel: str) -> None:
        ax.axvline(x=best_epoch, color='gray', linestyle='--')
        has_curve = False
        for feat_name in active_feature_names:
            grad_name = f"{feat_name}_{suffix}"
            if grad_name not in train_grads:
                continue
            grad_list = train_grads[grad_name]
            ax.plot(
                epochs,
                grad_list,
                label=f"{feat_name} (best={grad_list[best_epoch]:.4f}, last={grad_list[-1]:.4f})"
            )
            has_curve = True
        ax.set_xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if has_curve:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

    plt.subplot(2,4,1)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    for loss_name, loss_list in train_losses.items():
        plt.plot(epochs, loss_list, label=f"{loss_name} (best={loss_list[best_epoch]:.4f}, last={loss_list[-1]:.4f})")
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss')
    plt.legend()

    plt.subplot(2,4,2)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    for loss_name, loss_list in val_losses.items():
        plt.plot(epochs, loss_list, label=f"{loss_name} (best={loss_list[best_epoch]:.4f}, last={loss_list[-1]:.4f})")
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Valid Loss')
    plt.title('Valid Loss')
    plt.legend()

    plt.subplot(2,4,3)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    plt.plot(epochs, train_accs, label=f'Train Acc (best={train_accs[best_epoch]:.4f}, last={train_accs[-1]:.4f})')
    plt.plot(epochs, val_accs, label=f'Validation Acc (best={val_accs[best_epoch]:.4f}, last={val_accs[-1]:.4f})')
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2,4,4)
    plt.axvline(x=best_epoch, color='gray', linestyle='--')
    summary_terms = [
        "total_grad_l2",
        "total_weight_abs",
        "grad_prob_on_suppressed",
        "weight_prob_on_suppressed",
    ]
    has_summary = False
    for grad_name in summary_terms:
        if grad_name not in train_grads:
            continue
        grad_list = train_grads[grad_name]
        plt.plot(epochs, grad_list, label=f"{grad_name} (best={grad_list[best_epoch]:.4f}, last={grad_list[-1]:.4f})")
        has_summary = True
    plt.xticks(range(0, EPOCH + 1, max(EPOCH // 10, 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Summary')
    plt.title('Summary Terms')
    if has_summary:
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, "N/A", ha="center", va="center", transform=plt.gca().transAxes)

    _plot_metric_group(plt.subplot(2,4,5), "Feature Grad L2", "grad_l2", "Grad L2")
    _plot_metric_group(plt.subplot(2,4,6), "Feature Grad Prob", "grad_prob", "Grad Prob")
    _plot_metric_group(plt.subplot(2,4,7), "Feature Weight Abs", "weight_abs", "Weight Abs")
    _plot_metric_group(plt.subplot(2,4,8), "Feature Weight Prob", "weight_prob", "Weight Prob")

    plt.tight_layout()
    rmfeature_names = ", ".join([FEATURE_INDEX[feat_i] for feat_i in REMOVED_FEATURE_INDICES])
    plot_name = "All features" if len(REMOVED_FEATURE_INDICES) == 0 else f"Remove {rmfeature_names}"
    plt.savefig(f"{plot_path}/{plot_name} (repeat={repeat_i + 1}, datetime={date}).png")
    plt.close()


def plot_accdelta_bars(
    dataset: str, rmfeature_accdelta: Dict[str, Dict[str, float]],
    ID_base: float, OOD_MEAN_base: float, OOD_WORST_base: float, REPEAT: int,
) -> None:
    plot_path = f"./{dataset}/plots/accdelta"
    os.makedirs(plot_path, exist_ok=True)
    if not rmfeature_accdelta:
        return

    # Rank removed features by OOD WORST (best-to-worst from left to right).
    sorted_items = sorted(
        rmfeature_accdelta.items(),
        key=lambda x: x[1].get("OOD WORST", 0.0),
        reverse=True
    )

    features = [feat for feat, _ in sorted_items]
    id_vals = [vals.get("ID", 0.0) for _, vals in sorted_items]
    ood_mean_vals = [vals.get("OOD MEAN", 0.0) for _, vals in sorted_items]
    ood_worst_vals = [vals.get("OOD WORST", 0.0) for _, vals in sorted_items]

    x = np.arange(len(features))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_id = ax.bar(x - width, id_vals, width=width, label=f"ID (base acc = {ID_base:.4f})")
    bars_ood_mean = ax.bar(x, ood_mean_vals, width=width, label=f"OOD MEAN (base acc = {OOD_MEAN_base:.4f})")
    bars_ood_worst = ax.bar(x + width, ood_worst_vals, width=width, label=f"OOD WORST (base acc = {OOD_WORST_base:.4f})")

    ax.axhline(0.0, color="C0", linewidth=1.2)
    ax.set_title("Relative Accuracy vs. Removed Feature")
    ax.set_xlabel("Removed Feature")
    ax.set_ylabel("Δ Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right")
    ax.legend(loc="upper right")

    all_vals = id_vals + ood_mean_vals + ood_worst_vals
    max_abs = max(abs(v) for v in all_vals) if all_vals else 0.0
    y_pad = max(0.0004, max_abs * 0.06)

    def _annotate_bars(bar_container):
        for bar in bar_container:
            h = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = h + y_pad if h >= 0 else h - y_pad
            va = "bottom" if h >= 0 else "top"
            ax.text(xpos, ypos, f"{h:+.4f}", ha="center", va=va, fontsize=12)

    _annotate_bars(bars_id)
    _annotate_bars(bars_ood_mean)
    _annotate_bars(bars_ood_worst)

    ax.set_ylim(min(all_vals) - 3 * y_pad, max(all_vals) + 3 * y_pad)
    fig.tight_layout()
    fig.savefig(f"{plot_path}/accdelta_bars_repeat{REPEAT}.png", dpi=200)
    plt.close(fig)


def plot_shap_values(
    dataset: str, TRAIN_VAL_GROUP: str, TEST_GROUPS: List[str],
    FEATURE_INDEX: Dict[int, str], REMOVED_FEATURE_INDICES: List[int],
    shap_values: List[np.ndarray], repeat_i: int, date: str,
):
    plot_path = f"./{dataset}/plots/shap"
    os.makedirs(plot_path, exist_ok=True)

    active_feature_names = [
        name for idx, name in FEATURE_INDEX.items()
        if idx not in REMOVED_FEATURE_INDICES
    ]
    states = [TRAIN_VAL_GROUP] + TEST_GROUPS[:max(len(shap_values) - 1, 0)]

    for state, shap_value in zip(states, shap_values):
        shap_array = np.asarray(shap_value)
        if shap_array.ndim == 3:
            # For binary classification, use class 1 attribution by default.
            class_idx = 1 if shap_array.shape[-1] > 1 else 0
            shap_array = shap_array[:, :, class_idx]
        elif shap_array.ndim != 2:
            raise ValueError(
                f"Expected SHAP values with 2 or 3 dims, got shape {shap_array.shape} for state {state}."
            )

        if shap_array.shape[1] != len(active_feature_names):
            raise ValueError(
                f"Feature count mismatch for state {state}: "
                f"{shap_array.shape[1]} SHAP columns vs {len(active_feature_names)} feature names."
            )

        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [active_feature_names[idx] for idx in sorted_indices]
        sorted_values = mean_abs_shap[sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(sorted_features))
        bars = ax.barh(y_pos, sorted_values, color="C0")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("Feature")
        ax.set_title(f"SHAP Feature Importance ({state})")

        max_val = sorted_values.max() if len(sorted_values) > 0 else 0.0
        x_pad = max(1e-4, max_val * 0.02)
        for bar, value in zip(bars, sorted_values):
            ax.text(
                value + x_pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                fontsize=10,
            )

        ax.set_xlim(0.0, max_val + 6 * x_pad if max_val > 0 else 1.0)
        fig.tight_layout()
        fig.savefig(f"{plot_path}/shap_{state} (repeat={repeat_i + 1}, datetime={date}).png", dpi=200)
        plt.close(fig)
