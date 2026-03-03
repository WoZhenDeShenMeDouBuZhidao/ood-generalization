import os
import random
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
    repeat_i: int,
    FEATURE_INDEX: Dict[int, str],
    REMOVED_FEATURE_INDICES: List[str],
    EPOCH: int,
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float]
) -> None:
    os.makedirs("./plots", exist_ok=True)

    epochs = range(1, EPOCH + 1)
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xticks(range(0, EPOCH + 1, EPOCH // 10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Validation Acc')
    plt.title('Accuracy')
    plt.xticks(range(0, EPOCH + 1, EPOCH // 10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    rmfeature_names = ", ".join([FEATURE_INDEX[feat_i] for feat_i in REMOVED_FEATURE_INDICES])
    plot_name = "All features" if len(REMOVED_FEATURE_INDICES) == 0 else f"Remove {rmfeature_names}"
    plt.savefig(f"./plots/{plot_name} (repeat {repeat_i}).png")
    plt.close()


def plot_accdelta_bars(
    rmfeature_accdelta: Dict[str, Dict[str, float]],
    ID_base: float, OOD_MEAN_base: float, OOD_WORST_base: float
) -> None:
    os.makedirs("./plots", exist_ok=True)
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
    fig.savefig("./plots/accdelta_bars.png", dpi=200)
    plt.close(fig)
