from typing import Sequence, Tuple, Union

import numpy as np
import torch


def binary_counts_from_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    correct = (preds == labels).sum()
    return correct, tp, fp, fn


def binary_counts_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = logits.argmax(dim=1)
    return binary_counts_from_predictions(preds, labels)


def binary_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def binary_accuracy_f1(
    labels: Union[Sequence[int], np.ndarray],
    preds: Union[Sequence[int], np.ndarray],
) -> tuple[float, float]:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    if labels_np.size == 0:
        return 0.0, 0.0

    acc = float((preds_np == labels_np).mean())
    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())
    return acc, binary_f1_from_counts(tp, fp, fn)
