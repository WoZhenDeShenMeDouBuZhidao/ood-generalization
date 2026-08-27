from typing import Sequence, Tuple, Union

import numpy as np
import torch


def binary_confusion_from_predictions(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    return tp, tn, fp, fn


def binary_confusion_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = logits.argmax(dim=1)
    return binary_confusion_from_predictions(preds, labels)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_balanced_accuracy_macro_f1_from_confusion(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> tuple[float, float]:
    balanced_accuracy = (
        _ratio(tp, tp + fn)
        + _ratio(tn, tn + fp)
    ) / 2.0
    positive_f1 = _ratio(2 * tp, (2 * tp) + fp + fn)
    negative_f1 = _ratio(2 * tn, (2 * tn) + fp + fn)
    return balanced_accuracy, (positive_f1 + negative_f1) / 2.0


def binary_balanced_accuracy_macro_f1(
    labels: Union[Sequence[int], np.ndarray],
    preds: Union[Sequence[int], np.ndarray],
) -> tuple[float, float]:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    if labels_np.size == 0:
        return 0.0, 0.0

    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())
    return binary_balanced_accuracy_macro_f1_from_confusion(tp, tn, fp, fn)
