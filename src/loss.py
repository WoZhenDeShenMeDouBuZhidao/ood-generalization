import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


def _get_first_layer_weight(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "fc1") and hasattr(model.fc1, "weight"):
        return model.fc1.weight

    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.weight

    raise ValueError("Could not find a first linear layer weight for regularization.")


class FeatureGradCELoss(nn.Module):
    """
    CrossEntropyLoss + feature-wise input-gradient L2 regularization.

    Regularization term:
        sum_j w_j * E_batch[ || d(logit_pos) / d x_j ||_2^2 ]
    where logit_pos = logits[:, 1] for 2-class output.

    Notes:
    - FEATURE_INDEX maps original feature index -> feature name.
    - REMOVED_FEATURE_INDICES removes original indices before feeding model,
      so we rebuild the mapping from original indices to new X indices.
    - If a feature is removed, it is automatically ignored in the reg term.
    """

    def __init__(
        self,
        FEATURE_INDEX: Dict[int, str],
        REMOVED_FEATURE_INDICES: List[int],
        FEATURE_LOSS_WEIGHTS: Dict[str, float],
        reg_scale: float = 1.0, device: str = "cuda",
        eps: float = 1e-12, 
    ):
        super().__init__()
        self.FEATURE_INDEX = FEATURE_INDEX
        self.REMOVED_FEATURE_INDICES = set(REMOVED_FEATURE_INDICES)
        self.FEATURE_LOSS_WEIGHTS = FEATURE_LOSS_WEIGHTS
        self.reg_scale = reg_scale
        self.eps = eps
        self.device = device

        # Build mapping: current_x_index -> feature_name, after removal
        orig_indices_sorted = sorted(self.FEATURE_INDEX.keys())
        kept_orig_indices = [i for i in orig_indices_sorted if i not in self.REMOVED_FEATURE_INDICES]
        self.curidx_to_name = [self.FEATURE_INDEX[i] for i in kept_orig_indices]  # length == x_dim

        # calculate valid ranking pairs
        self.feature_weights = torch.tensor(
            [float(self.FEATURE_LOSS_WEIGHTS.get(feat_name, 0.0)) for feat_name in self.curidx_to_name],
            device=self.device,
            dtype=torch.float32,
        )
        self.rank_mask = self.feature_weights.unsqueeze(1) > self.feature_weights.unsqueeze(0)  # (d, d)

    def forward(
        self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        model: the MLP
        X: (B, d) float
        Y: (B,) long
        """
        # Make X differentiable for input-gradient regularization
        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        logits = model(X)  # (B, 2)
        ce = F.cross_entropy(logits, Y)

        # Use positive-class logit for gradient regularization (binary classification with 2 logits)
        logit_pos = logits[:, 1]  # (B,)
        grads = torch.autograd.grad(
            outputs=logit_pos.sum(),
            inputs=X,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # (B, d)

        # Feature-wise gradient L2 ranking penalty
        loss_terms = {}
        grad_sq = grads.pow(2)
        grad_abs = grads.abs()

        # For each sample, compare features across different weight groups.
        # If weight_j > weight_k but |grad_j| < |grad_k|, add |grad_k| - |grad_j| to the rank loss.
        # Example:
        #   if WKHP has weight 1.0 and all other features have weight 0.0,
        #   then the valid pairs are (WKHP, other_feature).
        #   For a sample, if |grad(WKHP)| = 0.2 and |grad(AGEP)| = 0.5,
        #   this pair contributes 0.5 - 0.2 = 0.3 to the rank loss.
        # The per-sample loss is normalized by the number of valid pairs,
        # then averaged over the batch.
        if self.rank_mask.any():
            valid_pair_count = self.rank_mask.sum().to(grads.dtype)
            higher_rank_grad_abs = grad_abs.unsqueeze(2)  # (B, d, 1), index j
            lower_rank_grad_abs = grad_abs.unsqueeze(1)   # (B, 1, d), index k
            violation_magnitude = F.relu(
                lower_rank_grad_abs - higher_rank_grad_abs
            )  # (B, d, d), [b, j, k] = max(|grad_k| - |grad_j|, 0)
            grad_abs_rank_loss = (
                violation_magnitude
                * self.rank_mask.unsqueeze(0).to(grads.dtype)
            ).sum(dim=(1, 2)).div(valid_pair_count).mean()
        else:
            grad_abs_rank_loss = grads.new_zeros(())

        grad_terms = {"total_grad_l2": 0.0}
        for j, feat_name in enumerate(self.curidx_to_name):
            grad_l2 = grad_sq[:, j].mean() # mean over batch of squared gradient for feature j
            grad_terms[f"{feat_name}_grad_l2"] = grad_l2
            grad_terms[f"total_grad_l2"] += grad_l2

        loss = ce + self.reg_scale * grad_abs_rank_loss
        loss_terms[f"CE_loss"] = ce
        loss_terms["grad_abs_rank_loss"] = self.reg_scale * grad_abs_rank_loss
        loss_terms[f"total_loss"] = loss
        return logits, loss, loss_terms, grad_terms


class FirstLayerWeightCELoss(nn.Module):
    """
    CrossEntropyLoss + feature-wise first-layer weight ranking regularization.

    Regularization term:
        sum_{j,k: w_j > w_k} max(|W_k| - |W_j|, 0)
    where |W_j| is the mean absolute value of the first-layer weights connected
    to input feature j.
    """

    def __init__(
        self,
        FEATURE_INDEX: Dict[int, str],
        REMOVED_FEATURE_INDICES: List[int],
        FEATURE_LOSS_WEIGHTS: Dict[str, float],
        reg_scale: float = 1.0, device: str = "cuda",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.FEATURE_INDEX = FEATURE_INDEX
        self.REMOVED_FEATURE_INDICES = set(REMOVED_FEATURE_INDICES)
        self.FEATURE_LOSS_WEIGHTS = FEATURE_LOSS_WEIGHTS
        self.reg_scale = reg_scale
        self.eps = eps
        self.device = device

        orig_indices_sorted = sorted(self.FEATURE_INDEX.keys())
        kept_orig_indices = [i for i in orig_indices_sorted if i not in self.REMOVED_FEATURE_INDICES]
        self.curidx_to_name = [self.FEATURE_INDEX[i] for i in kept_orig_indices]

        self.feature_weights = torch.tensor(
            [float(self.FEATURE_LOSS_WEIGHTS.get(feat_name, 0.0)) for feat_name in self.curidx_to_name],
            device=self.device,
            dtype=torch.float32,
        )
        self.rank_mask = self.feature_weights.unsqueeze(1) > self.feature_weights.unsqueeze(0)

    def forward(
        self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float]]:
        logits = model(X)
        ce = F.cross_entropy(logits, Y)

        first_layer_weight = _get_first_layer_weight(model)
        weight_abs = first_layer_weight.abs().mean(dim=0)

        loss_terms = {}

        if self.rank_mask.any():
            valid_pair_count = self.rank_mask.sum().to(weight_abs.dtype)
            higher_rank_weight_abs = weight_abs.unsqueeze(1)
            lower_rank_weight_abs = weight_abs.unsqueeze(0)
            violation_magnitude = F.relu(
                lower_rank_weight_abs - higher_rank_weight_abs
            )
            weight_abs_rank_loss = (
                violation_magnitude
                * self.rank_mask.to(weight_abs.dtype)
            ).sum().div(valid_pair_count)
        else:
            weight_abs_rank_loss = weight_abs.new_zeros(())

        grad_terms = {"total_weight_abs": 0.0}
        for j, feat_name in enumerate(self.curidx_to_name):
            feature_weight_abs = weight_abs[j]
            grad_terms[f"{feat_name}_weight_abs"] = feature_weight_abs
            grad_terms["total_weight_abs"] += feature_weight_abs

        loss = ce + self.reg_scale * weight_abs_rank_loss
        loss_terms["CE_loss"] = ce
        loss_terms["weight_abs_rank_loss"] = self.reg_scale * weight_abs_rank_loss
        loss_terms["total_loss"] = loss
        return logits, loss, loss_terms, grad_terms


class FeatureImportanceTargetCELoss(nn.Module):
    """
    CrossEntropyLoss + target-distribution regularization on feature importance.

    This loss aligns both:
    - feature-wise input-gradient L2
    - feature-wise first-layer mean absolute weight

    to a target feature-importance distribution derived from positive
    FEATURE_LOSS_WEIGHTS. Features with non-positive weights are treated as
    suppressed by default when at least one positive target exists.
    """

    def __init__(
        self,
        FEATURE_INDEX: Dict[int, str],
        REMOVED_FEATURE_INDICES: List[int],
        FEATURE_LOSS_WEIGHTS: Dict[str, float],
        reg_scale: float = 1.0,
        device: str = "cuda",
        eps: float = 1e-12,
        grad_scale: float = 1.0,
        weight_scale: float = 1.0,
        suppress_scale: float = 1.0,
        target_power: float = 1.0,
    ):
        super().__init__()
        self.FEATURE_INDEX = FEATURE_INDEX
        self.REMOVED_FEATURE_INDICES = set(REMOVED_FEATURE_INDICES)
        self.FEATURE_LOSS_WEIGHTS = FEATURE_LOSS_WEIGHTS
        self.reg_scale = reg_scale
        self.device = device
        self.eps = eps
        self.grad_scale = grad_scale
        self.weight_scale = weight_scale
        self.suppress_scale = suppress_scale
        self.target_power = target_power

        orig_indices_sorted = sorted(self.FEATURE_INDEX.keys())
        kept_orig_indices = [i for i in orig_indices_sorted if i not in self.REMOVED_FEATURE_INDICES]
        self.curidx_to_name = [self.FEATURE_INDEX[i] for i in kept_orig_indices]

        raw_feature_weights = torch.tensor(
            [float(self.FEATURE_LOSS_WEIGHTS.get(feat_name, 0.0)) for feat_name in self.curidx_to_name],
            device=self.device,
            dtype=torch.float32,
        )
        target_scores = raw_feature_weights.clamp_min(0.0).pow(self.target_power)
        if float(target_scores.sum()) <= self.eps:
            target_scores = torch.ones_like(target_scores)
        self.target_probs = target_scores / target_scores.sum().clamp_min(self.eps)
        self.suppressed_mask = raw_feature_weights <= 0.0

    def forward(
        self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float]]:
        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        logits = model(X)
        ce = F.cross_entropy(logits, Y)

        logit_pos = logits[:, 1]
        grads = torch.autograd.grad(
            outputs=logit_pos.sum(),
            inputs=X,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_l2 = grads.pow(2).mean(dim=0).clamp_min(self.eps)
        grad_probs = grad_l2 / grad_l2.sum().clamp_min(self.eps)

        first_layer_weight = _get_first_layer_weight(model)
        weight_abs = first_layer_weight.abs().mean(dim=0).clamp_min(self.eps)
        weight_probs = weight_abs / weight_abs.sum().clamp_min(self.eps)

        grad_target_loss = -(self.target_probs * grad_probs.log()).sum()
        weight_target_loss = -(self.target_probs * weight_probs.log()).sum()

        if self.suppressed_mask.any():
            suppressed_prob_loss = grad_probs[self.suppressed_mask].sum() + weight_probs[self.suppressed_mask].sum()
        else:
            suppressed_prob_loss = grad_probs.new_zeros(())

        reg_loss = (
            self.grad_scale * grad_target_loss
            + self.weight_scale * weight_target_loss
            + self.suppress_scale * suppressed_prob_loss
        )
        loss = ce + self.reg_scale * reg_loss

        loss_terms = {
            "CE_loss": ce,
            "grad_target_loss": self.reg_scale * self.grad_scale * grad_target_loss,
            "weight_target_loss": self.reg_scale * self.weight_scale * weight_target_loss,
            "suppressed_prob_loss": self.reg_scale * self.suppress_scale * suppressed_prob_loss,
            "total_loss": loss,
        }

        grad_terms = {
            "total_grad_l2": grad_l2.sum(),
            "total_weight_abs": weight_abs.sum(),
            "grad_prob_on_suppressed": (
                grad_probs[self.suppressed_mask].sum() if self.suppressed_mask.any() else grad_probs.new_zeros(())
            ),
            "weight_prob_on_suppressed": (
                weight_probs[self.suppressed_mask].sum() if self.suppressed_mask.any() else weight_probs.new_zeros(())
            ),
        }
        for j, feat_name in enumerate(self.curidx_to_name):
            grad_terms[f"{feat_name}_grad_l2"] = grad_l2[j]
            grad_terms[f"{feat_name}_grad_prob"] = grad_probs[j]
            grad_terms[f"{feat_name}_weight_abs"] = weight_abs[j]
            grad_terms[f"{feat_name}_weight_prob"] = weight_probs[j]

        return logits, loss, loss_terms, grad_terms
