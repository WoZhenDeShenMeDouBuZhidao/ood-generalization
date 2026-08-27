from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LossOutput = Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


class _ClassReweightedLossMixin:
    def forward(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        if self.reweighting:
            return self._forward_reweighted(model, X, Y)
        return self._forward_unweighted(model, X, Y)


def _present_class_indices(Y: torch.Tensor) -> List[torch.Tensor]:
    return [
        (Y == class_value).nonzero(as_tuple=True)[0]
        for class_value in torch.unique(Y.detach(), sorted=True)
    ]


def _mean_over_present_classes(
    values: torch.Tensor,
    class_indices: List[torch.Tensor],
) -> torch.Tensor:
    return torch.stack([
        values.index_select(0, indices).mean()
        for indices in class_indices
    ]).mean()


def _kept_feature_names(
    feature_index: Dict[int, str],
    removed_feature_indices: List[int],
) -> List[str]:
    removed = set(removed_feature_indices)
    return [
        feature_index[index]
        for index in sorted(feature_index)
        if index not in removed
    ]


def _importance_scale_tensor(
    importance_scale: Optional[List[float]],
    num_features: int,
    device: str,
) -> torch.Tensor:
    if importance_scale is None:
        return torch.ones(num_features, device=device, dtype=torch.float32)
    scale = torch.tensor(importance_scale, device=device, dtype=torch.float32)
    if scale.numel() != num_features:
        raise ValueError(
            f"importance_scale has {scale.numel()} values, expected {num_features}."
        )
    return scale.clamp_min(1e-6)


def logit_margin_gradients(
    logits: torch.Tensor,
    X: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    margin = logits[:, 1] - logits[:, 0]
    return torch.autograd.grad(
        outputs=margin.sum(),
        inputs=X,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]


def first_layer_weight(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "fc1") and hasattr(model.fc1, "weight"):
        return model.fc1.weight
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.weight
    raise ValueError("Could not find a first linear layer weight.")


def feature_gradient_terms(
    feature_names: List[str],
    grad_l2: torch.Tensor,
    grad_probs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    terms = {"total_grad_l2": grad_l2.sum()}
    for index, feature_name in enumerate(feature_names):
        terms[f"{feature_name}_grad_l2"] = grad_l2[index]
        if grad_probs is not None:
            terms[f"{feature_name}_grad_prob"] = grad_probs[index]
    return terms


def feature_probabilities(
    feature_signal: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    return F.softmax(feature_signal.clamp_min(eps).log() / temperature, dim=0)


def feature_weight_terms(
    feature_names: List[str],
    weight_abs: torch.Tensor,
    weight_probs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    terms = {"total_weight_abs": weight_abs.sum()}
    for index, feature_name in enumerate(feature_names):
        terms[f"{feature_name}_weight_abs"] = weight_abs[index]
        if weight_probs is not None:
            terms[f"{feature_name}_weight_prob"] = weight_probs[index]
    return terms


def diagnostic_gradient_terms(
    model: nn.Module,
    X: torch.Tensor,
    feature_names: List[str],
    grad_prob_temperature: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    X_diag = X.detach().requires_grad_(True)
    logits = model(X_diag)
    grads = logit_margin_gradients(
        logits,
        X_diag,
        create_graph=False,
        retain_graph=False,
    )
    model.train(was_training)
    grad_l2 = grads.pow(2).mean(dim=0).clamp_min(eps)
    grad_probs = feature_probabilities(grad_l2, grad_prob_temperature, eps)
    return feature_gradient_terms(feature_names, grad_l2, grad_probs)


def diagnostic_weight_terms(
    model: nn.Module,
    feature_names: List[str],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    weight_abs = (
        first_layer_weight(model)
        .detach()
        .abs()
        .mean(dim=0)
        .clamp_min(eps)
    )
    weight_probs = weight_abs / weight_abs.sum().clamp_min(eps)
    return feature_weight_terms(feature_names, weight_abs, weight_probs)


class CrossEntropyCELoss(_ClassReweightedLossMixin, nn.Module):
    def __init__(self, reweighting: bool = False):
        super().__init__()
        self.reweighting = reweighting

    def _forward_unweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        logits = model(X)
        ce = F.cross_entropy(logits, Y)
        return logits, ce, {"CE_loss": ce, "total_loss": ce}, {}

    def _forward_reweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        logits = model(X)
        class_indices = _present_class_indices(Y)
        ce = _mean_over_present_classes(
            F.cross_entropy(logits, Y, reduction="none"),
            class_indices,
        )
        return logits, ce, {"CE_loss": ce, "total_loss": ce}, {}


class GradientRegularizedCELoss(_ClassReweightedLossMixin, nn.Module):
    """
    CrossEntropyLoss plus plain input-gradient L2 regularization.

    This baseline does not use LLM rankings or feature scores. It simply
    penalizes sensitivity of the positive-class logit to every input feature.
    """

    def __init__(
        self,
        FEATURE_INDEX: Dict[int, str],
        REMOVED_FEATURE_INDICES: List[int],
        FEATURE_LOSS_WEIGHTS: Dict[str, float],
        reg_scale: float = 1.0,
        device: str = "cuda",
        reweighting: bool = False,
        importance_scale: Optional[List[float]] = None,
    ):
        super().__init__()
        self.reweighting = reweighting
        self.reg_scale = reg_scale
        self.feature_names = _kept_feature_names(FEATURE_INDEX, REMOVED_FEATURE_INDICES)
        self.importance_scale = _importance_scale_tensor(
            importance_scale,
            len(self.feature_names),
            device,
        )

    def _gradient_l2_terms(
        self,
        grads: torch.Tensor,
        class_indices: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        scaled_grads = grads * self.importance_scale.view(1, -1)
        if class_indices is None:
            return scaled_grads.pow(2).mean(dim=0)
        return torch.stack([
            scaled_grads.index_select(0, indices).pow(2).mean(dim=0)
            for indices in class_indices
        ]).mean(dim=0)

    def _forward_unweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        ce = F.cross_entropy(logits, Y)
        grad_l2 = self._gradient_l2_terms(logit_margin_gradients(logits, X))
        grad_l2_loss = grad_l2.sum()
        loss = ce + self.reg_scale * grad_l2_loss
        return (
            logits,
            loss,
            self._loss_terms(ce, grad_l2_loss, loss),
            feature_gradient_terms(self.feature_names, grad_l2),
        )

    def _forward_reweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        class_indices = _present_class_indices(Y)
        ce = _mean_over_present_classes(
            F.cross_entropy(logits, Y, reduction="none"),
            class_indices,
        )
        grad_l2 = self._gradient_l2_terms(logit_margin_gradients(logits, X), class_indices)
        grad_l2_loss = grad_l2.sum()
        loss = ce + self.reg_scale * grad_l2_loss
        return (
            logits,
            loss,
            self._loss_terms(ce, grad_l2_loss, loss),
            feature_gradient_terms(self.feature_names, grad_l2),
        )

    def _loss_terms(
        self,
        ce: torch.Tensor,
        grad_l2_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "CE_loss": ce,
            "gradient_l2_loss": self.reg_scale * grad_l2_loss,
            "total_loss": loss,
        }

class LLMAttributionAlignedCELoss(_ClassReweightedLossMixin, nn.Module):
    def __init__(
        self,
        FEATURE_INDEX: Dict[int, str],
        REMOVED_FEATURE_INDICES: List[int],
        FEATURE_LOSS_WEIGHTS: Dict[str, float],
        reg_scale: float = 1.0,
        device: str = "cuda",
        eps: float = 1e-12,
        reweighting: bool = False,
    ):
        super().__init__()
        self.reweighting = reweighting
        self.reg_scale = reg_scale
        self.eps = eps
        self.feature_names = _kept_feature_names(FEATURE_INDEX, REMOVED_FEATURE_INDICES)

        raw_scores = torch.tensor(
            [float(FEATURE_LOSS_WEIGHTS.get(name, 0.0)) for name in self.feature_names],
            device=device,
            dtype=torch.float32,
        )
        raw_scores = raw_scores.clamp_min(0.0)
        if float(raw_scores.sum()) <= self.eps:
            raw_scores = torch.ones_like(raw_scores)
        self.importance_scores = raw_scores

    def _attribution_loss(
        self,
        grads: torch.Tensor,
        X: torch.Tensor,
        class_indices: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attributions = grads * X
        target_attributions = self.importance_scores.view(1, -1) * X
        attribution_probs = F.normalize(attributions, dim=-1)
        target_probs = F.normalize(target_attributions, dim=-1).detach()
        sample_losses = (attribution_probs - target_probs).pow(2).mean(dim=1)
        attribution_abs = attributions.pow(2).mean(dim=0).clamp_min(self.eps)
        attribution_dist = attribution_abs / attribution_abs.sum().clamp_min(self.eps)

        if class_indices is None:
            return attribution_abs, attribution_dist, sample_losses.mean()

        factor = 1.0 / len(class_indices)
        class_attribution_abs = grads.new_zeros((grads.size(1),))
        class_attribution_dist = grads.new_zeros((grads.size(1),))
        attribution_loss = grads.new_zeros(())
        for indices in class_indices:
            indices_attribution_abs = (
                attributions.index_select(0, indices).pow(2).mean(dim=0).clamp_min(self.eps)
            )
            indices_attribution_dist = (
                indices_attribution_abs / indices_attribution_abs.sum().clamp_min(self.eps)
            )
            class_attribution_abs = class_attribution_abs + factor * indices_attribution_abs
            class_attribution_dist = class_attribution_dist + factor * indices_attribution_dist
            attribution_loss = attribution_loss + factor * sample_losses.index_select(0, indices).mean()
        return class_attribution_abs, class_attribution_dist, attribution_loss

    def _forward_unweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        ce = F.cross_entropy(logits, Y)
        attribution_abs, attribution_dist, attribution_loss = self._attribution_loss(
            logit_margin_gradients(logits, X),
            X,
        )
        loss = ce + self.reg_scale * attribution_loss
        return (
            logits,
            loss,
            self._loss_terms(ce, attribution_loss, loss),
            feature_gradient_terms(self.feature_names, attribution_abs, attribution_dist),
        )

    def _forward_reweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        class_indices = _present_class_indices(Y)
        ce = _mean_over_present_classes(
            F.cross_entropy(logits, Y, reduction="none"),
            class_indices,
        )
        attribution_abs, attribution_dist, attribution_loss = self._attribution_loss(
            logit_margin_gradients(logits, X),
            X,
            class_indices,
        )
        loss = ce + self.reg_scale * attribution_loss
        return (
            logits,
            loss,
            self._loss_terms(ce, attribution_loss, loss),
            feature_gradient_terms(self.feature_names, attribution_abs, attribution_dist),
        )

    def _loss_terms(
        self,
        ce: torch.Tensor,
        attribution_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "CE_loss": ce,
            "attribution_mse_loss": self.reg_scale * attribution_loss,
            "total_loss": loss,
        }

class FeatureImportanceTargetCELoss(_ClassReweightedLossMixin, nn.Module):
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
        grad_prob_temperature: float = 1.0,
        reweighting: bool = False,
        reweighting_scope: str = "all",
        importance_scale: Optional[List[float]] = None,
        gradient_feature_groups: Optional[Dict[str, str]] = None,
        gradient_group_weights: Optional[Dict[str, float]] = None,
        gradient_active_groups: Optional[List[str]] = None,
    ):
        super().__init__()
        if reweighting_scope not in {"all", "ce"}:
            raise ValueError(
                f"Unsupported reweighting_scope={reweighting_scope!r}; "
                "expected 'all' or 'ce'."
            )
        self.reweighting = reweighting
        self.reweighting_scope = reweighting_scope
        self.reg_scale = reg_scale
        self.eps = eps
        self.grad_scale = grad_scale
        self.weight_scale = weight_scale
        self.grad_prob_temperature = grad_prob_temperature
        self.feature_names = _kept_feature_names(FEATURE_INDEX, REMOVED_FEATURE_INDICES)
        self.importance_scale = _importance_scale_tensor(
            importance_scale,
            len(self.feature_names),
            device,
        )

        if (gradient_feature_groups is None) != (gradient_group_weights is None):
            raise ValueError(
                "gradient_feature_groups and gradient_group_weights must be provided together."
            )
        if gradient_feature_groups is None:
            gradient_feature_groups = {name: name for name in self.feature_names}
            gradient_group_weights = dict(FEATURE_LOSS_WEIGHTS)

        missing_groups = sorted(set(self.feature_names) - set(gradient_feature_groups))
        if missing_groups:
            raise ValueError(f"Missing gradient feature groups: {missing_groups[:10]}")
        self.gradient_group_names = []
        gradient_group_lookup = {}
        gradient_group_indices = []
        for feature_name in self.feature_names:
            group_name = gradient_feature_groups[feature_name]
            if group_name not in gradient_group_lookup:
                gradient_group_lookup[group_name] = len(self.gradient_group_names)
                self.gradient_group_names.append(group_name)
            gradient_group_indices.append(gradient_group_lookup[group_name])
        missing_group_weights = sorted(
            set(self.gradient_group_names) - set(gradient_group_weights)
        )
        if missing_group_weights:
            raise ValueError(
                f"Missing gradient group weights: {missing_group_weights[:10]}"
            )
        self.gradient_group_indices = torch.tensor(
            gradient_group_indices,
            device=device,
            dtype=torch.long,
        )

        raw_weights = torch.tensor(
            [float(FEATURE_LOSS_WEIGHTS.get(name, 0.0)) for name in self.feature_names],
            device=device,
            dtype=torch.float32,
        )
        self.weight_target_probs = self._target_probabilities(raw_weights)
        raw_group_weights = torch.tensor(
            [float(gradient_group_weights[name]) for name in self.gradient_group_names],
            device=device,
            dtype=torch.float32,
        )
        self.gradient_active_indices = None
        if gradient_active_groups is not None:
            if not gradient_active_groups:
                raise ValueError("gradient_active_groups must not be empty.")
            if len(set(gradient_active_groups)) != len(gradient_active_groups):
                raise ValueError("gradient_active_groups must be unique.")
            unknown_active_groups = sorted(
                set(gradient_active_groups) - set(self.gradient_group_names)
            )
            if unknown_active_groups:
                raise ValueError(
                    "Unknown active gradient groups: "
                    f"{unknown_active_groups[:10]}"
                )
            active_group_set = set(gradient_active_groups)
            self.gradient_active_indices = torch.tensor(
                [
                    index
                    for index, group_name in enumerate(self.gradient_group_names)
                    if group_name in active_group_set
                ],
                device=device,
                dtype=torch.long,
            )
            raw_group_weights = raw_group_weights.index_select(
                0,
                self.gradient_active_indices,
            )
        self.target_probs = self._target_probabilities(raw_group_weights)

    def _target_probabilities(self, raw_weights: torch.Tensor) -> torch.Tensor:
        target_scores = raw_weights.clamp_min(0.0)
        if float(target_scores.sum()) <= self.eps:
            target_scores = torch.ones_like(target_scores)
        return target_scores / target_scores.sum().clamp_min(self.eps)

    def _gradient_distribution(
        self,
        grads: torch.Tensor,
        class_indices: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if class_indices is None:
            grad_l2 = self._aggregate_gradient_signal(self._scaled_grad_l2(grads))
            if self.gradient_active_indices is not None:
                active_grad_probs = feature_probabilities(
                    grad_l2.index_select(0, self.gradient_active_indices),
                    self.grad_prob_temperature,
                    self.eps,
                )
                grad_probs = grad_l2.new_zeros(grad_l2.shape).scatter(
                    0,
                    self.gradient_active_indices,
                    active_grad_probs,
                )
                grad_target_loss = -(
                    self.target_probs * active_grad_probs.log()
                ).sum()
                return grad_l2, grad_probs, grad_target_loss
            grad_probs = feature_probabilities(
                grad_l2,
                self.grad_prob_temperature,
                self.eps,
            )
            grad_target_loss = -(self.target_probs * grad_probs.log()).sum()
            return grad_l2, grad_probs, grad_target_loss

        factor = 1.0 / len(class_indices)
        grad_l2 = grads.new_zeros((len(self.gradient_group_names),))
        grad_probs = grads.new_zeros((len(self.gradient_group_names),))
        grad_target_loss = grads.new_zeros(())
        for indices in class_indices:
            class_grad_l2 = self._aggregate_gradient_signal(
                self._scaled_grad_l2(grads.index_select(0, indices))
            )
            if self.gradient_active_indices is None:
                active_class_grad_probs = feature_probabilities(
                    class_grad_l2,
                    self.grad_prob_temperature,
                    self.eps,
                )
                class_grad_probs = active_class_grad_probs
            else:
                active_class_grad_probs = feature_probabilities(
                    class_grad_l2.index_select(0, self.gradient_active_indices),
                    self.grad_prob_temperature,
                    self.eps,
                )
                class_grad_probs = class_grad_l2.new_zeros(
                    class_grad_l2.shape
                ).scatter(
                    0,
                    self.gradient_active_indices,
                    active_class_grad_probs,
                )
            grad_l2 = grad_l2 + factor * class_grad_l2
            grad_probs = grad_probs + factor * class_grad_probs
            grad_target_loss = grad_target_loss + factor * (
                -(self.target_probs * active_class_grad_probs.log()).sum()
            )
        return grad_l2, grad_probs, grad_target_loss

    def _scaled_grad_l2(self, grads: torch.Tensor) -> torch.Tensor:
        scaled_grads = grads * self.importance_scale.view(1, -1)
        return scaled_grads.pow(2).mean(dim=0).clamp_min(self.eps)

    def _aggregate_gradient_signal(self, feature_signal: torch.Tensor) -> torch.Tensor:
        group_signal = feature_signal.new_zeros((len(self.gradient_group_names),))
        return group_signal.scatter_add(
            0,
            self.gradient_group_indices,
            feature_signal,
        ).clamp_min(self.eps)

    def diagnostic_gradient_terms(
        self,
        model: nn.Module,
        X: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()
        X_diag = X.detach().requires_grad_(True)
        logits = model(X_diag)
        grads = logit_margin_gradients(
            logits,
            X_diag,
            create_graph=False,
            retain_graph=False,
        )
        model.train(was_training)
        grad_l2 = self._aggregate_gradient_signal(self._scaled_grad_l2(grads))
        if self.gradient_active_indices is None:
            grad_probs = feature_probabilities(
                grad_l2,
                self.grad_prob_temperature,
                self.eps,
            )
        else:
            active_grad_probs = feature_probabilities(
                grad_l2.index_select(0, self.gradient_active_indices),
                self.grad_prob_temperature,
                self.eps,
            )
            grad_probs = grad_l2.new_zeros(grad_l2.shape).scatter(
                0,
                self.gradient_active_indices,
                active_grad_probs,
            )
        return feature_gradient_terms(
            self.gradient_group_names,
            grad_l2,
            grad_probs,
        )

    def _weight_distribution(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_abs = (
            first_layer_weight(model).abs().mean(dim=0) * self.importance_scale
        ).clamp_min(self.eps)
        weight_probs = weight_abs / weight_abs.sum().clamp_min(self.eps)
        weight_target_loss = -(self.weight_target_probs * weight_probs.log()).sum()
        return weight_abs, weight_probs, weight_target_loss

    def _forward_unweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        ce = F.cross_entropy(logits, Y)
        grad_l2, grad_probs, grad_target_loss = self._gradient_distribution(
            logit_margin_gradients(logits, X)
        )
        return self._finish_forward(
            model,
            logits,
            ce,
            grad_l2,
            grad_probs,
            grad_target_loss,
        )

    def _forward_reweighted(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> LossOutput:
        X = X.detach().requires_grad_(True)
        logits = model(X)
        class_indices = _present_class_indices(Y)
        ce = _mean_over_present_classes(
            F.cross_entropy(logits, Y, reduction="none"),
            class_indices,
        )
        regularizer_class_indices = (
            class_indices if self.reweighting_scope == "all" else None
        )
        grad_l2, grad_probs, grad_target_loss = self._gradient_distribution(
            logit_margin_gradients(logits, X),
            regularizer_class_indices,
        )
        return self._finish_forward(
            model,
            logits,
            ce,
            grad_l2,
            grad_probs,
            grad_target_loss,
            regularizer_class_indices,
        )

    def _finish_forward(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        ce: torch.Tensor,
        grad_l2: torch.Tensor,
        grad_probs: torch.Tensor,
        grad_target_loss: torch.Tensor,
        class_indices: Optional[List[torch.Tensor]] = None,
    ) -> LossOutput:
        weight_abs, weight_probs, weight_target_loss = self._weight_distribution(model)
        if class_indices is not None:
            # The global first-layer term is replicated in each class objective.
            # Class balancing is therefore explicit but algebraically invariant.
            weight_target_loss = torch.stack([
                weight_target_loss for _ in class_indices
            ]).mean()
        reg_loss = (
            self.grad_scale * grad_target_loss
            + self.weight_scale * weight_target_loss
        )
        loss = ce + self.reg_scale * reg_loss

        loss_terms = {
            "CE_loss": ce,
            "grad_target_loss": grad_target_loss,
            "weight_target_loss": weight_target_loss,
            "total_loss": loss,
        }
        grad_terms = feature_gradient_terms(
            self.gradient_group_names,
            grad_l2,
            grad_probs,
        )
        grad_terms.update(feature_weight_terms(self.feature_names, weight_abs, weight_probs))
        return logits, loss, loss_terms, grad_terms
