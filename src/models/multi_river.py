"""Shared multi-river graph GRU and an optional Manning consistency loss.

The consistency loss is a rectangular-channel algebraic regularizer. It does
not solve the Saint-Venant equations or sediment transport PDEs and is not a
full PDE PINN. Real operation requires measured geometry and calibration.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.multi_river import FEATURE_NAMES, TARGET_NAMES, depth_from_discharge, reach_geometry


class FeatureScaler:
    """Per-feature training statistics shared across time and generic reaches.

    fit() accepts raw [time,node,6] or windowed arrays. Prefer raw training hours
    to avoid weighting repeated context observations. Transform and inverse
    preserve tensor gradients, dtype and device when passed Torch tensors.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        values = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        if values.ndim < 2 or values.shape[-1] != 6 or not values.size or not np.isfinite(values).all():
            raise ValueError("Scaler needs finite nonempty training features [...,6]")
        values = values.reshape(-1, 6).astype(np.float64)
        self.mean = values.mean(axis=0)
        self.std = np.maximum(values.std(axis=0), 1e-6)
        return self

    def _statistics(self, x, n_features):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted")
        if n_features not in (3, 6):
            raise ValueError("Expected three targets or six input features")
        mean, std = self.mean[:n_features], self.std[:n_features]
        if torch.is_tensor(x):
            return torch.as_tensor(mean, dtype=x.dtype, device=x.device), torch.as_tensor(std, dtype=x.dtype, device=x.device)
        return mean, std

    def transform(self, x):
        if not torch.is_tensor(x):
            x = np.asarray(x)
        mean, std = self._statistics(x, x.shape[-1])
        result = (x - mean) / std
        return result if torch.is_tensor(result) else result.astype(np.float32)

    def inverse(self, y):
        """Restore the first three outputs (or all six inputs) to SI units."""
        if not torch.is_tensor(y):
            y = np.asarray(y)
        mean, std = self._statistics(y, y.shape[-1])
        return y * std + mean

    def to_dict(self):
        if self.mean is None:
            raise ValueError("Scaler has not been fitted")
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "feature_names": list(FEATURE_NAMES), "fit_scope": "training_only"}

    @classmethod
    def from_dict(cls, value):
        scaler = cls()
        scaler.mean = np.asarray(value["mean"], dtype=np.float64)
        scaler.std = np.asarray(value["std"], dtype=np.float64)
        if scaler.mean.shape != (6,) or scaler.std.shape != (6,) or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.std).all() or (scaler.std <= 0).any():
            raise ValueError("Invalid serialized feature statistics")
        if "feature_names" in value and tuple(value["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Serialized feature order differs from model contract")
        return scaler


class RiverGraphGRU(nn.Module):
    """Directed upstream aggregation followed by one GRU shared across reaches.

    Input [batch, history, nodes, 6]; output [batch, horizon, nodes, 3]. Residuals
    correct the last observed normalized depth/flow/concentration. A zero-initial
    head begins as persistence while allowing genuine learned future outputs.
    Physics/no-physics variants use exactly this same model architecture.
    """

    def __init__(self, n_nodes=5, n_features=6, hidden_size=32, horizon=6):
        super().__init__()
        if min(n_nodes, hidden_size, horizon) < 1 or n_features != 6:
            raise ValueError("Positive model sizes and six input features required")
        self.n_nodes, self.n_features, self.hidden_size, self.horizon = n_nodes, n_features, hidden_size, horizon
        adjacency = torch.eye(n_nodes)
        for destination in range(1, n_nodes):
            adjacency[destination, destination - 1] = 1
        adjacency /= adjacency.sum(dim=1, keepdim=True)
        self.register_buffer("adjacency", adjacency)
        self.gru = nn.GRU(2 * n_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon * 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[2:] != (self.n_nodes, self.n_features) or x.shape[1] < 1:
            raise ValueError("Expected [batch,history,nodes,6] model inputs")
        batch, history, _, _ = x.shape
        upstream = torch.einsum("ij,btjf->btif", self.adjacency, x)
        aggregated = torch.cat((x, upstream), dim=-1)
        node_sequences = aggregated.permute(0, 2, 1, 3).reshape(batch * self.n_nodes, history, 2 * self.n_features)
        _, hidden = self.gru(node_sequences)
        residual = self.output(hidden[-1]).reshape(batch, self.n_nodes, self.horizon, 3).permute(0, 2, 1, 3)
        return x[:, -1, :, :3].unsqueeze(1) + residual


def physics_penalty(pred_normalized, scaler: FeatureScaler, profile: dict):
    """Differentiable, dimensionless Manning + nonnegative-output penalties.

    Hydraulic consistency is computed in physical units and divided by scenario
    discharge. Negative depth/flow/concentration penalties use scenario scales.
    Only the future predictions participate; future forcing/targets are unused.
    """
    if not torch.is_tensor(pred_normalized) or pred_normalized.shape[-1] != 3:
        raise ValueError("Expected Torch normalized target predictions [...,nodes,3]")
    physical = scaler.inverse(pred_normalized)
    n_nodes = physical.shape[-2]
    widths, slopes, roughness = reach_geometry(profile, n_nodes)
    kwargs = {"dtype": physical.dtype, "device": physical.device}
    width = torch.as_tensor(widths, **kwargs)
    slope = torch.as_tensor(slopes, **kwargs)
    rough = torch.as_tensor(roughness, **kwargs)
    h, q, _ = physical.unbind(dim=-1)
    positive_depth = h.clamp_min(1e-6)
    area = width * positive_depth
    radius = area / (width + 2 * positive_depth)
    manning_q = area * radius.pow(2.0 / 3.0) * slope.sqrt() / rough
    q_scale = float(profile["base_discharge"])
    h_scale = float(depth_from_discharge(np.asarray(q_scale), widths[0], slopes[0], roughness[0]))
    scales = torch.as_tensor([max(h_scale, 1e-3), max(q_scale, 1e-3), max(float(profile["sediment_scale"]), 1e-6)], **kwargs)
    hydraulic = ((q - manning_q) / q_scale).square().mean()
    negative = (torch.relu(-physical) / scales).square().mean()
    return hydraulic + negative


def predict(model: RiverGraphGRU, x) -> np.ndarray:
    """Predict normalized targets without gradients; restore original train mode."""
    parameter = next(model.parameters())
    inputs = torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(inputs).detach().cpu().numpy()
    finally:
        model.train(training)


def metrics(pred_physical, target_physical) -> dict:
    """Per-target SI errors, NSE and range-normalized RMSE over hours/reaches.

    bias = mean(prediction - target); nrmse = RMSE / target observed range.
    Constant targets return None for NSE/NRMSE, preserving strict JSON (no NaN).
    Values aggregate all dimensions before the final three target features.
    """
    prediction, target = np.asarray(pred_physical, dtype=np.float64), np.asarray(target_physical, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-1] != 3 or not target.size:
        raise ValueError("Metrics need matching nonempty [...,3] arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metrics need finite predictions and targets")
    prediction, target = prediction.reshape(-1, 3), target.reshape(-1, 3)
    result = {}
    for index, name in enumerate(TARGET_NAMES):
        truth = target[:, index]
        error = prediction[:, index] - truth
        mse = float(np.mean(error**2))
        variance_sum = float(np.sum((truth - truth.mean())**2))
        observed_range = float(np.ptp(truth))
        result[name] = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(mse)), "nse": float(1 - np.sum(error**2) / variance_sum) if variance_sum > 1e-12 else None, "bias": float(error.mean()), "nrmse": float(np.sqrt(mse) / observed_range) if observed_range > 1e-12 else None}
    return result
