"""Open, single-station daily discharge GRU; no physics or synthetic features."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PROTOCOL_VERSION = "observed_daily_gru_v2"


class DischargeScaler:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, raw_training_values):
        values = np.asarray(raw_training_values, dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Finite nonempty raw training values required")
        self.mean, self.std = float(values.mean()), max(float(values.std()), 1e-6)
        return self

    def _check(self):
        if self.mean is None or self.std is None or not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Invalid/unfitted discharge scaler")

    def transform(self, values):
        self._check()
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, values):
        self._check()
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self):
        self._check()
        return {"mean": self.mean, "std": self.std, "fit_scope": "raw_training_period_only", "variable": "discharge", "unit": "m³/s"}

    @classmethod
    def from_dict(cls, value):
        scaler = cls(float(value["mean"]), float(value["std"]))
        scaler._check()
        return scaler


class DailyDischargeGRU(nn.Module):
    """[batch,past_days,1] -> next day [batch,1], residual over persistence."""
    def __init__(self, hidden_size=16):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Positive integer hidden_size required")
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != 1 or inputs.shape[1] < 1:
            raise ValueError("Expected [batch,past_days,1] discharge")
        _, hidden = self.gru(inputs)
        return inputs[:, -1, :] + self.output(hidden[-1])


def predict(model, inputs, scaler):
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(scaler.transform(inputs), dtype=torch.float32)).cpu().numpy()
    return np.maximum(scaler.inverse(output), 0)


def recursive_forecast(model, history, scaler, steps=6):
    """Free-running predictions: each prediction becomes a future input."""
    values = np.asarray(history, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all() or steps < 1:
        raise ValueError("Finite history and positive horizon required")
    outputs, window = [], values.copy()
    for _ in range(steps):
        next_value = float(predict(model, window.reshape(1, -1, 1), scaler)[0, 0])
        outputs.append(next_value)
        window = np.concatenate([window[1:], [next_value]])
    return np.asarray(outputs, dtype=np.float64)


def metrics(prediction, observed):
    prediction, observed = np.asarray(prediction, dtype=np.float64).reshape(-1), np.asarray(observed, dtype=np.float64).reshape(-1)
    if prediction.shape != observed.shape or not prediction.size or not np.isfinite(prediction).all() or not np.isfinite(observed).all():
        raise ValueError("Finite matching predictions/observations required")
    error = prediction - observed
    rmse = float(np.sqrt(np.mean(error**2)))
    variance = float(np.sum((observed-observed.mean())**2))
    span = float(np.ptp(observed))
    return {"mae": float(np.mean(np.abs(error))), "rmse": rmse, "bias": float(error.mean()),
            "nse": float(1-np.sum(error**2)/variance) if variance > 1e-12 else None, "nrmse": rmse/span if span > 1e-12 else None}


def summarize_metrics(runs):
    summary = {}
    for name in ("mae", "rmse", "bias", "nse", "nrmse"):
        values = [run["metrics"][name] for run in runs]
        summary[name] = {"mean": None, "std": None} if any(value is None for value in values) else {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else 0.}
    return summary


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("protocol_version") != PROTOCOL_VERSION or saved.get("variable") != "discharge" or saved.get("data_status") != "observed":
        raise ValueError("Not an observed daily discharge checkpoint")
    model = DailyDischargeGRU(**saved["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    return model.eval(), DischargeScaler.from_dict(saved["scaler"]), saved
