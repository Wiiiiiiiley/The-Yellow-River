"""Independent regression checks for temporal honesty and physical-unit contracts."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.multi_river import (
    FEATURE_NAMES,
    chronological_windows,
    generate_series,
    load_observed_csv,
    load_profiles,
    manning_discharge,
    reach_geometry,
)
from src.models.multi_river import FeatureScaler, RiverGraphGRU, metrics, physics_penalty, predict


ROOT = Path(__file__).resolve().parents[1]


class MultiRiverBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.profiles = load_profiles(ROOT / "configs/rivers.json")
        cls.profile = cls.profiles["yellow_river"]
        cls.series, cls.times = generate_series(cls.profile, 1000, seed=7)

    def test_all_six_scenarios_are_finite_reproducible_and_unambiguously_labeled(self):
        self.assertEqual(set(self.profiles), {"yellow_river", "mississippi", "yangtze", "amazon", "nile", "danube"})
        for river_id, profile in self.profiles.items():
            short, times = generate_series(profile, 48, seed=7)
            long, _ = generate_series(profile, 72, seed=7)
            np.testing.assert_array_equal(short, long[:48])
            self.assertEqual(short.shape, (48, 5, 6))
            self.assertEqual(times[0], "2024-01-01T00:00:00Z")
            self.assertTrue(np.isfinite(short).all(), river_id)
            self.assertTrue((short[..., [0, 1, 2, 4]] >= 0).all())
            self.assertEqual(profile["source_status"], "synthetic")
            self.assertEqual(profile["parameter_status"], "illustrative_uncalibrated")

    def test_manning_generator_and_regularizer_share_physical_units(self):
        for profile in self.profiles.values():
            series, _ = generate_series(profile, 72, seed=7)
            width, slope, roughness = reach_geometry(profile)
            calculated = manning_discharge(series[..., 0], width, slope, roughness)
            np.testing.assert_allclose(calculated, series[..., 1], rtol=3e-6)
            scaler = FeatureScaler().fit(series)
            physical = torch.tensor(scaler.transform(series[..., :3]), requires_grad=True)
            loss = physics_penalty(physical, scaler, profile)
            self.assertLess(float(loss.detach()), 1e-10)
            loss.backward()
            self.assertTrue(torch.isfinite(physical.grad).all())
            bad = physical.detach().clone()
            bad[..., 1] += 5
            bad.requires_grad_(True)
            bad_loss = physics_penalty(bad, scaler, profile)
            bad_loss.backward()
            self.assertGreater(float(bad_loss.detach()), 1e-5)
            self.assertGreater(float(bad.grad[..., 0].abs().sum()), 0)
            self.assertGreater(float(bad.grad[..., 1].abs().sum()), 0)

    def test_chronological_target_sets_are_disjoint(self):
        splits = chronological_windows(self.series, seq_len=24, horizon=6, stride=6)
        target_sets = {}
        for name, bounds in splits["boundaries"].items():
            origins = splits[name]["origins"]
            indices = {index for origin in origins for index in range(origin + 1, origin + 7)}
            self.assertTrue(all(bounds[0] <= index < bounds[1] for index in indices))
            self.assertEqual(splits[name]["x"].shape[1:], (24, 5, 6))
            self.assertEqual(splits[name]["y"].shape[1:], (6, 5, 3))
            target_sets[name] = indices
        self.assertFalse(target_sets["train"] & target_sets["val"])
        self.assertFalse(target_sets["val"] & target_sets["test"])
        self.assertLess(splits["val"]["origins"][0] - 23, splits["train_end"])
        np.testing.assert_array_equal(splits["test"]["y"][0], self.series[splits["test"]["origins"][0] + 1:splits["test"]["origins"][0] + 7, :, :3])

    def test_training_scaler_round_trip_and_gradient(self):
        train_end = chronological_windows(self.series)["train_end"]
        scaler = FeatureScaler().fit(self.series[:train_end])
        np.testing.assert_allclose(scaler.mean, self.series[:train_end].astype(np.float64).mean(axis=(0, 1)))
        normalized = scaler.transform(self.series[:train_end])
        np.testing.assert_allclose(normalized.mean(axis=(0, 1)), 0, atol=1e-5)
        np.testing.assert_allclose(scaler.inverse(normalized), self.series[:train_end], rtol=1e-5, atol=1e-5)
        restored = FeatureScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
        np.testing.assert_allclose(restored.transform(self.series), scaler.transform(self.series))
        tensor = torch.tensor(normalized[:4, :, :3], requires_grad=True)
        restored.inverse(tensor).sum().backward()
        np.testing.assert_allclose(tensor.grad[0, 0].numpy(), scaler.std[:3], rtol=1e-5)
        shifted = self.series[train_end:].copy()
        shifted[..., :3] += 100
        self.assertGreater(float(scaler.transform(shifted)[..., 0].mean()), 1)

    def test_graph_is_directed_and_models_learn_real_residuals(self):
        torch.manual_seed(4)
        scaler = FeatureScaler().fit(self.series[:700])
        splits = chronological_windows(self.series, stride=6)
        x = torch.tensor(scaler.transform(splits["train"]["x"][:8]))
        y = torch.tensor(scaler.transform(splits["train"]["y"][:8]))
        model = RiverGraphGRU(hidden_size=12)
        initial = model(x)
        self.assertEqual(tuple(initial.shape), (8, 6, 5, 3))
        torch.testing.assert_close(initial, x[:, -1, :, :3].unsqueeze(1).expand(-1, 6, -1, -1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(float((model(x) - initial).abs().sum().detach()), 0)
        downstream_modified = x.clone()
        downstream_modified[:, :, 4, :] += 100
        torch.testing.assert_close(model(x)[:, :, 0, :], model(downstream_modified)[:, :, 0, :])
        model.train()
        result = predict(model, x)
        self.assertTrue(model.training)
        self.assertEqual(result.shape, (8, 6, 5, 3))

    def test_metrics_constant_series_and_known_errors(self):
        target = np.ones((2, 6, 5, 3))
        result = metrics(target + 2, target)
        for values in result.values():
            self.assertEqual(values["mae"], 2)
            self.assertEqual(values["rmse"], 2)
            self.assertEqual(values["bias"], 2)
            self.assertIsNone(values["nse"])
            self.assertIsNone(values["nrmse"])
        json.dumps(result, allow_nan=False)
        varying = np.arange(60, dtype=float).reshape(4, 5, 3)
        perfect = metrics(varying, varying)
        self.assertTrue(all(values["nse"] == 1 for values in perfect.values()))
        with self.assertRaises(ValueError):
            metrics(target * np.nan, target)

    def _csv_rows(self):
        return [{"timestamp": timestamp, "station_id": station, **{feature: float(self.series[index, node, column]) for column, feature in enumerate(FEATURE_NAMES[:5])}} for index, timestamp in enumerate(self.times[:3]) for node, station in enumerate(self.profile["station_ids"])]

    def _load_rows(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observed.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "station_id", *FEATURE_NAMES[:5]])
                writer.writeheader()
                writer.writerows(rows)
            return load_observed_csv(path, self.profile)

    def test_strict_csv_rejects_duplicates_gaps_missing_and_invalid_values(self):
        rows = self._csv_rows()
        imported, times = self._load_rows(list(reversed(rows)))
        np.testing.assert_allclose(imported, self.series[:3])
        self.assertEqual(times, self.times[:3])
        mutations = [rows + [rows[0]], rows[:5] + rows[10:], rows[1:]]
        for bad_value in (-1, float("nan"), float("inf")):
            changed = copy.deepcopy(rows)
            changed[0]["discharge"] = bad_value
            mutations.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]["station_id"] = "official_station_not_in_profile"
        mutations.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]["timestamp"] = "2024-01-01T00:30:00Z"
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(rows=str(changed[0])):
                with self.assertRaises(ValueError):
                    self._load_rows(changed)
        # Signed Celsius temperatures are valid physical observations.
        rows[0]["temperature"] = -15
        self.assertEqual(self._load_rows(rows)[0][0, 0, 3], -15)


if __name__ == "__main__":
    unittest.main()
