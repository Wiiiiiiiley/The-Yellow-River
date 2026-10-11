import unittest
from datetime import date, timedelta

import numpy as np
import torch

from src.data.observed_rivers import daily_windows, validate_records
from src.models.observed_rivers import DailyDischargeGRU, DischargeScaler, metrics, recursive_forecast, summarize_metrics


class ObservedRiverTests(unittest.TestCase):
    def records(self):
        return [{"date": (date(2015,1,1)+timedelta(days=i)).isoformat(), "discharge": float(i+1)} for i in range(200)]

    def test_train_stats_and_disjoint_causal_targets(self):
        records = self.records()
        parts, raw_train = daily_windows(records, {"start": records[0]["date"], "end": records[-1]["date"]})
        self.assertEqual(len(raw_train), 140)
        self.assertEqual(DischargeScaler().fit(raw_train).mean, 70.5)
        self.assertEqual(len(parts["train"]["dates"]), 110)
        self.assertLess(parts["train"]["dates"][-1], parts["validation"]["dates"][0])
        self.assertLess(parts["validation"]["dates"][-1], parts["test"]["dates"][0])
        self.assertEqual(parts["validation"]["x"][0,-1,0], 140)
        self.assertEqual(parts["validation"]["y"][0,0], 141)

    def test_no_gap_crossing_or_interpolation(self):
        records = self.records()
        del records[100]
        parts, _ = daily_windows(records, {"start": "2015-01-01", "end": records[-1]["date"]})
        self.assertEqual(parts["train"]["skipped_windows"], 31)
        self.assertEqual(len(parts["train"]["dates"]), 79)

    def test_duplicates_and_invalid_discharge_rejected(self):
        records = self.records()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_records(records+[records[0]])
        records[0]["discharge"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_records(records)

    def test_residual_initialization_and_recursive_persistence(self):
        model = DailyDischargeGRU()
        inputs = torch.randn(4,30,1)
        torch.testing.assert_close(model(inputs), inputs[:,-1,:])
        scaler = DischargeScaler().fit([1,2,3])
        np.testing.assert_allclose(recursive_forecast(model, [1,2,3], scaler, 6), np.repeat(3.,6), atol=1e-6)

    def test_scaler_roundtrip_and_constant_metrics(self):
        scaler = DischargeScaler().fit([10,20,30])
        np.testing.assert_allclose(scaler.inverse(scaler.transform([10,20,30])), [10,20,30], atol=1e-6)
        self.assertIsNone(metrics([1,2,3], [1,1,1])["nse"])
        self.assertIsNone(metrics([1,2,3], [1,1,1])["nrmse"])

    def test_seed_std_is_sample_std(self):
        runs = [{"metrics": {name:value for name in ("mae","rmse","bias","nse","nrmse")}} for value in [1.,2.,3.]]
        self.assertEqual(summarize_metrics(runs)["rmse"], {"mean":2., "std":1.})


if __name__ == "__main__":
    unittest.main()
