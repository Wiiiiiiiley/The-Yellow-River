import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_multi_river import export_prediction_rows
from src.models.stgnn_enhanced import PhysicsGuidedAttention, create_enhanced_model


class RiverPipelineTests(unittest.TestCase):
    def test_original_model_accepts_non_yellow_geometry_and_nodes(self):
        model = create_enhanced_model({'n_nodes': 3, 'river_width': 1600.0,
                                      'gnn_hidden': 32, 'gnn_layers': 1,
                                      'transformer_heads': 4, 'transformer_layers': 1,
                                      'sequence_length': 4, 'prediction_horizon': 2,
                                      'use_multiscale_temporal': False})
        self.assertEqual(model.post_processor.B, 1600.0)
        self.assertEqual(tuple(model.physics_attn.physical_prior.shape), (3, 3))
        model.eval()
        with torch.no_grad():
            result = model(torch.randn(2, 4, 3, 6))
        self.assertEqual(tuple(result.shape), (2, 2, 3, 6))
        self.assertTrue(torch.isfinite(result).all())

    def test_invalid_attention_prior_rejected(self):
        with self.assertRaises(ValueError):
            PhysicsGuidedAttention(32, n_nodes=3, physical_prior=torch.eye(5))

    def test_forecast_export_has_no_invented_observations(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'forecast.csv'
            export_prediction_rows(path, ['2024-01-01T0' + str(i) + ':00:00' for i in range(6)],
                                   np.ones((6, 5, 3)), 'physics_guided', 'synthetic', origins=['2023-12-31T23:00:00'])
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 90)
            self.assertTrue(all(row['observed'] == '' for row in rows))
            self.assertEqual({row['origin'] for row in rows}, {'2023-12-31T23:00:00'})


if __name__ == '__main__':
    unittest.main()
