import unittest

import torch

from scripts.train_both_models import denormalize_data, normalize_data
from src.models.baseline_models import BaselineLSTM
from src.models.physics_constraints_enhanced import EnhancedPhysicsConstraintLoss
from src.data.data_adapter import SimulatedSource


class TrainingProtocolTests(unittest.TestCase):
    def test_validation_reuses_training_statistics(self):
        train = torch.arange(240, dtype=torch.float32).reshape(2, 4, 5, 6)
        targets = train[:, :2]
        train_norm, _, mean, std = normalize_data(train, targets)

        validation = train + 1000
        validation_norm, _, reused_mean, reused_std = normalize_data(
            validation, validation[:, :2], mean, std
        )

        self.assertTrue(torch.allclose(mean, reused_mean))
        self.assertTrue(torch.allclose(std, reused_std))
        self.assertFalse(torch.allclose(validation_norm.mean(), torch.tensor(0.0)))
        self.assertTrue(
            torch.allclose(denormalize_data(train_norm, mean, std), train, atol=1e-5)
        )

    def test_baseline_predicts_every_node(self):
        model = BaselineLSTM(hidden_size=32)
        predictions = model(torch.randn(2, 24, 5, 6))
        self.assertEqual(predictions.shape, (2, 6, 5, 6))

    def test_ntk_weights_do_not_require_existing_parameter_gradients(self):
        model = torch.nn.Linear(2, 1)
        inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
        predictions = model(inputs)
        data_loss = (predictions - 1.0).pow(2).mean()
        physics_loss = (predictions * 5.0).pow(2).mean()
        loss = EnhancedPhysicsConstraintLoss(
            use_ntk_adaptive=True,
            use_curriculum=False,
            use_sediment=False,
        )

        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        data_weight, physics_weight = loss.compute_ntk_weights(
            model, data_loss, physics_loss
        )

        self.assertAlmostEqual(data_weight + physics_weight, 1.0, places=7)
        self.assertGreaterEqual(physics_weight, 0.05)
        self.assertLessEqual(physics_weight, 0.95)

    def test_simulated_source_supports_single_reading(self):
        readings = SimulatedSource().get_readings('HYD_001', hours=1)
        self.assertEqual(len(readings), 1)


if __name__ == '__main__':
    unittest.main()
