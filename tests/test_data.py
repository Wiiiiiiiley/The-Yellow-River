"""Tests for data module."""

import unittest
import pandas as pd
import numpy as np
from src.features.builder import create_temporal_features, create_lag_features


class TestFeatureBuilder(unittest.TestCase):
    def test_temporal_features(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-06-01', '2023-12-01'],
            'value': [1, 2, 3]
        })
        result = create_temporal_features(df)
        self.assertIn('year', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('season', result.columns)
    
    def test_lag_features(self):
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        result = create_lag_features(df, ['value'], [1, 2])
        self.assertIn('value_lag_1', result.columns)
        self.assertIn('value_lag_2', result.columns)


if __name__ == '__main__':
    unittest.main()
