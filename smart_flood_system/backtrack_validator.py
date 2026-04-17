"""
Backtrack Validator - Digital Regression Testing
==================================================

Prevents catastrophic forgetting in autonomous evolution.

Logic:
1. Monthly scheduled test on historical extreme events (2021 flood)
2. Compare current model vs historical baseline
3. Alert if performance degrades (forgetting detected)
4. Trigger rollback if regression exceeds threshold

This ensures the system gets smarter over time, not "robbing Peter to pay Paul".
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import personality model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from train_v2_personality import YellowRiverPersonalityPINN, SeasonalEncoder

logger = logging.getLogger(__name__)


@dataclass
class BacktrackResult:
    """Result of backtrack validation."""
    test_date: str
    model_generation: int
    model_path: str
    baseline_metrics: Dict  # Historical baseline
    current_metrics: Dict   # Current model metrics
    degradation_detected: bool
    degradation_ratio: float
    alert_triggered: bool
    rollback_recommended: bool
    detailed_scores: Dict


class BacktrackValidator:
    """
    Monthly regression testing on 2021 extreme flood data.
    
    Ensures evolved models don't forget how to handle historical extremes.
    """
    
    def __init__(
        self,
        baseline_data_path: str = "data/historical/2021_extreme_flood.csv",
        baseline_metrics_path: str = "models/baselines/2021_flood_baseline.json",
        degradation_threshold: float = 0.10,  # 10% degradation triggers alert
        rollback_threshold: float = 0.20,     # 20% degradation triggers rollback
        test_schedule_day: int = 1,  # 1st of each month
        archive_dir: str = "models/backtrack_tests/"
    ):
        self.baseline_data_path = Path(baseline_data_path)
        self.baseline_metrics_path = Path(baseline_metrics_path)
        self.degradation_threshold = degradation_threshold
        self.rollback_threshold = rollback_threshold
        self.test_schedule_day = test_schedule_day
        self.archive_dir = Path(archive_dir)
        
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create baseline
        self.baseline_metrics = self._load_or_create_baseline()
        
        # History of all backtrack tests
        self.test_history: List[Dict] = []
        
        logger.info(f"BacktrackValidator initialized")
        logger.info(f"  Baseline: {self.baseline_metrics_path}")
        logger.info(f"  Degradation threshold: {degradation_threshold:.0%}")
        logger.info(f"  Rollback threshold: {rollback_threshold:.0%}")
        logger.info(f"  Test schedule: Monthly day {test_schedule_day}")
    
    def _load_or_create_baseline(self) -> Dict:
        """Load existing baseline or create new one."""
        if self.baseline_metrics_path.exists():
            with open(self.baseline_metrics_path, 'r') as f:
                baseline = json.load(f)
            logger.info(f"✓ Loaded existing baseline (Gen-{baseline.get('generation', 'N/A')})")
            return baseline
        
        # No baseline exists yet - will be created on first run
        logger.info("⚠ No baseline found - will create on first test")
        return {
            'status': 'pending',
            'created': None,
            'generation': 0,
            'metrics': {}
        }
    
    def run_monthly_backtrack_test(
        self,
        current_model_path: str,
        current_generation: int
    ) -> BacktrackResult:
        """
        Run regression test on 2021 flood data.
        
        Args:
            current_model_path: Path to model to test
            current_generation: Current model generation
        
        Returns:
            BacktrackResult with degradation analysis
        """
        logger.info("=" * 80)
        logger.info(f"🔄 BACKTRACK TEST: Gen-{current_generation} vs 2021 Flood Baseline")
        logger.info("=" * 80)
        
        test_date = datetime.now().strftime('%Y-%m-%d')
        
        # 1. Load 2021 extreme flood data
        flood_data = self._load_2021_flood_data()
        
        if flood_data is None or len(flood_data) == 0:
            logger.error("✗ Failed to load 2021 flood data")
            return self._create_error_result(test_date, current_generation, current_model_path)
        
        logger.info(f"✓ Loaded 2021 flood data: {len(flood_data)} samples")
        
        # 2. Load current model
        model = self._load_model(current_model_path)
        
        # 3. Evaluate on 2021 data
        current_metrics = self._evaluate_model(model, flood_data)
        
        logger.info("Current model performance on 2021 flood:")
        for key, val in current_metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        
        # 4. Compare with baseline
        baseline_metrics = self.baseline_metrics.get('metrics', {})
        
        if not baseline_metrics or self.baseline_metrics.get('status') == 'pending':
            # First run - establish baseline
            logger.info("⚠ First backtrack test - establishing baseline")
            self._establish_baseline(current_metrics, current_generation)
            
            return BacktrackResult(
                test_date=test_date,
                model_generation=current_generation,
                model_path=current_model_path,
                baseline_metrics=current_metrics,
                current_metrics=current_metrics,
                degradation_detected=False,
                degradation_ratio=0.0,
                alert_triggered=False,
                rollback_recommended=False,
                detailed_scores={'status': 'baseline_established'}
            )
        
        # 5. Calculate degradation
        comparison = self._compare_metrics(baseline_metrics, current_metrics)
        
        degradation_ratio = max(
            comparison.get('mse_degradation', 0),
            comparison.get('mae_degradation', 0),
            comparison.get('physics_residual_degradation', 0)
        )
        
        degradation_detected = degradation_ratio > self.degradation_threshold
        alert_triggered = degradation_detected
        rollback_recommended = degradation_ratio > self.rollback_threshold
        
        # 6. Log results
        logger.info("-" * 80)
        logger.info("BACKTRACK COMPARISON:")
        logger.info(f"  Degradation ratio: {degradation_ratio:.2%}")
        
        if degradation_detected:
            logger.warning(f"⚠ DEGRADATION DETECTED: {degradation_ratio:.1%}")
            logger.warning("  Model may have forgotten 2021 flood patterns!")
            
            if rollback_recommended:
                logger.error(f"🚨 CRITICAL: Degradation exceeds {self.rollback_threshold:.0%}")
                logger.error("  ROLLBACK RECOMMENDED")
        else:
            logger.info(f"✓ No degradation detected (-{comparison.get('mse_improvement', 0):.1%} improved)")
        
        logger.info("=" * 80)
        
        # 7. Save test result
        result = BacktrackResult(
            test_date=test_date,
            model_generation=current_generation,
            model_path=current_model_path,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            degradation_detected=degradation_detected,
            degradation_ratio=degradation_ratio,
            alert_triggered=alert_triggered,
            rollback_recommended=rollback_recommended,
            detailed_scores=comparison
        )
        
        self._save_test_result(result)
        
        # 8. If rollback recommended, create flag file
        if rollback_recommended:
            self._create_rollback_flag(current_generation, degradation_ratio)
        
        return result
    
    def _load_2021_flood_data(self) -> Optional[pd.DataFrame]:
        """Load 2021 extreme flood historical data."""
        # If file doesn't exist, generate synthetic extreme flood data
        if not self.baseline_data_path.exists():
            logger.warning("2021 flood data not found - generating synthetic extreme event")
            return self._generate_synthetic_2021_flood()
        
        try:
            df = pd.read_csv(self.baseline_data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Failed to load 2021 data: {e}")
            return None
    
    def _generate_synthetic_2021_flood(self) -> pd.DataFrame:
        """Generate synthetic 2021-style extreme flood event."""
        # Create 30-day extreme flood scenario
        dates = pd.date_range('2021-07-15', periods=30*24, freq='h')  # July flood season
        
        # Extreme flood pattern
        base_flow = 5000
        peak_flow = 12000  # 50-year flood scale
        
        flow_pattern = base_flow + (peak_flow - base_flow) * np.sin(
            np.linspace(0, np.pi, len(dates))
        ) ** 2
        
        # Add noise
        flow = flow_pattern + np.random.randn(len(dates)) * 500
        
        # Sediment (extreme concentration during flood)
        sediment = 5 + 0.003 * flow + 1e-6 * flow**2 + np.random.randn(len(dates)) * 2
        
        # Water level
        water_level = 50 + 0.005 * flow + np.random.randn(len(dates)) * 1
        
        df = pd.DataFrame({
            'timestamp': dates,
            'station_id': ['HYD_001'] * len(dates),
            'water_level': water_level,
            'flow_rate': flow,
            'sediment_content': np.maximum(1, sediment),
            'rainfall': np.random.exponential(5, len(dates)),
            'temperature': 28 + np.random.randn(len(dates)) * 3,
            'event_type': '2021_extreme_flood'
        })
        
        return df
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from path."""
        model = YellowRiverPersonalityPINN(
            input_dim=10,
            hidden_dim=128,
            output_dim=2,
            num_layers=4,
            use_feature_crossing=True
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            logger.warning(f"Model not found: {model_path}, using random init")
        
        return model
    
    def _evaluate_model(
        self,
        model: nn.Module,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate model on flood data."""
        # Add seasonal features
        encoder = SeasonalEncoder(base_year=2021)
        
        seasonal = encoder.encode(pd.DatetimeIndex(data['timestamp']))
        data = data.copy()
        data['sin_year'] = seasonal[:, 0]
        data['cos_year'] = seasonal[:, 1]
        data['sin_season'] = seasonal[:, 2]
        data['cos_season'] = seasonal[:, 3]
        data['is_flood_season'] = encoder.encode_flood_season(pd.DatetimeIndex(data['timestamp']))
        
        data['hour'] = pd.DatetimeIndex(data['timestamp']).hour
        data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Prepare tensors
        feature_cols = [
            'flow_rate', 'rainfall', 'temperature',
            'sin_year', 'cos_year', 'sin_season', 'cos_season',
            'is_flood_season', 'sin_hour', 'cos_hour'
        ]
        
        X = torch.FloatTensor(data[feature_cols].values)
        y_true = torch.FloatTensor(data[['water_level', 'sediment_content']].values)
        Q = torch.FloatTensor(data['flow_rate'].values).unsqueeze(1)
        
        # Predict
        with torch.no_grad():
            y_pred = model(X, Q)
        
        # Calculate metrics
        mse = torch.mean((y_pred - y_true) ** 2).item()
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        
        # Physics residual (simplified)
        physics_residual = mse * 0.5  # Proxy
        
        # Extreme event handling score
        high_flow_mask = data['flow_rate'] > 8000
        if high_flow_mask.any():
            extreme_mse = torch.mean((y_pred[high_flow_mask.values] - y_true[high_flow_mask.values]) ** 2).item()
        else:
            extreme_mse = mse
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'physics_residual': physics_residual,
            'extreme_mse': extreme_mse,
            'samples': len(data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _compare_metrics(
        self,
        baseline: Dict,
        current: Dict
    ) -> Dict:
        """Compare baseline vs current metrics."""
        comparison = {}
        
        # MSE comparison
        if 'mse' in baseline and 'mse' in current:
            baseline_mse = baseline['mse']
            current_mse = current['mse']
            
            if baseline_mse > 0:
                degradation = (current_mse - baseline_mse) / baseline_mse
                comparison['mse_degradation'] = max(0, degradation)
                comparison['mse_improvement'] = max(0, -degradation)
            else:
                comparison['mse_degradation'] = 0
                comparison['mse_improvement'] = 0
        
        # MAE comparison
        if 'mae' in baseline and 'mae' in current:
            baseline_mae = baseline['mae']
            current_mae = current['mae']
            
            if baseline_mae > 0:
                degradation = (current_mae - baseline_mae) / baseline_mae
                comparison['mae_degradation'] = max(0, degradation)
            else:
                comparison['mae_degradation'] = 0
        
        # Physics residual comparison
        if 'physics_residual' in baseline and 'physics_residual' in current:
            baseline_phys = baseline['physics_residual']
            current_phys = current['physics_residual']
            
            if baseline_phys > 0:
                degradation = (current_phys - baseline_phys) / baseline_phys
                comparison['physics_residual_degradation'] = max(0, degradation)
            else:
                comparison['physics_residual_degradation'] = 0
        
        # Overall verdict
        max_degradation = max(
            comparison.get('mse_degradation', 0),
            comparison.get('mae_degradation', 0),
            comparison.get('physics_residual_degradation', 0)
        )
        
        comparison['overall_degradation'] = max_degradation
        comparison['passed'] = max_degradation < self.degradation_threshold
        
        return comparison
    
    def _establish_baseline(self, metrics: Dict, generation: int):
        """Establish baseline from current model."""
        self.baseline_metrics = {
            'status': 'established',
            'created': datetime.now().isoformat(),
            'generation': generation,
            'metrics': metrics,
            'description': '2021 Extreme Flood Event Baseline'
        }
        
        # Save baseline
        self.baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_metrics_path, 'w') as f:
            json.dump(self.baseline_metrics, f, indent=2)
        
        logger.info(f"✓ Baseline established (Gen-{generation})")
    
    def _save_test_result(self, result: BacktrackResult):
        """Save test result to archive."""
        result_dict = asdict(result)
        
        # Save to history
        self.test_history.append(result_dict)
        
        # Save individual result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = self.archive_dir / f"backtrack_{timestamp}_gen{result.model_generation}.json"
        
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Save latest symlink
        latest_path = self.archive_dir / "backtrack_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        
        try:
            latest_path.symlink_to(result_path.name)
        except:
            import shutil
            shutil.copy(result_path, latest_path)
    
    def _create_rollback_flag(self, generation: int, degradation: float):
        """Create flag file to trigger rollback."""
        flag_path = self.archive_dir / "ROLLBACK_REQUIRED.flag"
        
        flag_content = f"""ROLLBACK REQUIRED
================
Generation: {generation}
Degradation: {degradation:.2%}
Timestamp: {datetime.now().isoformat()}
Reason: 2021 flood performance degraded beyond {self.rollback_threshold:.0%}

ACTION REQUIRED:
1. Rollback to previous generation
2. Investigate incremental learning process
3. Check for catastrophic forgetting
"""
        
        with open(flag_path, 'w') as f:
            f.write(flag_content)
        
        logger.error(f"🚨 Rollback flag created: {flag_path}")
    
    def _create_error_result(
        self,
        test_date: str,
        generation: int,
        model_path: str
    ) -> BacktrackResult:
        """Create error result when test fails."""
        return BacktrackResult(
            test_date=test_date,
            model_generation=generation,
            model_path=model_path,
            baseline_metrics={},
            current_metrics={},
            degradation_detected=False,
            degradation_ratio=0.0,
            alert_triggered=True,
            rollback_recommended=False,
            detailed_scores={'error': 'Failed to load test data'}
        )
    
    def get_test_history(self) -> List[Dict]:
        """Get history of all backtrack tests."""
        return self.test_history
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest test result."""
        if self.test_history:
            return self.test_history[-1]
        return None
    
    def should_run_today(self) -> bool:
        """Check if backtrack test should run today."""
        today = datetime.now()
        return today.day == self.test_schedule_day


class BacktrackAlertManager:
    """
    Alert management for backtrack degradation events.
    """
    
    def __init__(self):
        self.alert_history = []
    
    def send_alert(self, result: BacktrackResult):
        """Send alert when degradation detected."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL' if result.rollback_recommended else 'WARNING',
            'generation': result.model_generation,
            'degradation_ratio': result.degradation_ratio,
            'message': self._format_alert_message(result)
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        if result.rollback_recommended:
            logger.error("=" * 80)
            logger.error("🚨 BACKTRACK ALERT: CRITICAL DEGRADATION")
            logger.error("=" * 80)
            logger.error(f"Model Gen-{result.model_generation} forgot 2021 flood!")
            logger.error(f"Degradation: {result.degradation_ratio:.1%}")
            logger.error("IMMEDIATE ACTION REQUIRED")
            logger.error("=" * 80)
        else:
            logger.warning("=" * 80)
            logger.warning("⚠ BACKTRACK ALERT: DEGRADATION DETECTED")
            logger.warning("=" * 80)
            logger.warning(f"Degradation: {result.degradation_ratio:.1%}")
            logger.warning("Monitor closely for further regression")
            logger.warning("=" * 80)
        
        return alert
    
    def _format_alert_message(self, result: BacktrackResult) -> str:
        """Format alert message."""
        if result.rollback_recommended:
            return (
                f"CRITICAL: Model Gen-{result.model_generation} degraded by "
                f"{result.degradation_ratio:.1%} on 2021 flood test. "
                f"ROLLBACK REQUIRED to prevent catastrophic forgetting."
            )
        else:
            return (
                f"WARNING: Model Gen-{result.model_generation} degraded by "
                f"{result.degradation_ratio:.1%} on 2021 flood test. "
                f"Monitor for further regression."
            )


if __name__ == "__main__":
    """Test BacktrackValidator."""
    print("Testing BacktrackValidator...")
    
    # Initialize validator
    validator = BacktrackValidator(
        degradation_threshold=0.10,
        rollback_threshold=0.20
    )
    
    # Check if should run today
    if validator.should_run_today():
        print("Today is test day!")
    else:
        print(f"Test scheduled for day {validator.test_schedule_day}")
    
    # Mock test
    result = validator.run_monthly_backtrack_test(
        current_model_path="models/test.pth",
        current_generation=5
    )
    
    print(f"\nTest Result:")
    print(f"  Generation: {result.model_generation}")
    print(f"  Degradation: {result.degradation_ratio:.2%}")
    print(f"  Alert: {result.alert_triggered}")
    print(f"  Rollback: {result.rollback_recommended}")
    
    # Test alert manager
    if result.alert_triggered:
        alert_mgr = BacktrackAlertManager()
        alert = alert_mgr.send_alert(result)
        print(f"\nAlert sent: {alert['severity']}")
    
    print("\n✓ BacktrackValidator test complete!")
