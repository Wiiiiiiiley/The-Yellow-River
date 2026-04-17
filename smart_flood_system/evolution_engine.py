"""
Autonomous Evolution Engine for Yellow River AI
===============================================

Self-improving system that continuously evolves the flood prediction models
through error feedback loops, automatic calibration, and incremental learning.

Key Features:
1. Error Feedback Loop: Monitor prediction vs ground truth, save hard examples
2. Auto-Calibration: Fine-tune physics parameters (S0, B) when residuals rise
3. Incremental Learning: Auto-trigger fine-tuning on accumulated hard cases
4. Model Versioning: A/B testing with automatic promotion
5. Evolution Tracking: Dashboard visualization of improvement over time

Architecture:
    Monitor → Detect Anomaly → Capture Hard Example → 
    [Calibration if physics drift] + [Incremental Learning if enough data] → 
    Validate → A/B Test → Promote if Better
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class HardExample:
    """Structure for storing challenging prediction cases."""
    timestamp: str
    input_features: List[float]  # [time, location, flow, rain, temp]
    predicted: List[float]     # [water_level, sediment]
    actual: List[float]       # [water_level, sediment]
    error_rate: float          # Percentage error
    physics_residual: float    # Physics constraint violation
    severity: str              # normal/warning/critical
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HardExample':
        return cls(**data)


@dataclass
class EvolutionMetrics:
    """Track system evolution over time."""
    timestamp: str
    prediction_error: float
    physics_residual: float
    calibration_adjustment: float
    model_version: str
    hard_examples_count: int


class AnomalyDetector:
    """
    Real-time anomaly detection for prediction errors.
    
    Monitors the gap between predictions and ground truth,
    triggering hard example capture when threshold exceeded.
    """
    
    def __init__(
        self,
        error_threshold: float = 0.15,  # 15% error threshold
        physics_threshold: float = 1.0,
        window_size: int = 100
    ):
        self.error_threshold = error_threshold
        self.physics_threshold = physics_threshold
        self.window_size = window_size
        
        # Rolling statistics
        self.error_history = deque(maxlen=window_size)
        self.physics_history = deque(maxlen=window_size)
        
        # Anomaly counter
        self.anomaly_count = 0
        
    def check(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        physics_residual: float,
        input_features: np.ndarray
    ) -> Optional[HardExample]:
        """
        Check if current prediction is anomalous.
        
        Returns:
            HardExample if anomaly detected, None otherwise
        """
        # Compute error rate
        error = np.abs(prediction - ground_truth)
        relative_error = error / (np.abs(ground_truth) + 1e-6)
        max_error = np.max(relative_error)
        
        # Update history
        self.error_history.append(max_error)
        self.physics_history.append(physics_residual)
        
        # Determine severity
        is_anomaly = max_error > self.error_threshold or physics_residual > self.physics_threshold
        
        if is_anomaly:
            self.anomaly_count += 1
            
            severity = 'critical' if max_error > 0.30 or physics_residual > 2.0 else 'warning'
            
            return HardExample(
                timestamp=datetime.now().isoformat(),
                input_features=input_features.tolist(),
                predicted=prediction.tolist(),
                actual=ground_truth.tolist(),
                error_rate=float(max_error * 100),
                physics_residual=float(physics_residual),
                severity=severity
            )
        
        return None
    
    def get_trend(self) -> Dict[str, float]:
        """Get error trend statistics."""
        if len(self.error_history) < 10:
            return {'mean_error': 0, 'trend_slope': 0}
        
        errors = list(self.error_history)
        x = np.arange(len(errors))
        
        # Linear trend
        slope = np.polyfit(x, errors, 1)[0]
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'trend_slope': slope,
            'anomaly_rate': self.anomaly_count / max(1, len(errors))
        }


class AutoCalibrator:
    """
    Automatic calibration of physics parameters.
    
    When physics residuals rise, automatically fine-tune
    S0 (bed slope) and B (channel width) using gradient descent.
    """
    
    def __init__(
        self,
        initial_S0: float = 0.001,
        initial_B: float = 50.0,
        learning_rate: float = 0.001,
        calibration_window_hours: int = 48
    ):
        # Physics parameters as trainable variables
        self.S0 = nn.Parameter(torch.tensor([initial_S0], requires_grad=True))
        self.B = nn.Parameter(torch.tensor([initial_B], requires_grad=True))
        
        self.lr = learning_rate
        self.window_hours = calibration_window_hours
        
        # Historical data for calibration
        self.data_buffer = deque(maxlen=calibration_window_hours * 12)  # 5-min intervals
        
        # Constraints
        self.S0_bounds = (0.0001, 0.01)
        self.B_bounds = (20.0, 100.0)
        
    def add_observation(
        self,
        water_level: float,
        flow_rate: float,
        physics_residual: float,
        timestamp: datetime
    ):
        """Add observation to calibration buffer."""
        self.data_buffer.append({
            'h': water_level,
            'Q': flow_rate,
            'residual': physics_residual,
            'timestamp': timestamp
        })
    
    def should_calibrate(self) -> bool:
        """Check if calibration is needed based on recent residuals."""
        if len(self.data_buffer) < 100:
            return False
        
        recent = list(self.data_buffer)[-100:]
        recent_residuals = [d['residual'] for d in recent]
        
        # Calibrate if mean residual is high and increasing
        mean_residual = np.mean(recent_residuals)
        trend = np.polyfit(range(len(recent_residuals)), recent_residuals, 1)[0]
        
        return mean_residual > 0.5 and trend > 0.01
    
    def calibrate(self) -> Dict[str, float]:
        """
        Run calibration optimization.
        
        Returns:
            Dictionary with new parameter values
        """
        if len(self.data_buffer) < 50:
            return {'S0': self.S0.item(), 'B': self.B.item(), 'improvement': 0}
        
        # Prepare data
        data = list(self.data_buffer)
        h = torch.tensor([d['h'] for d in data])
        Q = torch.tensor([d['Q'] for d in data])
        target_residuals = torch.zeros(len(data))  # Target: minimize to 0
        
        # Optimizer
        optimizer = optim.Adam([self.S0, self.B], lr=self.lr)
        
        # Training loop
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Compute physics residuals with current parameters
            # Simplified momentum equation residual
            u = Q / (self.B * h + 1e-6)
            friction = 9.8 * (0.03 ** 2) * u * torch.abs(u) / ((h + 1e-6) ** (4/3))
            bed_slope_term = 9.8 * self.S0
            
            # Residual = friction - bed_slope (simplified)
            predicted_residuals = torch.abs(friction - bed_slope_term)
            
            # Loss: minimize residuals
            loss = torch.mean((predicted_residuals - target_residuals) ** 2)
            
            # Constraint penalty
            S0_penalty = torch.relu(self.S0 - self.S0_bounds[1]) + torch.relu(self.S0_bounds[0] - self.S0)
            B_penalty = torch.relu(self.B - self.B_bounds[1]) + torch.relu(self.B_bounds[0] - self.B)
            total_loss = loss + 0.1 * (S0_penalty + B_penalty)
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Calibration Epoch {epoch}: Loss={loss.item():.6f}, "
                          f"S0={self.S0.item():.6f}, B={self.B.item():.2f}")
        
        # Clamp to bounds
        with torch.no_grad():
            self.S0.clamp_(self.S0_bounds[0], self.S0_bounds[1])
            self.B.clamp_(self.B_bounds[0], self.B_bounds[1])
        
        improvement = losses[0] - losses[-1] if len(losses) > 1 else 0
        
        return {
            'S0': self.S0.item(),
            'B': self.B.item(),
            'improvement': improvement,
            'final_loss': losses[-1] if losses else 0
        }
    
    def save_params(self, path: str):
        """Save calibrated parameters."""
        params = {
            'S0': self.S0.item(),
            'B': self.B.item(),
            'calibrated_at': datetime.now().isoformat(),
            'data_points': len(self.data_buffer)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Saved calibrated physics params to {path}")


class IncrementalLearner:
    """
    Incremental learning pipeline for continuous model improvement.
    
    Accumulates hard examples and periodically triggers fine-tuning
    on Transformer attention layers only.
    """
    
    def __init__(
        self,
        model: nn.Module,
        min_hard_examples: int = 50,
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
        batch_size: int = 16,
        max_epochs: int = 10,
        patience: int = 3
    ):
        self.model = model
        self.min_examples = min_hard_examples
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        
        # Hard examples buffer
        self.hard_examples: List[HardExample] = []
        
        # Model versioning
        self.version = 1
        self.best_val_loss = float('inf')
        
        # Evolution tracking
        self.evolution_history: List[EvolutionMetrics] = []
        
    def add_hard_example(self, example: HardExample):
        """Add a hard example to the buffer."""
        self.hard_examples.append(example)
        logger.info(f"Added hard example (error={example.error_rate:.1f}%, "
                   f"buffer size={len(self.hard_examples)})")
    
    def should_retrain(self) -> bool:
        """Check if enough hard examples accumulated."""
        return len(self.hard_examples) >= self.min_examples
    
    def auto_retrain(
        self,
        validation_data: Optional[Tuple] = None
    ) -> Dict[str, any]:
        """
        Trigger incremental fine-tuning.
        
        Only updates Transformer attention layer weights
        with low learning rate to preserve general knowledge.
        
        Returns:
            Training results and metrics
        """
        if len(self.hard_examples) < self.min_examples:
            return {'status': 'insufficient_data', 'n_examples': len(self.hard_examples)}
        
        logger.info("=" * 80)
        logger.info(f"Starting Incremental Learning (v{self.version})")
        logger.info(f"Hard examples: {len(self.hard_examples)}")
        logger.info("=" * 80)
        
        # Prepare data
        X = torch.tensor([ex.input_features for ex in self.hard_examples], dtype=torch.float32)
        y = torch.tensor([ex.actual for ex in self.hard_examples], dtype=torch.float32)
        
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        
        # Freeze all layers except attention
        for name, param in self.model.named_parameters():
            if 'attention' in name or 'attn' in name:
                param.requires_grad = True
                logger.info(f"Unfrozen for training: {name}")
            else:
                param.requires_grad = False
        
        # Optimizer - only for unfrozen parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    pred = self.model(X_batch)
                    val_loss += criterion(pred, y_batch).item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.max_epochs}: "
                       f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best weights
        self.model.load_state_dict(best_state)
        
        # Record evolution
        self.evolution_history.append(EvolutionMetrics(
            timestamp=datetime.now().isoformat(),
            prediction_error=best_val_loss,
            physics_residual=0,  # Will be filled by caller
            calibration_adjustment=0,
            model_version=f"v{self.version}",
            hard_examples_count=len(self.hard_examples)
        ))
        
        # Clear hard examples after training
        n_used = len(self.hard_examples)
        self.hard_examples.clear()
        
        # Version increment
        old_version = self.version
        self.version += 1
        
        return {
            'status': 'success',
            'version': old_version,
            'next_version': self.version,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'n_examples_used': n_used,
            'epochs_trained': len(train_losses)
        }
    
    def save_checkpoint(self, path: str) -> str:
        """Save model checkpoint with version."""
        versioned_path = path.replace('.pt', f'_v{self.version}.pt')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'version': self.version,
            'evolution_history': [asdict(m) for m in self.evolution_history],
            'saved_at': datetime.now().isoformat()
        }, versioned_path)
        
        logger.info(f"Saved checkpoint: {versioned_path}")
        return versioned_path


class ModelVersionManager:
    """
    A/B Testing and automatic model promotion.
    
    Manages model versions and automatically promotes new versions
    if they perform better on validation set.
    """
    
    def __init__(self, models_dir: str = "models/evolution/"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.versions: List[Dict] = []
        self.active_version: Optional[str] = None
        
    def register_version(
        self,
        version: str,
        checkpoint_path: str,
        validation_metrics: Dict
    ):
        """Register a new model version."""
        version_info = {
            'version': version,
            'path': checkpoint_path,
            'metrics': validation_metrics,
            'registered_at': datetime.now().isoformat(),
            'status': 'candidate'  # candidate / active / deprecated
        }
        
        self.versions.append(version_info)
        
        # Compare with active version
        if self.active_version is None:
            self._promote(version_info)
        else:
            active = next(v for v in self.versions if v['version'] == self.active_version)
            
            if self._is_better(version_info, active):
                logger.info(f"New version {version} outperforms active {self.active_version}")
                self._promote(version_info)
            else:
                logger.info(f"Version {version} not better than active, keeping as candidate")
    
    def _is_better(self, candidate: Dict, current: Dict) -> bool:
        """Compare model versions."""
        cand_loss = candidate['metrics'].get('val_loss', float('inf'))
        curr_loss = current['metrics'].get('val_loss', float('inf'))
        
        # New version wins if 5% better
        return cand_loss < curr_loss * 0.95
    
    def _promote(self, version_info: Dict):
        """Promote version to active."""
        # Demote current active
        if self.active_version:
            old = next(v for v in self.versions if v['version'] == self.active_version)
            old['status'] = 'deprecated'
        
        # Promote new
        version_info['status'] = 'active'
        self.active_version = version_info['version']
        
        logger.info(f"🎉 Promoted {version_info['version']} to ACTIVE")
    
    def get_active_model_path(self) -> Optional[str]:
        """Get path to currently active model."""
        if self.active_version is None:
            return None
        
        active = next(v for v in self.versions if v['version'] == self.active_version)
        return active['path']
    
    def get_version_history(self) -> pd.DataFrame:
        """Get version history as DataFrame."""
        return pd.DataFrame(self.versions)


class EvolutionEngine:
    """
    Main evolution engine coordinating all components.
    
    Runs as background thread, continuously monitoring and improving
    the flood prediction system.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config_path: str = "configs/evolution_config.yaml",
        models_dir: str = "models/evolution/"
    ):
        self.model = model
        self.models_dir = models_dir
        
        # Load or create config
        self.config = self._load_config(config_path)
        
        # Components
        self.anomaly_detector = AnomalyDetector(
            error_threshold=self.config.get('error_threshold', 0.15),
            physics_threshold=self.config.get('physics_threshold', 1.0)
        )
        
        self.calibrator = AutoCalibrator(
            initial_S0=self.config.get('S0', 0.001),
            initial_B=self.config.get('B', 50.0)
        )
        
        self.learner = IncrementalLearner(
            model=model,
            min_hard_examples=self.config.get('min_hard_examples', 50),
            learning_rate=self.config.get('retrain_lr', 1e-5)
        )
        
        self.version_manager = ModelVersionManager(models_dir)
        
        # Hard examples storage
        self.hard_examples_path = "data/evolution/hard_examples.json"
        os.makedirs(os.path.dirname(self.hard_examples_path), exist_ok=True)
        
        # Evolution tracking
        self.evolution_metrics: List[Dict] = []
        
        # Threading
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        logger.info("Evolution Engine initialized")
    
    def _load_config(self, path: str) -> Dict:
        """Load or create default config."""
        if os.path.exists(path):
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        return {
            'error_threshold': 0.15,
            'physics_threshold': 1.0,
            'calibration_interval_hours': 24,
            'min_hard_examples': 50,
            'retrain_lr': 1e-5,
            'S0': 0.001,
            'B': 50.0
        }
    
    def process_prediction(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        physics_residual: float,
        input_features: np.ndarray
    ) -> Dict:
        """
        Process a single prediction through evolution pipeline.
        
        This is the main entry point called by the inference system.
        """
        results = {'action_taken': None}
        
        # Step 1: Anomaly detection
        hard_example = self.anomaly_detector.check(
            prediction, ground_truth, physics_residual, input_features
        )
        
        if hard_example:
            # Save hard example
            self._save_hard_example(hard_example)
            self.learner.add_hard_example(hard_example)
            results['hard_example_saved'] = True
            
            # Add to calibrator
            self.calibrator.add_observation(
                ground_truth[0],  # water level
                input_features[2],  # flow rate
                physics_residual,
                datetime.now()
            )
        
        # Step 2: Check if calibration needed
        if self.calibrator.should_calibrate():
            logger.info("Triggering automatic calibration...")
            calib_result = self.calibrator.calibrate()
            self.calibrator.save_params("config/physics_params.json")
            results['calibration'] = calib_result
            
            # Update config
            self.config['S0'] = calib_result['S0']
            self.config['B'] = calib_result['B']
        
        # Step 3: Check if retraining needed
        if self.learner.should_retrain():
            logger.info("Triggering incremental learning...")
            retrain_result = self.learner.auto_retrain()
            results['retraining'] = retrain_result
            
            if retrain_result['status'] == 'success':
                # Save new version
                checkpoint_path = self.learner.save_checkpoint(
                    f"{self.models_dir}/evolved_model.pt"
                )
                
                # Register with version manager
                self.version_manager.register_version(
                    version=f"v{retrain_result['next_version']}",
                    checkpoint_path=checkpoint_path,
                    validation_metrics={'val_loss': retrain_result['best_val_loss']}
                )
                
                results['new_version'] = retrain_result['next_version']
        
        # Record metrics
        trend = self.anomaly_detector.get_trend()
        self.evolution_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'mean_error': trend['mean_error'],
            'anomaly_rate': trend['anomaly_rate'],
            'hard_examples_count': len(self.learner.hard_examples),
            'current_version': self.version_manager.active_version
        })
        
        return results
    
    def _save_hard_example(self, example: HardExample):
        """Append hard example to JSON file."""
        examples = []
        if os.path.exists(self.hard_examples_path):
            with open(self.hard_examples_path, 'r') as f:
                examples = json.load(f)
        
        examples.append(example.to_dict())
        
        with open(self.hard_examples_path, 'w') as f:
            json.dump(examples, f, indent=2)
    
    def generate_evolution_report(self, save_path: str = "results/evolution_trajectory.png"):
        """Generate evolution trajectory visualization."""
        if len(self.evolution_metrics) < 2:
            logger.warning("Not enough data for evolution report")
            return
        
        df = pd.DataFrame(self.evolution_metrics)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Yellow River AI Evolution Trajectory', fontsize=14, fontweight='bold')
        
        # 1. Error trend
        ax = axes[0, 0]
        ax.plot(df['timestamp'], df['mean_error'], 'b-', linewidth=2)
        ax.axhline(y=0.15, color='r', linestyle='--', label='15% Threshold')
        ax.set_ylabel('Mean Prediction Error')
        ax.set_title('1. Prediction Error Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Anomaly rate
        ax = axes[0, 1]
        ax.plot(df['timestamp'], df['anomaly_rate'], 'orange', linewidth=2)
        ax.set_ylabel('Anomaly Rate')
        ax.set_title('2. Anomaly Detection Rate')
        ax.grid(True, alpha=0.3)
        
        # 3. Hard examples accumulation
        ax = axes[1, 0]
        ax.fill_between(df['timestamp'], 0, df['hard_examples_count'], alpha=0.5)
        ax.set_ylabel('Hard Examples Count')
        ax.set_title('3. Learning Buffer Size')
        ax.grid(True, alpha=0.3)
        
        # 4. Version progression
        ax = axes[1, 1]
        version_history = self.version_manager.get_version_history()
        if len(version_history) > 0:
            ax.scatter(version_history['registered_at'], 
                      [1 if s == 'active' else 0.5 for s in version_history['status']],
                      c=['green' if s == 'active' else 'gray' for s in version_history['status']],
                      s=100)
        ax.set_ylabel('Version Status')
        ax.set_title('4. Model Version Evolution')
        ax.set_yticks([0.5, 1])
        ax.set_yticklabels(['Candidate', 'Active'])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Evolution report saved: {save_path}")
    
    def start(self):
        """Start evolution engine in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.thread.start()
        logger.info("Evolution Engine started in background")
    
    def stop(self):
        """Stop evolution engine."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Evolution Engine stopped")
    
    def _evolution_loop(self):
        """Background evolution loop."""
        while self.running:
            try:
                # Periodic tasks
                if len(self.evolution_metrics) % 100 == 0:
                    self.generate_evolution_report()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                time.sleep(5)


if __name__ == "__main__":
    """Test evolution engine."""
    print("Testing Evolution Engine...")
    
    # Create dummy model
    from smart_flood_system.models.graph_transformer.stgnn import SpatioTemporalGNN
    
    model = SpatioTemporalGNN(
        in_channels=5,
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
        transformer_heads=4,
        transformer_layers=1
    )
    
    # Initialize engine
    engine = EvolutionEngine(model)
    
    # Simulate predictions
    for i in range(10):
        pred = np.array([50.0 + np.random.randn() * 5, 10.0 + np.random.randn()])
        actual = np.array([52.0, 10.5])
        physics_res = np.random.rand() * 0.5
        features = np.array([0.5, 0.3, 2000.0, 1.0, 20.0])
        
        results = engine.process_prediction(pred, actual, physics_res, features)
        print(f"Step {i+1}: {results}")
    
    # Generate report
    engine.generate_evolution_report("test_evolution.png")
    print("✓ Evolution engine test complete!")
