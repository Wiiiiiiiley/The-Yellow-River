"""Multi-Model Cross-Verification System for Yellow River PINN.

Implementing ensemble prediction with dynamic confidence indexing
by analyzing divergence between Physics-Informed (PINN) and Pure Data-Driven models.

Usage:
    python scripts/multi_model_verifier.py --test-data data/processed/test.csv
    python scripts/multi_model_verifier.py --generate-demo
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.pinn_model import YellowRiverPINN


class MultiModelVerifier:
    """Multi-Model Cross-Verification System.
    
    Recognizing that single-model predictions can be inherently biased,
    this system analyzes the divergence between Physics-Informed (PINN) 
    and Pure Data-Driven models to create a Dynamic Confidence Index.
    
    This allows the system to not only predict the river's state but also 
    quantify its own 'uncertainty'—a critical feature for real-world 
    disaster mitigation.
    
    Attributes:
        model_pinn: Physics-Informed Neural Network (λ=0.1)
        model_baseline: Pure Data-Driven model (λ=0.0)
        norm_params: Data normalization parameters
        divergence_threshold: Threshold for high uncertainty (default 0.10 = 10%)
    """
    
    def __init__(
        self,
        pinn_path: str = "models/model_pinn.pth",
        baseline_path: str = "models/model_baseline.pth",
        norm_params_path: str = None,
        divergence_threshold: float = 0.10,
        device: str = 'cpu'
    ):
        """Initialize verifier with two models.
        
        Args:
            pinn_path: Path to PINN model checkpoint
            baseline_path: Path to Baseline model checkpoint
            norm_params_path: Path to normalization parameters JSON
            divergence_threshold: Threshold for marking high uncertainty (0.10 = 10%)
            device: Device to run models on
        """
        self.device = torch.device(device)
        self.divergence_threshold = divergence_threshold
        
        # Load normalization parameters
        if norm_params_path is None:
            # Auto-detect from training directories
            for potential_path in [
                'models/pinn/20260417_101109/norm_params.json',
                'models/baseline/20260417_101109/norm_params.json'
            ]:
                if os.path.exists(potential_path):
                    norm_params_path = potential_path
                    break
        
        if norm_params_path and os.path.exists(norm_params_path):
            with open(norm_params_path, 'r') as f:
                self.norm_params = json.load(f)
            self.x_mean = np.array(self.norm_params['x_mean'])
            self.x_std = np.array(self.norm_params['x_std'])
            self.y_mean = np.array(self.norm_params['y_mean'])
            self.y_std = np.array(self.norm_params['y_std'])
        else:
            # Default normalization (approximate)
            self.x_mean = np.zeros(5)
            self.x_std = np.ones(5)
            self.y_mean = np.array([50.0, 10.0])  # water_level, sediment
            self.y_std = np.array([20.0, 5.0])
            print("[Warning] Using default normalization parameters")
        
        # Initialize models
        self.model_pinn = YellowRiverPINN(input_dim=5, output_dim=2, activation='swish')
        self.model_baseline = YellowRiverPINN(input_dim=5, output_dim=2, activation='swish')
        
        # Load model weights
        self._load_model(self.model_pinn, pinn_path, "PINN")
        self._load_model(self.model_baseline, baseline_path, "Baseline")
        
        self.model_pinn.to(self.device).eval()
        self.model_baseline.to(self.device).eval()
        
        print(f"[Verifier] Initialized with divergence threshold: {divergence_threshold:.1%}")
    
    def _load_model(self, model: torch.nn.Module, path: str, name: str):
        """Load model checkpoint."""
        try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[Verifier] ✓ Loaded {name} from {path}")
            else:
                print(f"[Warning] {name} model not found at {path}, using random weights")
        except Exception as e:
            print(f"[Warning] Failed to load {name}: {e}")
    
    def normalize_input(self, X: np.ndarray) -> torch.Tensor:
        """Normalize input features."""
        X_norm = (X - self.x_mean) / self.x_std
        return torch.FloatTensor(X_norm).to(self.device)
    
    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return y * self.y_std + self.y_mean
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from both models.
        
        Args:
            X: Input features (N, 5) [time, location, flow, rain, temp]
        
        Returns:
            Dictionary with predictions from both models
        """
        X_tensor = self.normalize_input(X)
        
        with torch.no_grad():
            pred_pinn = self.model_pinn(X_tensor).cpu().numpy()
            pred_baseline = self.model_baseline(X_tensor).cpu().numpy()
        
        # Denormalize
        pred_pinn = self.denormalize_output(pred_pinn)
        pred_baseline = self.denormalize_output(pred_baseline)
        
        return {
            'pinn': pred_pinn,
            'baseline': pred_baseline
        }
    
    def compute_divergence(
        self,
        pred_pinn: np.ndarray,
        pred_baseline: np.ndarray,
        method: str = 'relative'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute divergence between two model predictions.
        
        Args:
            pred_pinn: PINN predictions (N, 2)
            pred_baseline: Baseline predictions (N, 2)
            method: 'relative' (percentage) or 'absolute'
        
        Returns:
            divergence: (N, 2) array of divergence values
            is_high_uncertainty: (N, 2) boolean array
        """
        if method == 'relative':
            # Relative divergence: |pred1 - pred2| / mean(|pred1|, |pred2|)
            mean_pred = (np.abs(pred_pinn) + np.abs(pred_baseline)) / 2
            mean_pred = np.where(mean_pred < 1e-6, 1e-6, mean_pred)  # Avoid div by zero
            divergence = np.abs(pred_pinn - pred_baseline) / mean_pred
        else:
            # Absolute divergence
            divergence = np.abs(pred_pinn - pred_baseline)
        
        # Mark high uncertainty points
        is_high_uncertainty = divergence > self.divergence_threshold
        
        return divergence, is_high_uncertainty
    
    def ensemble_predict(
        self,
        X: np.ndarray,
        weights: List[float] = [0.5, 0.5]
    ) -> Dict[str, np.ndarray]:
        """Compute ensemble prediction with confidence band.
        
        Args:
            X: Input features
            weights: Ensemble weights [w_baseline, w_pinn]
        
        Returns:
            Dictionary with ensemble mean, std, and confidence band
        """
        preds = self.predict(X)
        pred_baseline = preds['baseline']
        pred_pinn = preds['pinn']
        
        # Weighted ensemble mean
        ensemble_mean = weights[0] * pred_baseline + weights[1] * pred_pinn
        
        # Standard deviation as uncertainty measure
        ensemble_std = np.sqrt(
            weights[0] * (pred_baseline - ensemble_mean)**2 +
            weights[1] * (pred_pinn - ensemble_mean)**2
        )
        
        # 95% confidence band (approximately 2*std)
        confidence_band = 2 * ensemble_std
        
        # Divergence analysis
        divergence, is_high_uncertainty = self.compute_divergence(pred_pinn, pred_baseline)
        
        return {
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'confidence_band': confidence_band,
            'baseline': pred_baseline,
            'pinn': pred_pinn,
            'divergence': divergence,
            'is_high_uncertainty': is_high_uncertainty,
            'high_uncertainty_ratio': np.mean(is_high_uncertainty, axis=0)
        }
    
    def verify_and_visualize(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray = None,
        save_path: str = 'results/multi_model_verification.png'
    ):
        """Run verification and generate comprehensive visualization.
        
        Creates a 6-panel dashboard showing:
        1. Water Level: Both models + Ensemble + Confidence Band
        2. Sediment: Both models + Ensemble + Confidence Band
        3. Divergence Heatmap
        4. High Uncertainty Points Map
        5. Model Agreement Distribution
        6. Confidence vs Error (if ground truth available)
        """
        results = self.ensemble_predict(X_test)
        
        # Create figure with 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            'Multi-Model Cross-Verification System\n'
            'Physics-Informed vs Data-Driven Ensemble with Dynamic Confidence Index',
            fontsize=14, fontweight='bold'
        )
        
        x_axis = X_test[:, 1]  # Location as x-axis
        
        # Panel 1: Water Level Predictions
        ax = axes[0, 0]
        ax.fill_between(
            x_axis,
            results['ensemble_mean'][:, 0] - results['confidence_band'][:, 0],
            results['ensemble_mean'][:, 0] + results['confidence_band'][:, 0],
            alpha=0.3, color='green', label='95% Confidence Band'
        )
        ax.plot(x_axis, results['baseline'][:, 0], 'b--', label='Baseline (λ=0.0)', linewidth=2)
        ax.plot(x_axis, results['pinn'][:, 0], 'r-', label='PINN (λ=0.1)', linewidth=2)
        ax.plot(x_axis, results['ensemble_mean'][:, 0], 'g:', label='Ensemble Mean', linewidth=2)
        if y_true is not None:
            ax.scatter(x_axis, y_true[:, 0], c='black', s=20, alpha=0.5, label='Observations', zorder=5)
        ax.set_xlabel('Location (normalized)')
        ax.set_ylabel('Water Level (m)')
        ax.set_title('1. Water Level: Ensemble Prediction')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Sediment Predictions
        ax = axes[0, 1]
        ax.fill_between(
            x_axis,
            results['ensemble_mean'][:, 1] - results['confidence_band'][:, 1],
            results['ensemble_mean'][:, 1] + results['confidence_band'][:, 1],
            alpha=0.3, color='orange', label='95% Confidence Band'
        )
        ax.plot(x_axis, results['baseline'][:, 1], 'b--', label='Baseline (λ=0.0)', linewidth=2)
        ax.plot(x_axis, results['pinn'][:, 1], 'r-', label='PINN (λ=0.1)', linewidth=2)
        ax.plot(x_axis, results['ensemble_mean'][:, 1], 'orange', linestyle=':', label='Ensemble Mean', linewidth=2)
        if y_true is not None:
            ax.scatter(x_axis, y_true[:, 1], c='black', s=20, alpha=0.5, label='Observations', zorder=5)
        ax.set_xlabel('Location (normalized)')
        ax.set_ylabel('Sediment Content (kg/m³)')
        ax.set_title('2. Sediment: Ensemble Prediction')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Divergence Analysis
        ax = axes[0, 2]
        ax.plot(x_axis, results['divergence'][:, 0], 'b-', label='Water Level', linewidth=2)
        ax.plot(x_axis, results['divergence'][:, 1], 'orange', label='Sediment', linewidth=2)
        ax.axhline(y=self.divergence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.divergence_threshold:.1%})')
        ax.set_xlabel('Location (normalized)')
        ax.set_ylabel('Relative Divergence')
        ax.set_title('3. Model Divergence Analysis')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: High Uncertainty Points
        ax = axes[1, 0]
        high_unc_water = results['is_high_uncertainty'][:, 0]
        high_unc_sediment = results['is_high_uncertainty'][:, 1]
        
        ax.scatter(x_axis[high_unc_water], results['ensemble_mean'][high_unc_water, 0],
                   c='red', s=50, marker='x', label='High Uncertainty (Water)', zorder=5)
        ax.scatter(x_axis[~high_unc_water], results['ensemble_mean'][~high_unc_water, 0],
                   c='green', s=20, alpha=0.5, label='Normal Confidence', zorder=4)
        ax.set_xlabel('Location (normalized)')
        ax.set_ylabel('Water Level (m)')
        ax.set_title(f'4. Uncertainty Map (Water)\n'
                     f'High Uncertainty: {results["high_uncertainty_ratio"][0]:.1%}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Divergence Distribution
        ax = axes[1, 1]
        ax.hist(results['divergence'][:, 0], bins=30, alpha=0.5, label='Water Level', color='blue')
        ax.hist(results['divergence'][:, 1], bins=30, alpha=0.5, label='Sediment', color='orange')
        ax.axvline(x=self.divergence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.divergence_threshold:.1%})')
        ax.set_xlabel('Relative Divergence')
        ax.set_ylabel('Frequency')
        ax.set_title('5. Divergence Distribution')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Confidence Metrics Summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Multi-Model Verification Summary
        =================================
        
        Divergence Threshold: {self.divergence_threshold:.1%}
        
        Water Level:
          Mean Divergence: {np.mean(results['divergence'][:, 0]):.2%}
          Max Divergence: {np.max(results['divergence'][:, 0]):.2%}
          High Uncertainty Ratio: {results['high_uncertainty_ratio'][0]:.1%}
        
        Sediment Content:
          Mean Divergence: {np.mean(results['divergence'][:, 1]):.2%}
          Max Divergence: {np.max(results['divergence'][:, 1]):.2%}
          High Uncertainty Ratio: {results['high_uncertainty_ratio'][1]:.1%}
        
        Ensemble Weights: [0.5, 0.5]
        
        Reliability Assessment:
          {'⚠️ CAUTION' if np.mean(results['is_high_uncertainty']) > 0.2 else '✓ RELIABLE'}
          ({np.mean(results['is_high_uncertainty']):.1%} points exceed threshold)
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('6. Verification Metrics')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[Verifier] Visualization saved to: {save_path}")
        
        return results, fig


def generate_demo_data(n_samples: int = 200, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """Generate demo test data with extreme conditions."""
    np.random.seed(seed)
    
    # Time and location
    time = np.linspace(0, 1, n_samples)
    location = np.linspace(0, 1, n_samples)
    
    # Extreme flow conditions (flood scenario)
    flow_rate = 4000 + 800 * np.sin(time * 4 * np.pi) + np.random.normal(0, 200, n_samples)
    flow_rate = np.clip(flow_rate, 500, 5000)
    
    # Heavy rainfall
    rainfall = np.maximum(0, 3.0 * np.sin(time * 2 * np.pi) + np.random.exponential(1, n_samples))
    
    # Temperature
    temperature = 25 + 10 * np.sin(time * 2 * np.pi) + np.random.normal(0, 3, n_samples)
    
    X = np.column_stack([time, location, flow_rate, rainfall, temperature])
    
    # Ground truth (for evaluation)
    water_level = 60 + 0.015 * flow_rate + 2 * rainfall + np.random.normal(0, 3, n_samples)
    sediment = 8 + 0.004 * flow_rate + 0.3 * rainfall + np.random.normal(0, 0.8, n_samples)
    y_true = np.column_stack([water_level, sediment])
    
    return X, y_true


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Cross-Verification System for Yellow River PINN'
    )
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data CSV')
    parser.add_argument('--generate-demo', action='store_true',
                        help='Generate demo data with extreme conditions')
    parser.add_argument('--pinn-model', type=str, default='models/model_pinn.pth')
    parser.add_argument('--baseline-model', type=str, default='models/model_baseline.pth')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Divergence threshold for high uncertainty (default 0.10 meaning 10%%)')
    parser.add_argument('--save-path', type=str, default='results/multi_model_verification.png')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Multi-Model Cross-Verification System")
    print("Physics-Informed vs Data-Driven Ensemble")
    print("=" * 70)
    
    # Initialize verifier
    verifier = MultiModelVerifier(
        pinn_path=args.pinn_model,
        baseline_path=args.baseline_model,
        divergence_threshold=args.threshold
    )
    
    # Load or generate test data
    if args.generate_demo or args.test_data is None:
        print("\n[Data] Generating demo data with extreme conditions...")
        X_test, y_true = generate_demo_data(n_samples=200)
    else:
        print(f"\n[Data] Loading test data from {args.test_data}")
        df = pd.read_csv(args.test_data)
        X_test = df[['time', 'location', 'flow_rate', 'rainfall', 'temperature']].values
        if 'water_level' in df.columns and 'sediment_content' in df.columns:
            y_true = df[['water_level', 'sediment_content']].values
        else:
            y_true = None
    
    # Run verification
    print("\n[Verify] Running cross-verification...")
    results, fig = verifier.verify_and_visualize(X_test, y_true, args.save_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Verification Results Summary")
    print("=" * 70)
    print(f"Water Level - Mean Divergence: {np.mean(results['divergence'][:, 0]):.2%}")
    print(f"             High Uncertainty:  {results['high_uncertainty_ratio'][0]:.1%}")
    print(f"Sediment    - Mean Divergence: {np.mean(results['divergence'][:, 1]):.2%}")
    print(f"             High Uncertainty:  {results['high_uncertainty_ratio'][1]:.1%}")
    
    if np.mean(results['is_high_uncertainty']) > 0.2:
        print("\n⚠️  CAUTION: Significant model disagreement detected!")
        print("    Recommend human review for high-uncertainty regions.")
    else:
        print("\n✓  Models show good agreement. Predictions are reliable.")
    
    print(f"\n[Output] Saved to: {args.save_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
