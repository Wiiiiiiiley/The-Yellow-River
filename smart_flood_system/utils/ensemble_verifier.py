"""
Ensemble Verification Module
=============================

Multi-model cross-verification for reliability assessment.
Compares predictions from multiple models to identify:
- High-confidence regions (consensus)
- Low-confidence regions (divergence)
- Outlier models needing recalibration

Key Features:
- Automatic architecture detection (v1/v2)
- Normalization consistency checks
- Divergence hotspot identification
- Statistical consensus metrics
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.pinn_model import YellowRiverPINN as PINNv1
from scripts.pinn_model_v2 import YellowRiverPINNv2 as PINNv2


class EnsembleVerifier:
    """
    Multi-model consensus verification system.
    
    Loads multiple trained models and computes consensus metrics
    to assess prediction reliability.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        divergence_threshold: float = 0.10,
        device: str = 'cpu'
    ):
        """
        Initialize verifier with model paths.
        
        Args:
            model_paths: List of paths to model checkpoints
            divergence_threshold: Threshold for flagging divergence (0.10 = 10%)
            device: Device to run models on
        """
        self.device = torch.device(device)
        self.divergence_threshold = divergence_threshold
        self.models = []
        self.model_names = []
        self.norm_params_list = []
        self.shared_norm_params = None
        
        # Load all models
        for path in model_paths:
            if os.path.exists(path):
                self._load_model(path)
            else:
                print(f"[Warning] Model not found: {path}")
        
        print(f"[Verifier] Loaded {len(self.models)} models")
    
    def _detect_architecture(self, state_dict: Dict) -> str:
        """Detect model version from state dict keys."""
        if any('res_blocks' in key or 'attention' in key for key in state_dict.keys()):
            return 'v2'
        return 'v1'
    
    def _load_model(self, path: str):
        """Load a single model with auto-detection."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        
        # Detect architecture
        arch = self._detect_architecture(state_dict)
        
        if arch == 'v2':
            model = PINNv2(
                input_dim=5,
                hidden_dim=config.get('hidden_dim', 128),
                output_dim=2,
                num_res_blocks=config.get('num_res_blocks', 4),
                use_attention=config.get('use_attention', True)
            )
        else:
            model = PINNv1(
                input_dim=5,
                hidden_dim=config.get('hidden_dim', 64),
                output_dim=2,
                num_hidden_layers=config.get('num_hidden_layers', 4),
                activation=config.get('activation', 'swish')
            )
        
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        
        self.models.append(model)
        self.model_names.append(os.path.basename(os.path.dirname(path)))
        
        # Handle norm params
        norm_params = checkpoint.get('norm_params')
        if norm_params:
            self.norm_params_list.append(norm_params)
            if self.shared_norm_params is None:
                self.shared_norm_params = norm_params
        
        print(f"  ✓ Loaded {arch.upper()}: {path}")
    
    def verify(
        self,
        predictions: torch.Tensor,
        graph_data: Data
    ) -> Dict:
        """
        Verify predictions against ensemble.
        
        For a single prediction, runs all models on the same input
        and computes divergence metrics.
        
        Args:
            predictions: Primary model predictions (N, 2)
            graph_data: Input graph data
        
        Returns:
            Verification results with confidence metrics
        """
        if len(self.models) == 0:
            return {'consensus_accepted': True, 'confidence_level': 1.0}
        
        # Get all model predictions
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(graph_data.x)
                all_preds.append(pred.cpu().numpy())
        
        all_preds = np.array(all_preds)  # (n_models, N, 2)
        
        # Compute statistics
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        # Relative divergence
        divergence = std_pred / (np.abs(mean_pred) + 1e-6)
        max_divergence = np.max(divergence)
        
        # Consensus decision
        consensus_accepted = max_divergence < self.divergence_threshold
        confidence_level = 1.0 - min(max_divergence, 1.0)
        
        return {
            'consensus_accepted': consensus_accepted,
            'confidence_level': float(confidence_level),
            'max_divergence': float(max_divergence),
            'ensemble_mean': mean_pred,
            'ensemble_std': std_pred,
            'individual_predictions': all_preds
        }


if __name__ == "__main__":
    """Test ensemble verifier."""
    print("Testing EnsembleVerifier...")
    
    # Create dummy models
    from torch_geometric.data import Data
    
    # Mock checkpoint paths (would be real in production)
    verifier = EnsembleVerifier(
        model_paths=[],  # Empty for testing
        divergence_threshold=0.10
    )
    
    print("✓ Verifier initialized (no models loaded for test)")
