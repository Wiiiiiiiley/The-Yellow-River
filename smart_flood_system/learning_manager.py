"""
Learning Manager - Automated Incremental Training
=================================================

Autonomous training pipeline:
1. Monitor accumulated data
2. Merge new data with historical buffer
3. Fine-tune model (preserving personality)
4. Validate and checkpoint

Key Features:
- Continuous learning on 7-day rolling windows
- Memory retention (20% historical buffer)
- Personality-preserving fine-tuning
- Automatic validation
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import personality model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from train_v2_personality import YellowRiverPersonalityPINN, SeasonalEncoder

logger = logging.getLogger(__name__)


class LearningManager:
    """
    Autonomous learning manager for continuous model improvement.
    
    Strategy:
    - Accumulate 7 days of new data
    - Merge with 20% historical buffer (30 days)
    - Fine-tune with low learning rate
    - Preserve learned personality
    """
    
    def __init__(
        self,
        trigger_days: int = 7,
        buffer_days: int = 30,
        lr: float = 0.0001,
        epochs: int = 100,
        batch_size: int = 32,
        model_dir: str = "models/autopilot/"
    ):
        self.trigger_days = trigger_days
        self.buffer_days = buffer_days
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Data tracking
        self.accumulated_data: List[pd.DataFrame] = []
        self.training_history: List[Dict] = []
        
        logger.info(f"LearningManager initialized: trigger={trigger_days}d, buffer={buffer_days}d")
    
    def add_new_data(self, data: pd.DataFrame, date: datetime):
        """Add newly downloaded data to accumulation buffer."""
        self.accumulated_data.append(data)
        
        logger.info(f"Added data for {date}: {len(data)} records")
        logger.info(f"Accumulated: {len(self.accumulated_data)} days")
        
        # Check if training should trigger
        return len(self.accumulated_data) >= self.trigger_days
    
    def run_incremental_training(self) -> Dict:
        """
        Run incremental training on accumulated data.
        
        Returns:
            Training result with model path and metrics
        """
        logger.info("=" * 80)
        logger.info("Starting Incremental Training")
        logger.info("=" * 80)
        
        try:
            # 1. Load historical buffer (20% = 6 days from last 30)
            historical_data = self._load_historical_buffer()
            
            # 2. Merge with new accumulated data
            training_data = self._prepare_training_data(historical_data)
            
            # 3. Load current best model
            model, seasonal_encoder = self._load_current_model()
            
            # 4. Fine-tune with personality preservation
            training_result = self._fine_tune_model(
                model, seasonal_encoder, training_data
            )
            
            # 5. Save checkpoint
            checkpoint_path = self._save_checkpoint(
                model, seasonal_encoder, training_result
            )
            
            # 6. Clear accumulated data
            n_used = len(self.accumulated_data)
            self.accumulated_data.clear()
            
            logger.info(f"✓ Training complete: {checkpoint_path}")
            
            return {
                'success': True,
                'model_path': checkpoint_path,
                'val_loss': training_result['val_loss'],
                'train_loss': training_result['train_loss'],
                'epochs_trained': training_result['epochs'],
                'records_used': len(training_data),
                'n_new_days': n_used,
                'n_historical_days': len(historical_data) if historical_data is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_historical_buffer(self) -> Optional[pd.DataFrame]:
        """Load historical data buffer (last 30 days)."""
        archive_dir = Path("data/archive/hydrology")
        
        if not archive_dir.exists():
            return None
        
        # Find last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.buffer_days)
        
        historical_files = []
        
        for year_month_dir in archive_dir.glob("*/*"):
            for file in year_month_dir.glob("*.parquet"):
                # Parse date from filename
                try:
                    file_date = datetime.strptime(file.stem[:8], "%Y%m%d")
                    if start_date <= file_date <= end_date:
                        historical_files.append(file)
                except:
                    continue
        
        if not historical_files:
            return None
        
        # Load and sample 20% (stratified by time)
        all_data = []
        for file in sorted(historical_files):
            df = pd.read_parquet(file)
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Sample 20% to maintain memory without overloading
        sample_size = int(len(combined) * 0.2)
        sampled = combined.sample(n=sample_size, random_state=42)
        
        logger.info(f"Loaded historical buffer: {len(sampled)} records from {len(historical_files)} days")
        
        return sampled
    
    def _prepare_training_data(
        self,
        historical_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge accumulated new data with historical buffer."""
        # Combine all accumulated data
        new_data = pd.concat(self.accumulated_data, ignore_index=True)
        
        # Add seasonal features
        encoder = SeasonalEncoder()
        
        seasonal = encoder.encode(pd.DatetimeIndex(new_data['timestamp']))
        new_data['sin_year'] = seasonal[:, 0]
        new_data['cos_year'] = seasonal[:, 1]
        new_data['sin_season'] = seasonal[:, 2]
        new_data['cos_season'] = seasonal[:, 3]
        
        # Merge with historical
        if historical_data is not None:
            # Ensure historical has seasonal features
            if 'sin_year' not in historical_data.columns:
                seasonal_hist = encoder.encode(pd.DatetimeIndex(historical_data['timestamp']))
                historical_data['sin_year'] = seasonal_hist[:, 0]
                historical_data['cos_year'] = seasonal_hist[:, 1]
                historical_data['sin_season'] = seasonal_hist[:, 2]
                historical_data['cos_season'] = seasonal_hist[:, 3]
            
            combined = pd.concat([new_data, historical_data], ignore_index=True)
            logger.info(f"Combined data: {len(new_data)} new + {len(historical_data)} historical = {len(combined)} total")
        else:
            combined = new_data
            logger.info(f"Using new data only: {len(combined)} records")
        
        return combined
    
    def _load_current_model(self) -> Tuple[YellowRiverPersonalityPINN, SeasonalEncoder]:
        """Load current best model for fine-tuning."""
        best_model_path = self.model_dir / "best_model.pth"
        
        if best_model_path.exists():
            logger.info(f"Loading model: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            
            model = YellowRiverPersonalityPINN(
                input_dim=10,
                hidden_dim=128,
                output_dim=2,
                num_layers=4,
                use_feature_crossing=True
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            seasonal_encoder = checkpoint.get('seasonal_encoder', SeasonalEncoder())
            
            logger.info("✓ Loaded existing model for fine-tuning")
        else:
            logger.info("No existing model found - initializing new model")
            model = YellowRiverPersonalityPINN(
                input_dim=10,
                hidden_dim=128,
                output_dim=2,
                num_layers=4,
                use_feature_crossing=True
            )
            seasonal_encoder = SeasonalEncoder()
        
        return model, seasonal_encoder
    
    def _fine_tune_model(
        self,
        model: YellowRiverPersonalityPINN,
        encoder: SeasonalEncoder,
        data: pd.DataFrame
    ) -> Dict:
        """
        Fine-tune model with personality preservation.
        
        Strategy:
        - Freeze lower layers (preserve learned patterns)
        - Unfreeze top layers (adapt to new data)
        - Low learning rate (prevent catastrophic forgetting)
        """
        # Prepare tensors
        feature_cols = [
            'flow_rate', 'rainfall', 'temperature',
            'sin_year', 'cos_year', 'sin_season', 'cos_season'
        ]
        
        # Add derived features if missing
        if 'is_flood_season' not in data.columns:
            data['is_flood_season'] = encoder.encode_flood_season(pd.DatetimeIndex(data['timestamp']))
        
        if 'sin_hour' not in data.columns:
            data['hour'] = pd.DatetimeIndex(data['timestamp']).hour
            data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        feature_cols.extend(['is_flood_season', 'sin_hour', 'cos_hour'])
        
        # Convert to tensors
        X = torch.FloatTensor(data[feature_cols].values)
        y = torch.FloatTensor(data[['water_level', 'sediment_content']].values)
        Q = torch.FloatTensor(data['flow_rate'].values).unsqueeze(1)
        
        # Split train/val
        train_size = int(0.8 * len(data))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        Q_train, Q_val = Q[:train_size], Q[train_size:]
        
        # Freeze bottom layers (personality preservation)
        for name, param in model.named_parameters():
            if 'network.0' in name or 'network.1' in name:  # First 2 layers
                param.requires_grad = False
                logger.debug(f"Frozen: {name}")
            else:
                param.requires_grad = True
        
        # Optimizer (only trainable params)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        logger.info(f"Fine-tuning: {self.epochs} epochs, lr={self.lr}")
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward
            pred = model(X_train, Q_train)
            loss = criterion(pred, y_train)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val, Q_val)
                    val_loss = criterion(val_pred, y_val).item()
                    val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
        
        # Final validation
        model.eval()
        with torch.no_grad():
            final_pred = model(X_val, Q_val)
            final_val_loss = criterion(final_pred, y_val).item()
        
        return {
            'train_loss': train_losses[-1],
            'val_loss': final_val_loss,
            'epochs': self.epochs,
            'model': model,
            'encoder': encoder
        }
    
    def _save_checkpoint(
        self,
        model: YellowRiverPersonalityPINN,
        encoder: SeasonalEncoder,
        result: Dict
    ) -> str:
        """Save trained model checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"incremental_{timestamp}.pth"
        checkpoint_path = self.model_dir / checkpoint_name
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'seasonal_encoder': encoder,
            'train_loss': result['train_loss'],
            'val_loss': result['val_loss'],
            'epochs': result['epochs'],
            'timestamp': timestamp,
            'type': 'incremental'
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history


if __name__ == "__main__":
    """Test LearningManager."""
    print("Testing LearningManager...")
    
    # Initialize
    manager = LearningManager(
        trigger_days=2,  # Trigger quickly for testing
        buffer_days=7,
        lr=0.001,
        epochs=10
    )
    
    # Simulate adding data
    for i in range(3):
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
            'station_id': ['HYD_001'] * 24,
            'water_level': np.random.randn(24) * 5 + 50,
            'flow_rate': np.random.randn(24) * 500 + 2000,
            'sediment_content': np.random.randn(24) * 2 + 10,
            'rainfall': np.random.exponential(2, 24),
            'temperature': np.random.randn(24) * 5 + 20
        })
        
        should_trigger = manager.add_new_data(dummy_data, datetime(2024, 1, 1) + timedelta(days=i))
        print(f"Day {i+1}: Trigger={should_trigger}")
    
    # Trigger training
    if len(manager.accumulated_data) >= manager.trigger_days:
        print("\nTriggering training...")
        result = manager.run_incremental_training()
        print(f"Result: {result}")
    
    print("\n✓ LearningManager test complete!")
