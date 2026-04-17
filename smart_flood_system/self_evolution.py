"""
Self-Evolution Engine - Autonomous Model Calibration
====================================================

Post-training evolution with physics-aware model promotion.

Includes "Digital Backtrack" - monthly regression testing on 2021 flood
to prevent catastrophic forgetting.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from backtrack_validator import BacktrackValidator, BacktrackAlertManager

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Result of evolution process."""
    generation: int
    promoted: bool
    physics_residual: float
    val_loss: float
    improvement_ratio: float
    auto_hp_triggered: bool
    timestamp: str
    new_model_path: str


class SelfEvolutionEngine:
    """Autonomous model evolution with physics-aware decision making."""
    
    def __init__(
        self,
        residual_threshold: float = 0.1,
        improvement_threshold: float = 0.05,
        auto_hp_search: bool = True,
        model_dir: str = "models/autopilot/",
        enable_backtrack: bool = True
    ):
        self.residual_threshold = residual_threshold
        self.improvement_threshold = improvement_threshold
        self.auto_hp_search = auto_hp_search
        self.model_dir = Path(model_dir)
        self.enable_backtrack = enable_backtrack
        
        self.current_generation = 1
        
        # Initialize backtrack validator
        if enable_backtrack:
            self.backtrack_validator = BacktrackValidator(
                degradation_threshold=0.10,
                rollback_threshold=0.20
            )
            self.alert_manager = BacktrackAlertManager()
            logger.info("✓ Digital Backtrack enabled (monthly 2021 flood test)")
        
        logger.info(f"SelfEvolutionEngine initialized (Gen-{self.current_generation})")
    
    def evaluate_and_evolve(
        self,
        new_model_path: str,
        val_loss: float
    ) -> Dict:
        """Evaluate new model and decide on evolution."""
        logger.info("=" * 80)
        logger.info("Self-Evolution: Evaluating new model")
        logger.info("=" * 80)
        
        # Mock physics residual evaluation
        new_residual = val_loss * 0.5  # Simplified
        old_residual = 0.15  # Mock old
        
        # === DIGITAL BACKTRACK: Monthly 2021 flood test ===
        backtrack_passed = True
        backtrack_result = None
        
        if self.enable_backtrack:
            # Check if it's test day (1st of month) or forced test
            if self.backtrack_validator.should_run_today():
                logger.info("[BACKTRACK] Running monthly 2021 flood regression test...")
                
                backtrack_result = self.backtrack_validator.run_monthly_backtrack_test(
                    current_model_path=new_model_path,
                    current_generation=self.current_generation
                )
                
                backtrack_passed = not backtrack_result.degradation_detected
                
                # Send alert if degradation detected
                if backtrack_result.alert_triggered:
                    self.alert_manager.send_alert(backtrack_result)
                
                logger.info(f"[BACKTRACK] Test passed: {backtrack_passed}")
                if backtrack_result.degradation_detected:
                    logger.warning(f"[BACKTRACK] Degradation: {backtrack_result.degradation_ratio:.1%}")
        
        # Decision logic: must pass both physics AND backtrack
        physics_ok = new_residual < old_residual * 0.95 and val_loss < 0.1
        promoted = physics_ok and backtrack_passed
        
        auto_hp_triggered = new_residual > self.residual_threshold and self.auto_hp_search
        
        if promoted:
            self.current_generation += 1
            logger.info(f"🎉 Model promoted to Generation {self.current_generation}!")
            if backtrack_result:
                logger.info(f"   ✓ Backtrack test passed ({backtrack_result.degradation_ratio:.1%})")
            
            # Copy to best_model
            best_path = self.model_dir / "best_model.pth"
            shutil.copy(new_model_path, best_path)
        elif not backtrack_passed:
            logger.error(f"🚫 Model NOT promoted - failed 2021 flood backtrack test!")
            logger.error(f"   Degradation: {backtrack_result.degradation_ratio:.1%}")
            logger.error(f"   Model may have forgotten extreme flood patterns!")
        
        if auto_hp_triggered:
            logger.warning(f"Physics residual ({new_residual:.4f}) exceeds threshold")
            self._run_auto_hp_search()
        
        result = EvolutionResult(
            generation=self.current_generation,
            promoted=promoted,
            physics_residual=new_residual,
            val_loss=val_loss,
            improvement_ratio=(old_residual - new_residual) / old_residual,
            auto_hp_triggered=auto_hp_triggered,
            timestamp=datetime.now().isoformat(),
            new_model_path=new_model_path
        )
        
        # Add backtrack info
        if backtrack_result:
            result['backtrack_test'] = asdict(backtrack_result)
        
        return asdict(result)
    
    def _run_auto_hp_search(self):
        """Auto-HP search for parameter calibration."""
        logger.info("Running Auto-HP Search for n, S0, B calibration...")
        # Simplified implementation
        logger.info("✓ Auto-HP search complete (mock)")


if __name__ == "__main__":
    print("Testing SelfEvolutionEngine...")
    engine = SelfEvolutionEngine()
    result = engine.evaluate_and_evolve("models/test.pth", 0.08)
    print(f"Result: {result}")
    print("✓ SelfEvolutionEngine test complete!")
