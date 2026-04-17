"""
AutoPilot: Fully Autonomous "Collect-Train-Evolve" Closed-Loop System
======================================================================

Central orchestrator for autonomous AI lifecycle management:
1. Auto-Downloader: Daily API data ingestion
2. Auto-Trainer: Incremental learning on accumulated data
3. Self-Evolution: Physics calibration & model promotion
4. Auto-Dashboard: Evolution reporting & monitoring

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Scheduler  │────→│ Async Queue │────→│  Workers    │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                    │
    ┌─────────────┐     ┌─────────────┐          │
    │   Monitor   │←────│  AutoPilot  │←─────────┘
    └─────────────┘     └─────────────┘

Usage:
    python auto_pilot.py --mode full
    python auto_pilot.py --mode download-only
    python auto_pilot.py --trigger-evolution
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue

# APScheduler for task management
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'auto_pilot_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from api_agent import APIDownloader
from learning_manager import LearningManager
from self_evolution import SelfEvolutionEngine
from auto_dashboard import AutoDashboard


@dataclass
class SystemStatus:
    """Track system operational status."""
    version: str
    last_download: Optional[str]
    last_training: Optional[str]
    last_evolution: Optional[str]
    accumulated_data_days: int
    current_model_generation: int
    physics_residual: float
    is_healthy: bool
    active_jobs: List[str]


class AutoPilot:
    """
    Central orchestrator for autonomous AI lifecycle.
    
    Manages the complete loop:
    1. Daily data download (2 AM)
    2. Weekly training check (accumulated data threshold)
    3. Post-training evolution (model promotion/demotion)
    4. Continuous monitoring & reporting
    """
    
    def __init__(self, config_path: str = "configs/auto_pilot.yaml"):
        """Initialize autonomous system."""
        logger.info("=" * 80)
        logger.info("🚀 Initializing AutoPilot - Autonomous AI Lifecycle Manager")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize status tracking
        self.status = SystemStatus(
            version="1.0.0",
            last_download=None,
            last_training=None,
            last_evolution=None,
            accumulated_data_days=0,
            current_model_generation=1,
            physics_residual=float('inf'),
            is_healthy=True,
            active_jobs=[]
        )
        
        # Async task queue
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        
        # Initialize modules
        self._init_modules()
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        self._setup_schedules()
        
        logger.info("✓ AutoPilot initialization complete")
        
    def _load_config(self, path: str) -> Dict:
        """Load or create default configuration."""
        import yaml
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        default_config = {
            'download': {
                'schedule': {'hour': 2, 'minute': 0},  # 2:00 AM daily
                'apis': ['hydrology', 'meteorology'],
                'retry_attempts': 3,
                'archive_dir': 'data/archive/'
            },
            'training': {
                'trigger_days': 7,  # Trigger after 7 days of new data
                'historical_buffer_days': 30,  # Keep 30 days history
                'learning_rate': 0.0001,
                'epochs': 100
            },
            'evolution': {
                'physics_residual_threshold': 0.1,
                'validation_improvement_threshold': 0.05,
                'auto_hp_search': True
            },
            'dashboard': {
                'report_interval_hours': 24,
                'streamlit_port': 8501
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(default_config, f)
        
        return default_config
    
    def _init_modules(self):
        """Initialize autonomous modules."""
        logger.info("[AutoPilot] Initializing modules...")
        
        # Module 1: Auto-Downloader
        self.downloader = APIDownloader(
            apis=self.config['download']['apis'],
            archive_dir=self.config['download']['archive_dir'],
            retry_attempts=self.config['download']['retry_attempts']
        )
        logger.info("  ✓ APIDownloader ready")
        
        # Module 2: Auto-Trainer
        self.learning_manager = LearningManager(
            trigger_days=self.config['training']['trigger_days'],
            buffer_days=self.config['training']['historical_buffer_days'],
            lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs']
        )
        logger.info("  ✓ LearningManager ready")
        
        # Module 3: Self-Evolution
        self.evolution_engine = SelfEvolutionEngine(
            residual_threshold=self.config['evolution']['physics_residual_threshold'],
            improvement_threshold=self.config['evolution']['validation_improvement_threshold'],
            auto_hp_search=self.config['evolution']['auto_hp_search']
        )
        logger.info("  ✓ SelfEvolutionEngine ready")
        
        # Module 4: Auto-Dashboard
        self.dashboard = AutoDashboard(
            report_interval=self.config['dashboard']['report_interval_hours'],
            port=self.config['dashboard']['streamlit_port']
        )
        logger.info("  ✓ AutoDashboard ready")
    
    def _setup_schedules(self):
        """Setup scheduled jobs."""
        logger.info("[AutoPilot] Setting up schedules...")
        
        # Job 1: Daily data download at 2 AM
        download_time = self.config['download']['schedule']
        self.scheduler.add_job(
            self._scheduled_download,
            CronTrigger(hour=download_time['hour'], minute=download_time['minute']),
            id='daily_download',
            name='Daily API Data Download',
            replace_existing=True
        )
        logger.info(f"  ✓ Scheduled: Daily download at {download_time['hour']:02d}:{download_time['minute']:02d}")
        
        # Job 2: Check training trigger every 6 hours
        self.scheduler.add_job(
            self._check_training_trigger,
            IntervalTrigger(hours=6),
            id='training_check',
            name='Training Trigger Check',
            replace_existing=True
        )
        logger.info("  ✓ Scheduled: Training check every 6 hours")
        
        # Job 3: Health monitoring every 5 minutes
        self.scheduler.add_job(
            self._health_check,
            IntervalTrigger(minutes=5),
            id='health_monitor',
            name='System Health Check',
            replace_existing=True
        )
        logger.info("  ✓ Scheduled: Health check every 5 minutes")
        
        # Job 4: Generate evolution report daily at 8 AM
        self.scheduler.add_job(
            self._generate_evolution_report,
            CronTrigger(hour=8, minute=0),
            id='daily_report',
            name='Daily Evolution Report',
            replace_existing=True
        )
        logger.info("  ✓ Scheduled: Daily report at 08:00")
        
        # Job 5: Monthly backtrack test (1st of each month at 3 AM)
        self.scheduler.add_job(
            self._monthly_backtrack_test,
            CronTrigger(day=1, hour=3, minute=0),
            id='monthly_backtrack',
            name='Monthly 2021 Flood Backtrack Test',
            replace_existing=True
        )
        logger.info("  ✓ Scheduled: Monthly backtrack test (1st @ 03:00) - prevents catastrophic forgetting")
    
    async def _scheduled_download(self):
        """Execute scheduled data download."""
        logger.info("=" * 80)
        logger.info("[SCHEDULED] Starting daily data download")
        logger.info("=" * 80)
        
        try:
            # Run download in background thread
            result = await asyncio.to_thread(
                self.downloader.download_yesterday_data
            )
            
            if result['success']:
                self.status.last_download = datetime.now().isoformat()
                self.status.accumulated_data_days += 1
                
                logger.info(f"✓ Downloaded {result['records']} records")
                logger.info(f"✓ Accumulated data: {self.status.accumulated_data_days} days")
                
                # Notify learning manager
                await self.task_queue.put({
                    'type': 'new_data',
                    'records': result['records'],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error(f"✗ Download failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            self.status.is_healthy = False
    
    async def _check_training_trigger(self):
        """Check if training should be triggered."""
        if self.status.accumulated_data_days >= self.config['training']['trigger_days']:
            logger.info("=" * 80)
            logger.info(f"[TRIGGER] Accumulated {self.status.accumulated_data_days} days of data")
            logger.info("[TRIGGER] Starting autonomous training...")
            logger.info("=" * 80)
            
            try:
                # Trigger training
                training_result = await asyncio.to_thread(
                    self.learning_manager.run_incremental_training
                )
                
                if training_result['success']:
                    self.status.last_training = datetime.now().isoformat()
                    
                    logger.info("✓ Training completed")
                    logger.info(f"  Model: {training_result['model_path']}")
                    logger.info(f"  Val loss: {training_result['val_loss']:.4f}")
                    
                    # Reset accumulated days
                    self.status.accumulated_data_days = 0
                    
                    # Trigger evolution
                    await self._run_evolution(training_result)
                else:
                    logger.error(f"✗ Training failed: {training_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Training error: {e}")
    
    async def _run_evolution(self, training_result: Dict):
        """Run self-evolution after training."""
        logger.info("=" * 80)
        logger.info("[EVOLUTION] Starting self-evolution process")
        logger.info("=" * 80)
        
        try:
            evolution_result = await asyncio.to_thread(
                self.evolution_engine.evaluate_and_evolve,
                new_model_path=training_result['model_path'],
                val_loss=training_result['val_loss']
            )
            
            self.status.last_evolution = datetime.now().isoformat()
            self.status.current_model_generation = evolution_result['generation']
            self.status.physics_residual = evolution_result['physics_residual']
            
            if evolution_result['promoted']:
                logger.info(f"🎉 Model promoted to Gen-{evolution_result['generation']}!")
                logger.info(f"  Physics residual: {evolution_result['physics_residual']:.4f}")
            else:
                logger.info("Model not promoted - keeping previous version")
                
            # Generate report
            await asyncio.to_thread(
                self.dashboard.generate_evolution_report,
                evolution_result
            )
            
        except Exception as e:
            logger.error(f"Evolution error: {e}")
    
    async def _health_check(self):
        """Monitor system health."""
        try:
            # Check disk space
            import shutil
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024**3)
            
            if free_gb < 1.0:
                logger.warning(f"⚠ Low disk space: {free_gb:.1f} GB remaining")
                self.status.is_healthy = False
            
            # Check API connectivity
            api_healthy = await asyncio.to_thread(self.downloader.check_health)
            if not api_healthy:
                logger.warning("⚠ API connectivity issues detected")
            
            # Update status
            jobs = self.scheduler.get_jobs()
            self.status.active_jobs = [j.name for j in jobs]
            
            logger.debug(f"Health check passed. Active jobs: {len(jobs)}")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def _generate_evolution_report(self):
        """Generate daily evolution report."""
        try:
            report_path = await asyncio.to_thread(
                self.dashboard.generate_daily_report,
                asdict(self.status)
            )
            
            logger.info(f"✓ Evolution report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
    
    async def _monthly_backtrack_test(self):
        """
        Monthly regression test on 2021 extreme flood data.
        
        This prevents catastrophic forgetting - ensures evolved models
        still remember how to handle historical extreme events.
        """
        logger.info("=" * 80)
        logger.info("[SCHEDULED] Monthly 2021 Flood Backtrack Test")
        logger.info("=" * 80)
        
        try:
            from backtrack_validator import BacktrackValidator, BacktrackAlertManager
            
            # Initialize validator
            validator = BacktrackValidator(
                degradation_threshold=0.10,
                rollback_threshold=0.20
            )
            
            # Get current best model
            best_model_path = self.config.get('evolution', {}).get('model_dir', 'models/autopilot/') + "best_model.pth"
            
            # Run backtrack test
            result = await asyncio.to_thread(
                validator.run_monthly_backtrack_test,
                current_model_path=best_model_path,
                current_generation=self.status.current_model_generation
            )
            
            # Handle results
            if result.degradation_detected:
                logger.error("=" * 80)
                logger.error("🚨 BACKTRACK ALERT: Catastrophic Forgetting Detected!")
                logger.error("=" * 80)
                logger.error(f"Model Gen-{result.model_generation} degraded by {result.degradation_ratio:.1%}")
                logger.error("Failed to handle 2021 extreme flood patterns!")
                
                # Send alert
                alert_mgr = BacktrackAlertManager()
                await asyncio.to_thread(alert_mgr.send_alert, result)
                
                # Update status
                self.status.is_healthy = False
                
                if result.rollback_recommended:
                    logger.error("🚫 CRITICAL: Rollback recommended!")
                    # Could trigger automatic rollback here
            else:
                logger.info("✓ Backtrack test passed - no forgetting detected")
                logger.info(f"  Degradation: {result.degradation_ratio:.1%} (within tolerance)")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Backtrack test error: {e}")
    
    def trigger_manual_evolution(self):
        """Manual trigger for evolution (for testing)."""
        logger.info("[MANUAL] Triggering evolution...")
        asyncio.create_task(self._run_evolution({
            'model_path': 'models/latest_checkpoint.pt',
            'val_loss': 0.05
        }))
    
    async def start(self):
        """Start the autonomous system."""
        logger.info("=" * 80)
        logger.info("🚀 Starting AutoPilot - All systems autonomous")
        logger.info("=" * 80)
        
        # Start scheduler
        self.scheduler.start()
        logger.info("✓ Scheduler started")
        
        # Start dashboard server
        dashboard_task = asyncio.create_task(
            self.dashboard.start_server()
        )
        logger.info("✓ Dashboard server starting")
        
        # Main loop
        try:
            while True:
                # Process any queued tasks
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                    logger.info(f"Processing task: {task['type']}")
                    
                except asyncio.TimeoutError:
                    pass
                
                # Periodic status log
                if datetime.now().minute == 0:  # Every hour
                    logger.info(f"[STATUS] Gen-{self.status.current_model_generation}, "
                              f"{self.status.accumulated_data_days} days queued, "
                              f"residual={self.status.physics_residual:.4f}")
                
                await asyncio.sleep(60)  # Sleep 1 minute
                
        except KeyboardInterrupt:
            logger.info("Shutting down AutoPilot...")
            self.scheduler.shutdown()
            dashboard_task.cancel()
            
    async def stop(self):
        """Gracefully stop the system."""
        logger.info("Stopping AutoPilot...")
        self.scheduler.shutdown(wait=True)
        logger.info("✓ AutoPilot stopped")


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='AutoPilot - Autonomous AI Lifecycle')
    parser.add_argument('--config', type=str, default='configs/auto_pilot.yaml')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'download-only', 'train-only', 'evolve-only'])
    parser.add_argument('--trigger-evolution', action='store_true',
                        help='Manually trigger evolution cycle')
    
    args = parser.parse_args()
    
    # Initialize
    autopilot = AutoPilot(config_path=args.config)
    
    if args.trigger_evolution:
        autopilot.trigger_manual_evolution()
        await asyncio.sleep(5)  # Let it process
    else:
        # Start full system
        await autopilot.start()


if __name__ == '__main__':
    # Run async main
    asyncio.run(main())
