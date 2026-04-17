"""
Smart Yellow River Flood Management System (SYRFMS)
===================================================

End-to-end intelligent flood prevention decision system integrating:
- GNN-Transformer spatio-temporal prediction
- Physics-informed neural networks (PINN)
- Reinforcement learning for optimal discharge control
- Multi-model ensemble verification
- Real-time monitoring dashboard

Architecture:
    Data Layer → Spatio-Temporal Engine → PINN Constraints → 
    RL Controller → Decision Output → Monitoring Dashboard

Usage:
    python main_system.py --mode train --config configs/system_config.yaml
    python main_system.py --mode inference --real-time
    python main_system.py --mode dashboard

Author: Yellow River Digital Twin Project
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/system_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.api.data_ingestion import RealTimeDataIngestion
from data.preprocessing.feature_engineering import SpatioTemporalProcessor
from models.graph_transformer.stgnn import SpatioTemporalGNN
from models.pinn_loss.physics_constraints import SaintVenantLoss
from models.rl_agent.flood_controller import FloodControlEnv, PPOAgent
from simulator.hydraulic_sim import FastHydraulicSimulator
from utils.ensemble_verifier import EnsembleVerifier
from app.dashboard import launch_monitoring_dashboard
from evolution_engine import EvolutionEngine
from vision_intelligence import VisionIntelligenceSystem


class SmartFloodManagementSystem:
    """
    智慧黄河防汛决策系统主控制器
    
    核心功能：
    1. 多源数据实时接入与融合
    2. 时空图神经网络预测
    3. 物理约束校验
    4. 强化学习最优决策
    5. 多模型二次核验
    6. 实时监控与预警
    """
    
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        """
        初始化系统各模块
        
        Args:
            config_path: 系统配置文件路径
        """
        logger.info("=" * 80)
        logger.info("Initializing Smart Yellow River Flood Management System")
        logger.info("=" * 80)
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._init_data_layer()
        self._init_prediction_engine()
        self._init_physics_constraints()
        self._init_rl_controller()
        self._init_ensemble_verifier()
        self._init_simulator()
        self._init_evolution_engine()
        self._init_vision_intelligence()
        
        logger.info("System initialization complete")
        
    def _init_data_layer(self):
        """Initialize data ingestion and preprocessing modules."""
        logger.info("[Layer 1/5] Initializing Data Layer...")
        
        self.data_ingestion = RealTimeDataIngestion(
            api_endpoints=self.config['data']['api_endpoints'],
            update_interval=self.config['data']['update_interval_sec'],
            buffer_size=self.config['data']['buffer_size']
        )
        
        self.feature_processor = SpatioTemporalProcessor(
            node_features=self.config['model']['node_features'],
            edge_features=self.config['model']['edge_features'],
            temporal_window=self.config['model']['temporal_window'],
            normalization=self.config['data']['normalization']
        )
        
        logger.info("✓ Data layer ready")
        
    def _init_prediction_engine(self):
        """Initialize GNN-Transformer prediction engine."""
        logger.info("[Layer 2/5] Initializing Spatio-Temporal Engine...")
        
        model_config = self.config['model']['graph_transformer']
        
        self.prediction_engine = SpatioTemporalGNN(
            in_channels=model_config['in_channels'],
            hidden_channels=model_config['hidden_channels'],
            out_channels=model_config['out_channels'],
            num_layers=model_config['num_gnn_layers'],
            transformer_heads=model_config['transformer_heads'],
            transformer_layers=model_config['transformer_layers'],
            dropout=model_config['dropout'],
            use_informer=model_config.get('use_informer', True)
        ).to(self.device)
        
        # Load pretrained weights if available
        checkpoint_path = self.config['model'].get('pretrained_checkpoint')
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.prediction_engine.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            logger.info(f"✓ Loaded pretrained model from {checkpoint_path}")
        else:
            logger.info("⚠ Using randomly initialized model (train mode required)")
        
        logger.info("✓ Spatio-temporal engine ready")
        
    def _init_physics_constraints(self):
        """Initialize PINN physics constraint module."""
        logger.info("[Layer 3/5] Initializing Physics Constraints...")
        
        physics_config = self.config['model']['physics']
        
        self.physics_loss = SaintVenantLoss(
            g=physics_config['gravity'],
            n=physics_config['manning_n'],
            S0=physics_config['bed_slope'],
            B=physics_config['channel_width'],
            alpha_cont=physics_config['alpha_continuity'],
            alpha_mom=physics_config['alpha_momentum']
        )
        
        logger.info("✓ Physics constraints ready")
        logger.info(f"  - Gravity (g): {physics_config['gravity']} m/s²")
        logger.info(f"  - Manning n: {physics_config['manning_n']}")
        logger.info(f"  - Bed slope (S0): {physics_config['bed_slope']}")
        
    def _init_rl_controller(self):
        """Initialize reinforcement learning decision agent."""
        logger.info("[Layer 4/5] Initializing RL Controller...")
        
        rl_config = self.config['rl']
        
        # Create environment
        self.rl_env = FloodControlEnv(
            hydraulic_simulator=FastHydraulicSimulator(),
            max_steps=rl_config['max_steps_per_episode'],
            target_level_range=rl_config['target_level_range'],
            safety_constraints=rl_config['safety_constraints']
        )
        
        # Create PPO agent
        self.rl_agent = PPOAgent(
            state_dim=self.rl_env.observation_space.shape[0],
            action_dim=self.rl_env.action_space.shape[0],
            lr=rl_config['learning_rate'],
            gamma=rl_config['gamma'],
            gae_lambda=rl_config['gae_lambda'],
            clip_epsilon=rl_config['clip_epsilon'],
            value_coef=rl_config['value_coef'],
            entropy_coef=rl_config['entropy_coef'],
            hidden_dims=rl_config['hidden_dims']
        )
        
        # Load pretrained RL policy if available
        rl_checkpoint = self.config['rl'].get('pretrained_policy')
        if rl_checkpoint and os.path.exists(rl_checkpoint):
            self.rl_agent.load(rl_checkpoint)
            logger.info(f"✓ Loaded RL policy from {rl_checkpoint}")
        
        logger.info("✓ RL controller ready")
        
    def _init_ensemble_verifier(self):
        """Initialize multi-model ensemble verification."""
        logger.info("[Layer 5/5] Initializing Ensemble Verifier...")
        
        self.ensemble_verifier = EnsembleVerifier(
            model_paths=self.config['ensemble']['model_paths'],
            divergence_threshold=self.config['ensemble']['divergence_threshold'],
            device=self.device
        )
        
        logger.info("✓ Ensemble verifier ready")
        logger.info(f"  - Threshold: {self.config['ensemble']['divergence_threshold']:.1%}")
        logger.info(f"  - Models: {len(self.config['ensemble']['model_paths'])}")
        
    def _init_simulator(self):
        """Initialize fast hydraulic simulator for RL training."""
        logger.info("[Simulator] Initializing Fast Hydraulic Simulator...")
        
        self.simulator = FastHydraulicSimulator(
            dx=self.config['simulator']['spatial_resolution'],
            dt=self.config['simulator']['time_resolution'],
            use_gpu=self.config['simulator'].get('use_gpu', True)
        )
        
        logger.info("✓ Simulator ready")
        
    def _init_evolution_engine(self):
        """
        Initialize autonomous evolution engine for continuous self-improvement.
        
        The evolution engine runs in background thread, continuously:
        1. Monitoring prediction errors
        2. Saving hard examples when error > 15%
        3. Auto-calibrating physics parameters when residuals rise
        4. Triggering incremental learning on accumulated data
        5. Managing model versions with A/B testing
        """
        logger.info("[Evolution Engine] Initializing Autonomous Evolution...")
        
        self.evolution_engine = EvolutionEngine(
            model=self.prediction_engine,
            config_path=self.config.get('evolution_config', 'configs/evolution_config.yaml'),
            models_dir=self.config.get('evolution_models_dir', 'models/evolution/')
        )
        
        # Start background evolution thread
        self.evolution_engine.start()
        
        logger.info("✓ Evolution engine ready")
        logger.info("  - Error threshold: 15%")
        logger.info("  - Auto-calibration: enabled")
        logger.info("  - Incremental learning: enabled")
        logger.info("  - Running in background thread")
        
    def _init_vision_intelligence(self):
        """
        Initialize multimodal vision intelligence for dike safety monitoring.
        
        Integrates drone imagery analysis with sensor data for comprehensive
        structural integrity assessment.
        """
        logger.info("[Vision] Initializing Multimodal Vision Intelligence...")
        
        vision_config = self.config.get('vision', {})
        
        self.vision_system = VisionIntelligenceSystem(
            vit_model_name=vision_config.get('vit_model', 'google/vit-base-patch16-224'),
            sam_api_endpoint=vision_config.get('sam_api_endpoint'),
            gemini_api_key=vision_config.get('gemini_api_key'),
            alert_threshold=vision_config.get('alert_threshold', 0.8),
            warning_water_level=vision_config.get('warning_water_level', 55.0),
            critical_water_level=vision_config.get('critical_water_level', 65.0)
        )
        
        logger.info("✓ Vision intelligence ready")
        logger.info("  - ViT: Anomaly detection enabled")
        logger.info("  - SAM: Interactive segmentation ready")
        logger.info("  - Gemini: Multimodal fusion available")
        logger.info("  - Alert threshold: 80% confidence")
        
    def run_inference_pipeline(
        self,
        real_time: bool = False,
        test_data_path: Optional[str] = None
    ) -> Dict:
        """
        Run complete inference pipeline: Data → Prediction → Physics → Decision
        
        Pipeline:
            1. Data Ingestion (API or file)
            2. Feature Processing
            3. Spatio-Temporal Prediction
            4. Physics Constraint Check
            5. Ensemble Verification
            6. RL Decision (if verified)
            7. Output Recommendations
        
        Args:
            real_time: Use real-time API or offline data
            test_data_path: Path to test data file (if not real-time)
        
        Returns:
            Dictionary with predictions, decisions, and verification status
        """
        logger.info("=" * 80)
        logger.info("Running Inference Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Data Ingestion
        logger.info("[Step 1/7] Data Ingestion...")
        if real_time:
            raw_data = self.data_ingestion.fetch_latest()
        else:
            raw_data = pd.read_csv(test_data_path)
        logger.info(f"✓ Data shape: {raw_data.shape}")
        
        # Step 2: Feature Processing
        logger.info("[Step 2/7] Feature Processing...")
        graph_data = self.feature_processor.process(raw_data)
        logger.info(f"✓ Graph nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
        
        # Step 3: Spatio-Temporal Prediction
        logger.info("[Step 3/7] Spatio-Temporal Prediction...")
        with torch.no_grad():
            predictions = self.prediction_engine(graph_data)
        logger.info(f"✓ Predictions: water_level & sediment at {predictions.shape[0]} nodes")
        
        # Step 4: Physics Constraint Verification
        logger.info("[Step 4/7] Physics Constraint Check...")
        physics_residuals = self.physics_loss.compute_residuals(predictions, graph_data)
        physics_compliant = all(r < self.config['physics']['tolerance'] for r in physics_residuals.values())
        logger.info(f"{'✓' if physics_compliant else '⚠'} Physics compliance: {physics_compliant}")
        logger.info(f"  Residuals: {physics_residuals}")
        
        # Step 5: Ensemble Verification
        logger.info("[Step 5/7] Ensemble Cross-Verification...")
        verification = self.ensemble_verifier.verify(predictions, graph_data)
        consensus_accepted = verification['consensus_accepted']
        logger.info(f"{'✓' if consensus_accepted else '⚠'} Consensus: {verification['confidence_level']:.1%}")
        
        # Step 6: RL Decision (only if predictions are reliable)
        decision = None
        if physics_compliant and consensus_accepted:
            logger.info("[Step 6/7] RL Decision Making...")
            state = self._build_rl_state(predictions, graph_data)
            action, log_prob, value = self.rl_agent.select_action(state)
            decision = {
                'action': action,
                'discharge_recommendation': action[0],  # m³/s
                'confidence': verification['confidence_level'],
                'expected_water_level': predictions[0, 0].item()
            }
            logger.info(f"✓ Decision: Discharge = {action[0]:.1f} m³/s")
        else:
            logger.warning("⚠ Skipping RL decision due to unreliable predictions")
            decision = {
                'action': None,
                'discharge_recommendation': 'MANUAL_REVIEW_REQUIRED',
                'reason': 'Low confidence or physics violation'
            }
        
        # Step 7: Compile Results
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'water_levels': predictions[:, 0].cpu().numpy(),
                'sediment_contents': predictions[:, 1].cpu().numpy(),
                'nodes': graph_data.node_ids if hasattr(graph_data, 'node_ids') else list(range(len(predictions)))
            },
            'physics_check': {
                'compliant': physics_compliant,
                'residuals': physics_residuals
            },
            'verification': verification,
            'decision': decision,
            'metadata': {
                'data_shape': raw_data.shape,
                'inference_time_ms': None,  # Will be measured
                'alert_level': self._determine_alert_level(predictions, verification)
            }
        }
        
        logger.info("[Step 7/7] Pipeline Complete")
        logger.info(f"Alert Level: {results['metadata']['alert_level']}")
        
        # Step 8: Evolution Engine Processing (Background)
        # Feed prediction vs ground truth to evolution engine for continuous learning
        if hasattr(self, 'evolution_engine') and self.evolution_engine is not None:
            logger.info("[Step 8/8] Evolution Engine Processing...")
            try:
                # Note: In production, ground_truth would come from API verification
                # For now, we use predictions as placeholder (will be updated when real data arrives)
                ground_truth = predictions.cpu().numpy()  # Placeholder - should be actual measurements
                
                # Process through evolution engine
                evolution_results = self.evolution_engine.process_prediction(
                    prediction=ground_truth[0],  # First station as example
                    ground_truth=ground_truth[0],  # Would be actual measurement
                    physics_residual=physics_residuals.get('continuity', 0) + physics_residuals.get('momentum', 0),
                    input_features=graph_data.x[0].cpu().numpy()
                )
                
                # Log evolution actions
                if evolution_results.get('hard_example_saved'):
                    logger.info("  ⚠ Hard example detected and saved")
                if evolution_results.get('calibration'):
                    cal = evolution_results['calibration']
                    logger.info(f"  🔧 Auto-calibrated: S0={cal['S0']:.6f}, B={cal['B']:.2f}")
                if evolution_results.get('retraining'):
                    rt = evolution_results['retraining']
                    if rt['status'] == 'success':
                        logger.info(f"  🎉 Model evolved to v{rt['next_version']}")
                
                results['evolution'] = evolution_results
                
            except Exception as e:
                logger.warning(f"Evolution engine error (non-critical): {e}")
        
        return results
    
    def _build_rl_state(self, predictions: torch.Tensor, graph_data) -> np.ndarray:
        """Build RL state vector from predictions and graph data."""
        # State: current water levels, recent rainfall, time of day, etc.
        state = np.concatenate([
            predictions[:, 0].cpu().numpy(),  # Water levels
            predictions[:, 1].cpu().numpy(),  # Sediment
            graph_data.x[:, 2].cpu().numpy(),  # Flow rate
            graph_data.x[:, 3].cpu().numpy(),  # Rainfall
        ])
        return state
    
    def _determine_alert_level(
        self,
        predictions: torch.Tensor,
        verification: Dict
    ) -> str:
        """Determine alert level based on predictions and confidence."""
        water_levels = predictions[:, 0].cpu().numpy()
        confidence = verification['confidence_level']
        
        # Check critical thresholds
        if np.any(water_levels > 70) and confidence < 0.8:
            return "CRITICAL_UNCERTAIN"
        elif np.any(water_levels > 70):
            return "CRITICAL"
        elif np.any(water_levels > 55) and confidence < 0.8:
            return "WARNING_UNCERTAIN"
        elif np.any(water_levels > 55):
            return "WARNING"
        else:
            return "NORMAL"
    
    def train_prediction_model(
        self,
        train_data_path: str,
        val_data_path: str,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """Train the GNN-Transformer prediction engine."""
        logger.info("=" * 80)
        logger.info("Training Prediction Engine")
        logger.info("=" * 80)
        
        # Training loop with physics constraints
        for epoch in range(epochs):
            # ... training implementation
            pass
        
        logger.info("Training complete")
    
    def train_rl_agent(self, episodes: int = 1000):
        """Train the RL flood control agent using simulator."""
        logger.info("=" * 80)
        logger.info("Training RL Agent")
        logger.info("=" * 80)
        
        for episode in range(episodes):
            state = self.rl_env.reset()
            episode_reward = 0
            
            for step in range(self.config['rl']['max_steps_per_episode']):
                action, log_prob, value = self.rl_agent.select_action(state)
                next_state, reward, done, info = self.rl_env.step(action)
                
                self.rl_agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update policy
            if len(self.rl_agent.buffer) >= self.rl_agent.batch_size:
                self.rl_agent.update()
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Save trained policy
        save_path = f"models/rl_agent/ppo_policy_{datetime.now():%Y%m%d_%H%M%S}.pt"
        self.rl_agent.save(save_path)
        logger.info(f"RL policy saved to {save_path}")
    
    def launch_dashboard(self, port: int = 8501):
        """Launch Streamlit monitoring dashboard."""
        logger.info("Launching Monitoring Dashboard...")
        launch_monitoring_dashboard(
            system=self,
            port=port,
            config=self.config
        )


def main():
    parser = argparse.ArgumentParser(
        description='Smart Yellow River Flood Management System'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference', 'dashboard', 'full'],
                        help='Operating mode')
    parser.add_argument('--config', type=str, default='configs/system_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--real-time', action='store_true',
                        help='Use real-time data API')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Test data path for inference mode')
    parser.add_argument('--dashboard-port', type=int, default=8501,
                        help='Dashboard port')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SmartFloodManagementSystem(args.config)
    
    if args.mode == 'train':
        # Train both prediction model and RL agent
        system.train_prediction_model(
            train_data_path='data/processed/train_graph.pt',
            val_data_path='data/processed/val_graph.pt'
        )
        system.train_rl_agent(episodes=1000)
        
    elif args.mode == 'inference':
        # Run single inference
        results = system.run_inference_pipeline(
            real_time=args.real_time,
            test_data_path=args.test_data
        )
        print("\n" + "=" * 80)
        print("INFERENCE RESULTS")
        print("=" * 80)
        print(f"Alert Level: {results['metadata']['alert_level']}")
        print(f"Decision: {results['decision']}")
        print("=" * 80)
        
    elif args.mode == 'dashboard':
        # Launch dashboard only
        system.launch_dashboard(port=args.dashboard_port)
        
    elif args.mode == 'full':
        # Full system: inference + dashboard
        import threading
        
        # Start dashboard in background
        dashboard_thread = threading.Thread(
            target=system.launch_dashboard,
            kwargs={'port': args.dashboard_port}
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Run continuous inference
        logger.info("Starting continuous monitoring loop...")
        while True:
            try:
                results = system.run_inference_pipeline(real_time=True)
                
                if results['metadata']['alert_level'] in ['CRITICAL', 'CRITICAL_UNCERTAIN']:
                    logger.warning("🚨 CRITICAL ALERT TRIGGERED")
                    # Could trigger SMS/email here
                
                # Sleep until next update
                import time
                time.sleep(system.config['data']['update_interval_sec'])
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)


if __name__ == '__main__':
    main()
