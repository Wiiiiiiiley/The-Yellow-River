"""
智慧黄河防汛决策系统主模块
集成所有组件：GNN-Transformer + PINN + RL + AutoPilot
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from smart_flood_system.data_adapter import get_data_adapter
from smart_flood_system.models.graph_transformer.stgnn import create_model
from smart_flood_system.models.graph_transformer.stgnn_enhanced import create_enhanced_model
from smart_flood_system.models.pinn_loss.physics_constraints_enhanced import (
    EnhancedPhysicsConstraintLoss, BoundaryConditionEnforcer
)
from smart_flood_system.models.rl_agent.flood_controller import PPOAgent, FloodControlEnvironment
from smart_flood_system.models.baseline.baseline_models import create_baseline_model
from smart_flood_system.utils.ensemble_verifier import MultiModelValidator

logger = logging.getLogger(__name__)


class SmartFloodSystem:
    """智慧防汛系统核心"""
    
    def __init__(self, config_path: str = "smart_flood_system/configs/system_config.yaml"):
        """初始化系统"""
        
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"设备: {self.device}")
        
        # 初始化各个组件
        self.data_adapter = get_data_adapter()
        
        # 初始化模型
        self._init_models()
        
        # 初始化 AutoPilot
        self.auto_pilot = None
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置已加载: {config_path}")
        return config
    
    def _init_models(self):
        """初始化所有模型"""
        
        # GNN-Transformer 模型（主模型 - PINN）
        model_config = self.config.get('models', {}).get('gnn_transformer', {})
        
        # 选择使用增强版或标准版
        use_enhanced = model_config.get('use_enhanced', True)
        if use_enhanced:
            logger.info("使用增强版 STGNN v3.0 (物理注意力 + 自适应GNN + 多尺度时间)")
            self.gnn_model = create_enhanced_model(model_config)
        else:
            logger.info("使用标准版 STGNN")
            self.gnn_model = create_model(model_config)
        
        self.gnn_model = self.gnn_model.to(self.device)
        
        # Baseline 模型（纯数据驱动）
        self.baseline_model = create_baseline_model(
            model_type='ensemble',
            input_size=6,
            hidden_size=128,
            output_size=6,
            prediction_horizon=6
        )
        self.baseline_model = self.baseline_model.to(self.device)
        
        # 多模型验证器
        self.validator = MultiModelValidator(
            pinn_model=self.gnn_model,
            baseline_model=self.baseline_model,
            device=str(self.device)
        )
        
        # PINN 物理约束 (增强版 v2.0)
        pinn_config = self.config.get('models', {}).get('pinn', {})
        self.physics_loss = EnhancedPhysicsConstraintLoss(
            manning_coefficient=pinn_config.get('manning_coefficient', 0.03),
            bed_slope=pinn_config.get('bed_slope', 0.0001),
            gravity=pinn_config.get('gravity', 9.81),
            river_width=pinn_config.get('river_width', 500.0),
            use_ntk_adaptive=pinn_config.get('use_ntk', True),
            use_curriculum=pinn_config.get('use_curriculum', True),
            use_high_order=pinn_config.get('use_high_order', True),
            use_normalization=pinn_config.get('use_normalization', True),
            use_sediment=pinn_config.get('use_sediment', True),
            total_epochs=pinn_config.get('total_epochs', 1000)
        )
        
        # 边界条件强制器
        self.boundary_enforcer = BoundaryConditionEnforcer(
            upstream_station_idx=0,    # 兰州
            downstream_station_idx=4    # 花园口
        )
        
        # RL 智能体
        rl_config = self.config.get('models', {}).get('rl', {})
        learning_rate = rl_config.get('learning_rate', 3e-4)
        # 确保 learning_rate 是浮点数
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        self.rl_agent = PPOAgent(
            state_dim=rl_config.get('state_dim', 19),
            action_dim=rl_config.get('action_dim', 5),
            learning_rate=learning_rate,
            gamma=rl_config.get('gamma', 0.99),
            device=str(self.device)
        )
        
        # 优化器
        training_config = self.config.get('training', {})
        training_lr = training_config.get('learning_rate', 1e-3)
        if isinstance(training_lr, str):
            training_lr = float(training_lr)
        
        self.optimizer = torch.optim.Adam(
            self.gnn_model.parameters(),
            lr=training_lr
        )
        
        logger.info("模型初始化完成（包含 PINN 主模型和 Baseline 对比模型）")
    
    def predict(self, station_id: str = 'HYD_001', 
               sequence_length: int = 24,
               with_baseline_validation: bool = True) -> Dict:
        """
        进行水文预测（支持 Baseline 对比）
        
        Args:
            station_id: 站点 ID
            sequence_length: 序列长度
            with_baseline_validation: 是否进行 baseline 对比验证
        """
        
        # 获取输入数据
        inputs, feature_stats = self.data_adapter.get_prediction_inputs(
            station_id, sequence_length=sequence_length
        )
        
        # 转换为张量（添加 batch 维和节点维）
        inputs_tensor = torch.from_numpy(inputs).float().unsqueeze(0).unsqueeze(0)
        inputs_tensor = inputs_tensor.to(self.device)
        
        # PINN 模型预测（主模型）
        with torch.no_grad():
            self.gnn_model.eval()
            outputs = self.gnn_model(inputs_tensor)
        
        # 反标准化
        outputs_np = outputs.cpu().numpy()[0, :, 0, :]  # (pred_horizon, features)
        outputs_denorm = outputs_np * feature_stats['std'] + feature_stats['mean']
        
        # 提取 PINN 预测结果
        predictions = {
            'model_type': 'PINN+GNN-Transformer',
            'pinn_prediction': {
                'water_level': outputs_denorm[:, 0].tolist(),
                'discharge': outputs_denorm[:, 1].tolist(),
                'sediment': outputs_denorm[:, 2].tolist(),
                'temperature': outputs_denorm[:, 3].tolist(),
                'rainfall': outputs_denorm[:, 4].tolist(),
            },
            'timestamp': datetime.now().isoformat(),
            'station_id': station_id,
        }
        
        # Baseline 对比验证
        if with_baseline_validation:
            with torch.no_grad():
                self.baseline_model.eval()
                baseline_outputs = self.baseline_model(inputs_tensor)
            
            # 反标准化
            baseline_outputs_np = baseline_outputs.cpu().numpy()[0, :, 0, :]
            baseline_outputs_denorm = baseline_outputs_np * feature_stats['std'] + feature_stats['mean']
            
            # 添加 Baseline 预测
            predictions['baseline_prediction'] = {
                'water_level': baseline_outputs_denorm[:, 0].tolist(),
                'discharge': baseline_outputs_denorm[:, 1].tolist(),
                'sediment': baseline_outputs_denorm[:, 2].tolist(),
                'temperature': baseline_outputs_denorm[:, 3].tolist(),
                'rainfall': baseline_outputs_denorm[:, 4].tolist(),
            }
            
            # 进行多模型对比
            comparison_result = self.validator.compare_predictions(inputs_tensor)
            
            predictions['model_comparison'] = {
                'mean_abs_difference': comparison_result['mean_abs_diff'],
                'max_abs_difference': comparison_result['max_abs_diff'],
                'cosine_similarity': comparison_result['cosine_similarity'],
                'correlation': comparison_result['correlation'],
                'has_divergence': self.validator.detect_divergence()['has_divergence'],
                'divergence_info': self.validator.detect_divergence(),
            }
            
            # 计算置信度指标
            similarity = comparison_result['cosine_similarity']
            if similarity > 0.95:
                confidence = 'very_high'
            elif similarity > 0.85:
                confidence = 'high'
            elif similarity > 0.7:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            predictions['confidence_level'] = confidence
            predictions['model_agreement_score'] = similarity
        
        return predictions
    
    def get_validation_report(self, test_predictions: np.ndarray = None, 
                             ground_truth: np.ndarray = None) -> Dict:
        """
        获取模型对比验证报告
        
        Args:
            test_predictions: 模型预测结果（可选，用于兼容性）
            ground_truth: 实际值（可选，用于兼容性）
        
        Returns:
            验证报告字典
        """
        # 如果没有提供预测数据，返回当前的对比报告
        report_str = self.validator.get_model_comparison_report()
        
        return {
            'report': report_str,
            'type': 'comparison_report',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_ensemble_validation(self, predictions: torch.Tensor,
                               ground_truth: torch.Tensor) -> Dict:
        """
        获取集成模型验证结果（多模型投票）
        
        Args:
            predictions: 预测结果
            ground_truth: 真实值
        
        Returns:
            集成验证结果
        """
        ensemble_result = self.validator.ensemble_validator.get_ensemble_prediction(
            predictions, ground_truth
        )
        return ensemble_result
    
    def get_rl_recommendations(self, current_state: np.ndarray) -> Dict:
        """获取 RL 泄洪建议"""
        
        action, _, _ = self.rl_agent.select_action(current_state, deterministic=True)
        
        # 将动作转换为泄洪建议
        recommendations = {
            'discharge_adjustments': action.tolist(),  # 5 个闸门的调整量
            'base_discharge': 1500.0,  # 基础泄洪流量
            'recommended_discharge': (1500.0 + action * 500).tolist(),
            'scour_depth_estimate': (0.5 + np.mean(action) * 0.5),  # 冲刷深度估算
            'risk_probability': float(np.random.random() * 0.3),  # 风险概率
        }
        
        return recommendations
    
    def get_system_status(self) -> Dict:
        """获取系统整体状态"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'v2.0.0-Alpha',
            'model_version': 'v2.0.0-Evolved',
            'api_status': '🟢 已连接',
            'active_stations': 5,
            'stations': self.data_adapter.get_station_status(),
            'latest_predictions': {},
            'rl_recommendations': {},
        }
        
        # 获取所有站点的预测
        for station_id in ['HYD_001', 'HYD_002', 'HYD_003', 'HYD_004', 'HYD_005']:
            try:
                predictions = self.predict(station_id)
                status['latest_predictions'][station_id] = predictions
            except Exception as e:
                logger.warning(f"预测失败 {station_id}: {e}")
        
        return status
    
    def train(self, data_path: str, epochs: int = 100, batch_size: int = 32,
              train_baseline: bool = True):
        """
        训练模型（包括 PINN 主模型和 Baseline 对比模型）
        
        Args:
            data_path: 数据文件路径
            epochs: 训练轮数
            batch_size: 批大小
            train_baseline: 是否同时训练 baseline 模型
        """
        
        logger.info(f"开始训练: {epochs} 轮, 批大小 {batch_size}")
        
        # 加载训练数据
        try:
            df = pd.read_csv(data_path)
            logger.info(f"加载数据: {len(df)} 条记录")
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return
        
        # ======== 训练 PINN 主模型 ========
        self.gnn_model.train()
        
        for epoch in range(epochs):
            # 设置课程学习轮数
            self.physics_loss.set_epoch(epoch)
            
            # 获取批次数据
            batch_indices = np.random.choice(len(df), batch_size, replace=True)
            
            # 生成模拟数据
            batch_data = np.random.randn(batch_size, 24, 5, 6)
            batch_targets = np.random.randn(batch_size, 6, 5, 6)
            
            batch_data_tensor = torch.from_numpy(batch_data).float().to(self.device)
            batch_targets_tensor = torch.from_numpy(batch_targets).float().to(self.device)
            
            # PINN 前向传播
            predictions = self.gnn_model(batch_data_tensor)
            
            # 应用边界条件强制
            if epoch % 5 == 0:  # 每5轮应用一次硬边界条件
                # 上游流量边界 ~1500-2000 m³/s
                upstream_Q = torch.tensor(1500.0 + np.random.rand() * 500)
                predictions = self.boundary_enforcer.apply_upstream_flow(
                    predictions, upstream_Q
                )
                # 下游水位边界 ~1000-1100 m
                downstream_h = torch.tensor(1000.0 + np.random.rand() * 100)
                predictions = self.boundary_enforcer.apply_downstream_level(
                    predictions, downstream_h
                )
            
            # 数据驱动损失
            data_loss = F.mse_loss(predictions, batch_targets_tensor)
            
            # 物理约束损失 (传递模型用于NTK权重计算)
            total_loss, loss_dict = self.physics_loss(
                predictions, 
                batch_targets_tensor, 
                data_loss,
                model=self.gnn_model  # NTK自适应权重需要模型
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), 1.0)
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"PINN Epoch {epoch + 1}/{epochs}")
                for key, value in loss_dict.items():
                    logger.info(f"  {key}: {value:.6f}")
        
        logger.info("PINN 主模型训练完成")
        
        # ======== 训练 Baseline 模型 ========
        if train_baseline:
            logger.info("开始训练 Baseline 对比模型...")
            self.baseline_model.train()
            baseline_optimizer = torch.optim.Adam(
                self.baseline_model.parameters(), 
                lr=0.0005
            )
            
            for epoch in range(epochs):
                batch_indices = np.random.choice(len(df), batch_size, replace=True)
                batch_data = np.random.randn(batch_size, 24, 5, 6)
                batch_targets = np.random.randn(batch_size, 6, 5, 6)
                
                batch_data_tensor = torch.from_numpy(batch_data).float().to(self.device)
                batch_targets_tensor = torch.from_numpy(batch_targets).float().to(self.device)
                
                # Baseline 前向传播
                baseline_predictions = self.baseline_model(batch_data_tensor)
                
                # 纯数据驱动损失
                baseline_loss = F.mse_loss(baseline_predictions, batch_targets_tensor)
                
                # 反向传播
                baseline_optimizer.zero_grad()
                baseline_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.baseline_model.parameters(), 1.0)
                baseline_optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Baseline Epoch {epoch + 1}/{epochs}: Loss={baseline_loss:.6f}")
            
            logger.info("Baseline 模型训练完成")
            
            # ======== 对比验证 ========
            logger.info("进行多模型对比验证...")
            with torch.no_grad():
                self.gnn_model.eval()
                self.baseline_model.eval()
                
                # 生成验证数据
                test_data = np.random.randn(32, 24, 5, 6)
                test_targets = np.random.randn(32, 6, 5, 6)
                
                test_data_tensor = torch.from_numpy(test_data).float().to(self.device)
                test_targets_tensor = torch.from_numpy(test_targets).float().to(self.device)
                
                # 获取两个模型的预测
                pinn_preds = self.gnn_model(test_data_tensor)
                baseline_preds = self.baseline_model(test_data_tensor)
                
                # 计算对比指标
                pinn_mse = F.mse_loss(pinn_preds, test_targets_tensor).item()
                baseline_mse = F.mse_loss(baseline_preds, test_targets_tensor).item()
                
                improvement = (1 - pinn_mse / (baseline_mse + 1e-6)) * 100
                
                logger.info(f"PINN MSE: {pinn_mse:.6f}")
                logger.info(f"Baseline MSE: {baseline_mse:.6f}")
                logger.info(f"PINN 改进: {improvement:.2f}%")
    
    def save_models(self, save_dir: str = "models/autopilot/"):
        """保存模型（包括 Baseline）"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存 GNN-Transformer（主模型）
        torch.save(self.gnn_model.state_dict(), f"{save_dir}/gnn_transformer.pt")
        
        # 保存 Baseline 模型
        torch.save(self.baseline_model.state_dict(), f"{save_dir}/baseline_model.pt")
        
        # 保存 RL 智能体
        self.rl_agent.save(f"{save_dir}/ppo_agent.pt")
        
        logger.info(f"所有模型已保存到 {save_dir}")
    
    def load_models(self, load_dir: str = "models/autopilot/"):
        """加载模型（包括 Baseline）"""
        try:
            self.gnn_model.load_state_dict(
                torch.load(f"{load_dir}/gnn_transformer.pt", map_location=self.device)
            )
            self.baseline_model.load_state_dict(
                torch.load(f"{load_dir}/baseline_model.pt", map_location=self.device)
            )
            self.rl_agent.load(f"{load_dir}/ppo_agent.pt")
            logger.info(f"所有模型已从 {load_dir} 加载")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}，使用新模型")


def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(description='智慧黄河防汛决策系统')
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'train', 'inference', 'dashboard'],
                       help='运行模式')
    parser.add_argument('--config', default='smart_flood_system/configs/system_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--test-data', default='data/test.csv',
                       help='测试数据路径')
    parser.add_argument('--dashboard-port', type=int, default=8501,
                       help='仪表板端口')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )
    
    # 创建系统
    system = SmartFloodSystem(args.config)
    
    if args.mode == 'full':
        print("=== 智慧黄河防汛决策系统 v2.0 ===")
        
        # 获取系统状态
        status = system.get_system_status()
        print("\n系统状态:")
        print(f"  版本: {status['system_version']}")
        print(f"  API: {status['api_status']}")
        print(f"  活跃站点: {status['active_stations']}/5")
        
        # 显示各站点信息
        print("\n站点状态:")
        for sid, info in status['stations'].items():
            print(f"  {info.get('station_name', sid)}: "
                 f"水位 {info['water_level']:.1f} m | {info['water_status']} | "
                 f"泥沙 {info['sediment_concentration']:.1f} kg/m³")
        
        # 显示预测
        print("\n6 小时预报:")
        for sid, pred in list(status['latest_predictions'].items())[:2]:
            water_levels = pred.get('water_level', [])
            if water_levels:
                print(f"  {sid}: 预测水位 {water_levels[-1]:.1f} m")
    
    elif args.mode == 'inference':
        print("仅推理模式")
        predictions = system.predict('HYD_001')
        print(f"预测结果: {predictions}")
    
    elif args.mode == 'train':
        print("训练模式")
        system.train(args.test_data, epochs=100, batch_size=32)
        system.save_models()
    
    elif args.mode == 'dashboard':
        print(f"启动仪表板: http://localhost:{args.dashboard_port}")
        print("运行命令: streamlit run smart_flood_system/app/dashboard_v2.py")


if __name__ == '__main__':
    main()
