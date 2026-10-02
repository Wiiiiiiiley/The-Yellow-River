"""
智慧黄河防汛决策系统主模块
集成所有组件：GNN-Transformer + PINN + RL + AutoPilot
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from src.data.data_adapter import get_data_adapter
from src.models.stgnn_enhanced import create_enhanced_model
from src.models.physics_constraints_enhanced import (
    EnhancedPhysicsConstraintLoss, BoundaryConditionEnforcer
)
from src.models.baseline_models import create_baseline_model
from src.utils.ensemble_verifier import MultiModelValidator, EnsembleValidator

logger = logging.getLogger(__name__)


class SmartFloodSystem:
    """智慧防汛系统核心"""
    
    def __init__(self, config_path: str = "configs/system_config.yaml"):
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
        
        logger.info("使用增强版 STGNN v3.0 (物理注意力 + 自适应GNN + 多尺度时间)")
        self.gnn_model = create_enhanced_model(model_config)
        
        self.gnn_model = self.gnn_model.to(self.device)
        
        # Baseline 模型（纯数据驱动）
        self.baseline_model = create_baseline_model(
            model_type='lstm',
            input_size=6,
            hidden_size=128,
            output_size=6,
            prediction_horizon=6,
            n_nodes=model_config.get('n_nodes', 5),
        )
        self.baseline_model = self.baseline_model.to(self.device)
        
        # 多模型验证器
        self.validator = MultiModelValidator(
            pinn_model=self.gnn_model,
            baseline_model=self.baseline_model,
            device=str(self.device)
        )
        self.ensemble_validator = EnsembleValidator(
            models=[self.gnn_model, self.baseline_model],
            device=str(self.device),
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
        
        # 仓库已不包含原 RL 实现，显式标记为不可用，避免伪造决策建议。
        self.rl_agent = None
        
        # 优化器
        training_config = self.config.get('training', {})
        training_lr = training_config.get('learning_rate', 1e-3)
        if isinstance(training_lr, str):
            training_lr = float(training_lr)
        
        self.optimizer = torch.optim.Adam(
            self.gnn_model.parameters(),
            lr=training_lr
        )

        self.model_ready = False
        self.feature_mean = None
        self.feature_std = None
        self._load_experiment_checkpoints()
        
        logger.info("模型初始化完成（包含 PINN 主模型和 Baseline 对比模型）")

    def _load_experiment_checkpoints(self):
        """只加载使用当前公平对比协议训练的成对模型。"""
        pinn_path = Path("experiments/models/pinn_best.pth")
        baseline_path = Path("experiments/models/baseline_best.pth")
        if not pinn_path.exists() or not baseline_path.exists():
            logger.warning("未找到成对的 PINN/Baseline 检查点，预测功能暂不可用")
            return

        pinn_checkpoint = torch.load(pinn_path, map_location=self.device)
        baseline_checkpoint = torch.load(baseline_path, map_location=self.device)
        if (pinn_checkpoint.get('protocol_version', 1) < 2 or
                baseline_checkpoint.get('protocol_version', 1) < 2):
            logger.warning("检测到旧评估协议检查点，请重新运行 scripts/train_both_models.py")
            return

        mean = pinn_checkpoint.get('mean', pinn_checkpoint.get('data_mean'))
        std = pinn_checkpoint.get('std', pinn_checkpoint.get('data_std'))
        if mean is None or std is None:
            logger.warning("PINN 检查点缺少训练集 mean/std，预测功能暂不可用")
            return

        self.gnn_model.load_state_dict(pinn_checkpoint['model_state_dict'])
        self.baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        self.feature_mean = mean.squeeze().view(1, 1, 1, -1).to(self.device)
        self.feature_std = std.squeeze().view(1, 1, 1, -1).to(self.device)
        self.model_ready = True
        logger.info("已加载 protocol v2 PINN/Baseline 检查点")
    
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

        if not self.model_ready:
            raise RuntimeError("没有可用的 protocol v2 模型，请先运行 scripts/train_both_models.py")
        
        station_ids = list(self.data_adapter.STATIONS.keys())
        if station_id not in station_ids:
            raise ValueError(f"未知站点: {station_id}")

        # STGNN 必须同时接收整条河道的 5 个节点，不能把单站时间维
        # 误当成节点维。逐站取得序列后组装为 (time, nodes, features)。
        node_inputs = []
        for current_station_id in station_ids:
            station_inputs, station_stats = self.data_adapter.get_prediction_inputs(
                current_station_id, sequence_length=sequence_length
            )
            # DataAdapter 返回单站归一化值；先恢复原始水文量，再使用
            # 训练集全局统计量归一化，保证推理与训练一致。
            raw_inputs = station_inputs.copy()
            raw_inputs[:, :5] = (
                station_inputs[:, :5] * station_stats['std'] + station_stats['mean']
            )
            node_inputs.append(raw_inputs)

        network_inputs = np.stack(node_inputs, axis=1)
        inputs_tensor = torch.from_numpy(network_inputs).float().unsqueeze(0).to(self.device)
        inputs_tensor = (inputs_tensor - self.feature_mean) / self.feature_std
        station_index = station_ids.index(station_id)
        
        # PINN 模型预测（主模型）
        with torch.no_grad():
            self.gnn_model.eval()
            outputs = self.gnn_model(inputs_tensor)
        
        # 反标准化
        outputs = outputs * self.feature_std + self.feature_mean
        outputs_denorm = outputs.cpu().numpy()[0, :, station_index, :5]
        
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
            baseline_outputs = baseline_outputs * self.feature_std + self.feature_mean
            baseline_outputs_denorm = baseline_outputs.cpu().numpy()[0, :, station_index, :5]
            
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
    
    def get_ensemble_validation(self, inputs: torch.Tensor,
                               ground_truth: Optional[torch.Tensor] = None) -> Dict:
        """
        获取集成模型验证结果（多模型投票）
        
        Args:
            inputs: 模型输入
            ground_truth: 真实值（可选）
        
        Returns:
            集成验证结果
        """
        inputs = inputs.to(self.device)
        ensemble_prediction, uncertainty, details = self.ensemble_validator.ensemble_predict(inputs)
        result = {
            'prediction': ensemble_prediction,
            'uncertainty': uncertainty,
            'details': details,
        }
        if ground_truth is not None:
            result['mse'] = F.mse_loss(
                ensemble_prediction, ground_truth.to(self.device)
            ).item()
        return result
    
    def get_rl_recommendations(self, current_state: np.ndarray) -> Dict:
        """获取 RL 泄洪建议"""

        return {
            'available': False,
            'reason': '当前仓库不包含已训练的 RL 泄洪控制器，未生成决策建议。',
        }
    
    def get_system_status(self) -> Dict:
        """获取系统整体状态"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'v3.0.0',
            'model_version': 'v3.0.0',
            'api_status': '🟡 本地/模拟数据',
            'active_stations': 5,
            'model_ready': self.model_ready,
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

        # 统一走已修正的公平对比训练入口，避免旧集成代码
        # 读取 CSV 后却用随机张量训练。
        from scripts.train_both_models import main as train_both_models

        if data_path and Path(data_path).exists():
            logger.warning(
                "当前对比训练器使用物理合成数据；未将 %s 冒充为训练输入",
                data_path,
            )
        if not train_baseline:
            logger.warning("公平对比协议要求同时训练 Baseline，已忽略 train_baseline=False")
        if batch_size != 32:
            logger.warning("当前统一训练器使用内置批大小，已忽略 batch_size=%s", batch_size)

        train_both_models(epochs=epochs)
        self._load_experiment_checkpoints()
    
    def save_models(self, save_dir: str = "models/autopilot/"):
        """保存模型（包括 Baseline）"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存 GNN-Transformer（主模型）
        torch.save(self.gnn_model.state_dict(), f"{save_dir}/gnn_transformer.pt")
        
        # 保存 Baseline 模型
        torch.save(self.baseline_model.state_dict(), f"{save_dir}/baseline_model.pt")
        
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
            logger.info(f"所有模型已从 {load_dir} 加载")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}，使用新模型")


def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(description='智慧黄河防汛决策系统')
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'train', 'inference', 'dashboard'],
                       help='运行模式')
    parser.add_argument('--config', default='configs/system_config.yaml',
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
        print("运行命令: streamlit run src/core/dashboard_v2.py")


if __name__ == '__main__':
    main()
