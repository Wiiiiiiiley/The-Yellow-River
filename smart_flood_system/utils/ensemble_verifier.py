"""
多模型验证和对比模块
验证 PINN 模型与 Baseline 模型的预测差异
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MultiModelValidator:
    """多模型验证器 - 对比不同模型的预测"""
    
    def __init__(self, 
                 pinn_model: torch.nn.Module,
                 baseline_model: torch.nn.Module,
                 device: str = 'cpu'):
        """
        Args:
            pinn_model: PINN 模型（带物理约束）
            baseline_model: Baseline 模型（纯数据驱动）
            device: 设备
        """
        self.pinn_model = pinn_model.to(device)
        self.baseline_model = baseline_model.to(device)
        self.device = device
        
        self.pinn_model.eval()
        self.baseline_model.eval()
        
        # 验证历史记录
        self.validation_history = []
    
    def compare_predictions(self, x: torch.Tensor, 
                           y_true: Optional[torch.Tensor] = None) -> Dict:
        """
        对比两个模型的预测结果
        
        Args:
            x: 输入数据 (batch_size, seq_len, n_nodes, features)
            y_true: 真实标签（可选）
        
        Returns:
            comparison_result: 对比结果字典
        """
        
        with torch.no_grad():
            # PINN 预测
            pinn_pred = self.pinn_model(x)
            
            # Baseline 预测
            baseline_pred = self.baseline_model(x)
        
        # 计算基本统计
        result = {
            'timestamp': datetime.now().isoformat(),
            'pinn_pred': pinn_pred.cpu().numpy(),
            'baseline_pred': baseline_pred.cpu().numpy(),
            'pinn_mean': pinn_pred.mean().item(),
            'baseline_mean': baseline_pred.mean().item(),
            'pinn_std': pinn_pred.std().item(),
            'baseline_std': baseline_pred.std().item(),
        }
        
        # 计算预测差异
        diff = pinn_pred - baseline_pred
        result['prediction_diff'] = diff.cpu().numpy()
        result['mean_abs_diff'] = torch.abs(diff).mean().item()
        result['max_abs_diff'] = torch.abs(diff).max().item()
        
        # 计算相似度
        result['cosine_similarity'] = self._compute_cosine_similarity(pinn_pred, baseline_pred).item()
        result['correlation'] = self._compute_correlation(pinn_pred, baseline_pred)
        
        # 如果有真实标签，计算误差
        if y_true is not None:
            y_true = y_true.to(self.device)
            
            pinn_mse = F.mse_loss(pinn_pred, y_true).item()
            baseline_mse = F.mse_loss(baseline_pred, y_true).item()
            
            result['pinn_mse'] = pinn_mse
            result['baseline_mse'] = baseline_mse
            result['mse_improvement'] = (baseline_mse - pinn_mse) / (baseline_mse + 1e-8)
            
            pinn_mae = F.l1_loss(pinn_pred, y_true).item()
            baseline_mae = F.l1_loss(baseline_pred, y_true).item()
            
            result['pinn_mae'] = pinn_mae
            result['baseline_mae'] = baseline_mae
            result['mae_improvement'] = (baseline_mae - pinn_mae) / (baseline_mae + 1e-8)
        
        # 存储记录
        self.validation_history.append(result)
        
        return result
    
    def _compute_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度"""
        # 展开为 1D
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        
        # 余弦相似度
        similarity = F.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0))
        return similarity
    
    def _compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """计算皮尔逊相关系数"""
        x_flat = x.reshape(-1).cpu().numpy()
        y_flat = y.reshape(-1).cpu().numpy()
        
        correlation = np.corrcoef(x_flat, y_flat)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def detect_divergence(self, threshold: float = 0.15) -> Dict:
        """
        检测两个模型是否有显著分歧
        
        Args:
            threshold: 分歧阈值（相对差异）
        
        Returns:
            divergence_info: 分歧信息
        """
        
        if not self.validation_history:
            return {'status': 'no_data'}
        
        # 获取最近的验证结果
        results = self.validation_history[-10:]  # 最近 10 次
        
        divergences = []
        for result in results:
            mean_diff = result['mean_abs_diff']
            pinn_mean = abs(result['pinn_mean']) + 1e-8
            relative_diff = mean_diff / pinn_mean
            
            if relative_diff > threshold:
                divergences.append({
                    'time': result['timestamp'],
                    'relative_diff': relative_diff,
                    'abs_diff': mean_diff,
                    'reason': self._analyze_divergence_reason(result)
                })
        
        return {
            'has_divergence': len(divergences) > 0,
            'divergence_count': len(divergences),
            'threshold': threshold,
            'recent_divergences': divergences,
            'average_correlation': np.mean([r['correlation'] for r in results]),
        }
    
    def _analyze_divergence_reason(self, result: Dict) -> str:
        """分析分歧原因"""
        cosine_sim = result['cosine_similarity']
        
        if cosine_sim < 0.5:
            return "方向差异显著 - 模型预测方向相反"
        elif result['max_abs_diff'] > result['pinn_std'] * 2:
            return "幅值差异显著 - 某个预测值偏差过大"
        elif 'pinn_mae' in result and 'baseline_mae' in result:
            if result['baseline_mse'] > result['pinn_mse']:
                return "PINN 模型性能更优，baseline 可能过度拟合或欠拟合"
            else:
                return "Baseline 模型性能更优，PINN 物理约束可能过强"
        else:
            return "分歧原因未知"
    
    def get_model_comparison_report(self) -> str:
        """生成模型对比报告"""
        
        if not self.validation_history:
            return "未进行任何验证"
        
        # 统计数据
        results = self.validation_history
        
        pinn_mses = [r.get('pinn_mse', 0) for r in results if 'pinn_mse' in r]
        baseline_mses = [r.get('baseline_mse', 0) for r in results if 'baseline_mse' in r]
        correlations = [r.get('correlation', 0) for r in results]
        improvements = [r.get('mse_improvement', 0) for r in results if 'mse_improvement' in r]
        
        report = f"""
{'='*60}
             多模型验证对比报告
{'='*60}

验证统计:
  - 总验证次数: {len(results)}
  - 包含真实标签的验证: {len(pinn_mses)}

PINN 模型性能:
  - 平均 MSE: {np.mean(pinn_mses):.6f}
  - 标准差: {np.std(pinn_mses):.6f}
  - 最小 MSE: {np.min(pinn_mses) if pinn_mses else 'N/A'}
  - 最大 MSE: {np.max(pinn_mses) if pinn_mses else 'N/A'}

Baseline 模型性能:
  - 平均 MSE: {np.mean(baseline_mses):.6f}
  - 标准差: {np.std(baseline_mses):.6f}
  - 最小 MSE: {np.min(baseline_mses) if baseline_mses else 'N/A'}
  - 最大 MSE: {np.max(baseline_mses) if baseline_mses else 'N/A'}

对比结果:
  - 平均性能改进: {(np.mean(improvements) * 100) if improvements else 'N/A'}%
  - 平均相关系数: {np.mean(correlations):.4f}
  - 预测分歧检测: {self.detect_divergence()}

模型差异特征:
  - 方向一致性 (余弦相似度): {np.mean([r['cosine_similarity'] for r in results]):.4f}
  - 幅值差异 (平均绝对差): {np.mean([r['mean_abs_diff'] for r in results]):.6f}
  - 最大单点差异: {np.max([r['max_abs_diff'] for r in results]):.6f}

建议:
  {self._generate_recommendations(results)}

{'='*60}
"""
        return report
    
    def _generate_recommendations(self, results: List[Dict]) -> str:
        """根据对比结果生成建议"""
        
        avg_improvement = np.mean([r.get('mse_improvement', 0) for r in results if 'mse_improvement' in r])
        avg_correlation = np.mean([r.get('correlation', 0) for r in results])
        divergence_info = self.detect_divergence()
        
        recommendations = []
        
        if avg_improvement > 0.1:
            recommendations.append("✓ PINN 模型性能优异，物理约束有效")
        elif avg_improvement < -0.1:
            recommendations.append("⚠ Baseline 模型表现更好，考虑调整物理权重")
        else:
            recommendations.append("◆ 两个模型性能接近，可考虑集成")
        
        if avg_correlation > 0.9:
            recommendations.append("✓ 模型预测高度一致")
        elif avg_correlation < 0.7:
            recommendations.append("⚠ 模型预测差异显著，需要分析原因")
        
        if divergence_info['has_divergence']:
            recommendations.append(f"⚠ 检测到 {divergence_info['divergence_count']} 次显著分歧")
        
        return "\n  ".join(recommendations) if recommendations else "模型表现良好"


class EnsembleValidator:
    """集成验证器 - 使用多个模型进行投票"""
    
    def __init__(self, models: List[torch.nn.Module], 
                 weights: Optional[List[float]] = None,
                 device: str = 'cpu'):
        """
        Args:
            models: 模型列表
            weights: 模型权重
            device: 设备
        """
        self.models = [m.to(device) for m in models]
        self.device = device
        
        # 标准化权重
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
        
        for m in self.models:
            m.eval()
    
    def ensemble_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        集成预测
        
        Returns:
            ensemble_pred: 集成预测
            uncertainty: 预测不确定性（标准差）
            details: 详细信息
        """
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        # 堆叠预测
        predictions_stack = torch.stack(predictions)  # (n_models, batch_size, horizon, features)
        
        # 加权平均
        weighted_preds = [self.weights[i] * predictions_stack[i] for i in range(len(self.models))]
        ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        
        # 计算不确定性
        uncertainty = torch.std(predictions_stack, dim=0)
        
        # 计算置信度
        confidence = 1.0 / (1.0 + uncertainty.mean())
        
        details = {
            'n_models': len(self.models),
            'weights': self.weights,
            'uncertainty': uncertainty.cpu().numpy(),
            'mean_uncertainty': uncertainty.mean().item(),
            'confidence': confidence.item(),
            'individual_predictions': [p.cpu().numpy() for p in predictions],
        }
        
        return ensemble_pred, uncertainty, details
    
    def get_divergence_scores(self, x: torch.Tensor) -> Dict:
        """获取模型间的分歧分数"""
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions_stack = torch.stack(predictions)
        
        # 计算方差（分歧度）
        variance = torch.var(predictions_stack, dim=0)
        
        # 计算相关系数矩阵
        n_models = len(self.models)
        correlations = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i, n_models):
                pred_i = predictions[i].reshape(-1).cpu().numpy()
                pred_j = predictions[j].reshape(-1).cpu().numpy()
                corr = np.corrcoef(pred_i, pred_j)[0, 1]
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        return {
            'variance': variance.cpu().numpy(),
            'mean_variance': variance.mean().item(),
            'correlations': correlations,
            'average_correlation': correlations[np.triu_indices_from(correlations, k=1)].mean(),
            'min_correlation': correlations[np.triu_indices_from(correlations, k=1)].min(),
            'max_correlation': correlations[np.triu_indices_from(correlations, k=1)].max(),
        }


if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型（模拟）
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(6, 6)
        
        def forward(self, x):
            # 简化：直接输出
            batch_size, seq_len, n_nodes, features = x.shape
            return torch.randn(batch_size, 6, n_nodes, features)
    
    pinn_model = DummyModel()
    baseline_model = DummyModel()
    
    # 创建验证器
    validator = MultiModelValidator(pinn_model, baseline_model)
    
    # 测试数据
    x = torch.randn(2, 24, 5, 6)
    y_true = torch.randn(2, 6, 5, 6)
    
    # 对比预测
    result = validator.compare_predictions(x, y_true)
    print("对比结果:")
    for key, value in result.items():
        if not isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
    
    # 生成报告
    print("\n" + validator.get_model_comparison_report())
