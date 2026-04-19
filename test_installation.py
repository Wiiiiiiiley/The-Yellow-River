"""
测试脚本 - 验证系统安装
"""

import sys
from pathlib import Path

def test_imports():
    """测试导入"""
    print("测试模块导入...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch 未安装: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas 未安装: {e}")
        return False
    
    try:
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"  ✗ Streamlit 未安装: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ PyYAML 已安装")
    except ImportError as e:
        print(f"  ✗ PyYAML 未安装: {e}")
        return False
    
    return True


def test_project_structure():
    """测试项目结构"""
    print("\n检查项目结构...")
    
    required_files = [
        "smart_flood_system/main_system.py",
        "smart_flood_system/auto_pilot.py",
        "smart_flood_system/data_adapter.py",
        "smart_flood_system/monitor.py",
        "smart_flood_system/models/graph_transformer/stgnn.py",
        "smart_flood_system/models/pinn_loss/physics_constraints.py",
        "smart_flood_system/models/rl_agent/flood_controller.py",
        "smart_flood_system/app/dashboard_v2.py",
        "smart_flood_system/configs/system_config.yaml",
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ 缺失: {file}")
            all_exist = False
    
    return all_exist


def test_system_initialization():
    """测试系统初始化"""
    print("\n初始化系统...")
    
    try:
        from smart_flood_system.data_adapter import get_data_adapter
        adapter = get_data_adapter()
        print("  ✓ 数据适配器初始化成功")
        
        # 测试获取数据
        status = adapter.get_station_status()
        print(f"  ✓ 可用站点数: {len(status)}")
        
        return True
    except Exception as e:
        print(f"  ✗ 初始化失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n创建模型...")
    
    try:
        from smart_flood_system.models.graph_transformer.stgnn import create_model
        
        config = {
            'node_features': 6,
            'n_nodes': 5,
            'gnn_hidden': 64,
            'gnn_layers': 3,
            'transformer_heads': 8,
            'sequence_length': 24,
            'prediction_horizon': 6,
        }
        
        model = create_model(config)
        print(f"  ✓ GNN-Transformer 模型创建成功")
        print(f"    参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"  ✗ 模型创建失败: {e}")
        return False


def main():
    """主测试程序"""
    print("=" * 60)
    print("  🌊 智慧黄河防汛决策系统 - 安装验证")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("项目结构", test_project_structure()))
    results.append(("系统初始化", test_system_initialization()))
    results.append(("模型创建", test_model_creation()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 所有测试通过！系统已准备就绪。")
        print("\n快速开始:")
        print("  1. 启动仪表板:")
        print("     streamlit run smart_flood_system/app/dashboard_v2.py")
        print("  2. 启动 AutoPilot:")
        print("     python smart_flood_system/auto_pilot.py --mode full")
        print("  3. CLI 监控器:")
        print("     python smart_flood_system/monitor.py")
        return 0
    else:
        print("\n✗ 某些测试失败，请检查安装。")
        print("   运行: pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
