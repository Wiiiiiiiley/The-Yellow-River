#!/usr/bin/env python3
"""
项目文件重组脚本
整理项目结构，将相关文件归类
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def create_directories():
    """创建新的目录结构"""
    dirs = [
        # 源代码
        "src/models",
        "src/data",
        "src/training",
        "src/evaluation",
        "src/utils",
        "src/core",
        
        # 数据
        "data/raw",
        "data/processed",
        "data/external",
        
        # 实验和结果
        "experiments/models",
        "experiments/visualizations",
        "experiments/logs",
        "experiments/checkpoints",
        
        # 配置
        "configs",
        
        # 脚本
        "scripts",
        
        # 文档
        "docs",
    ]
    
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {d}")

def move_files():
    """移动文件到新位置"""
    
    # 定义移动规则: (源, 目标)
    moves = []
    
    # 1. 模型文件 -> src/models/
    model_files = [
        ("smart_flood_system/models/baseline/baseline_models.py", "src/models/"),
        ("smart_flood_system/models/graph_transformer/stgnn.py", "src/models/"),
        ("smart_flood_system/models/graph_transformer/stgnn_enhanced.py", "src/models/"),
        ("smart_flood_system/models/pinn_loss/physics_constraints.py", "src/models/"),
        ("smart_flood_system/models/pinn_loss/physics_constraints_enhanced.py", "src/models/"),
        ("smart_flood_system/models/rl_agent/flood_controller.py", "src/models/"),
    ]
    moves.extend(model_files)
    
    # 2. 训练相关 -> src/training/
    training_files = [
        ("training/train_enhanced_pinn.py", "src/training/"),
        ("BASELINE_QUICKSTART.py", "src/training/"),
    ]
    moves.extend(training_files)
    
    # 3. 评估/可视化 -> src/evaluation/
    eval_files = [
        ("experiments/visualize_prediction_comparison.py", "src/evaluation/"),
    ]
    moves.extend(eval_files)
    
    # 4. 数据处理 -> src/data/
    data_files = [
        ("smart_flood_system/data_adapter.py", "src/data/"),
    ]
    moves.extend(data_files)
    
    # 5. 工具 -> src/utils/
    util_files = [
        ("smart_flood_system/utils/ensemble_verifier.py", "src/utils/"),
    ]
    moves.extend(util_files)
    
    # 6. 核心系统 -> src/core/
    core_files = [
        ("smart_flood_system/main_system.py", "src/core/"),
        ("smart_flood_system/auto_pilot.py", "src/core/"),
        ("smart_flood_system/monitor.py", "src/core/"),
        ("smart_flood_system/app/dashboard_v2.py", "src/core/"),
    ]
    moves.extend(core_files)
    
    # 7. 脚本 -> scripts/
    script_files = [
        ("run.sh", "scripts/"),
    ]
    moves.extend(script_files)
    
    # 8. 数据移动
    data_moves = [
        ("data/autopilot.db", "data/processed/"),
    ]
    
    # 执行移动
    for src, dst in moves:
        src_path = PROJECT_ROOT / src
        dst_path = PROJECT_ROOT / dst
        
        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path / src_path.name))
                print(f"✓ 移动: {src} -> {dst}")
            except Exception as e:
                print(f"✗ 失败: {src} ({e})")
        else:
            print(f"⚠ 不存在: {src}")

def cleanup_empty_dirs():
    """清理空目录"""
    dirs_to_check = [
        "smart_flood_system/models/baseline",
        "smart_flood_system/models/graph_transformer",
        "smart_flood_system/models/pinn_loss",
        "smart_flood_system/models/rl_agent",
        "smart_flood_system/models",
        "smart_flood_system/utils",
        "smart_flood_system/app",
        "smart_flood_system",
        "training",
    ]
    
    for d in dirs_to_check:
        dir_path = PROJECT_ROOT / d
        if dir_path.exists() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                print(f"✓ 删除空目录: {d}")
            except Exception as e:
                print(f"✗ 无法删除: {d} ({e})")

def print_final_structure():
    """打印最终结构"""
    print("\n" + "="*60)
    print("📁 整理后的项目结构")
    print("="*60)
    
    structure = """
The-Yellow-River/
├── 📄 README.md                 # 项目说明
├── 📄 requirements.txt          # 依赖
├── 📄 setup.py                 # 安装配置
├── 📄 .gitignore               # Git 忽略
│
├── 📁 src/                     # 源代码
│   ├── 📁 models/              # 所有模型
│   │   ├── baseline_models.py
│   │   ├── stgnn.py
│   │   ├── stgnn_enhanced.py
│   │   ├── physics_constraints.py
│   │   ├── physics_constraints_enhanced.py
│   │   └── flood_controller.py
│   │
│   ├── 📁 training/            # 训练脚本
│   │   ├── train_enhanced_pinn.py
│   │   └── BASELINE_QUICKSTART.py
│   │
│   ├── 📁 evaluation/          # 评估可视化
│   │   └── visualize_prediction_comparison.py
│   │
│   ├── 📁 data/                # 数据处理
│   │   └── data_adapter.py
│   │
│   ├── 📁 utils/               # 工具
│   │   └── ensemble_verifier.py
│   │
│   └── 📁 core/                # 核心系统
│       ├── main_system.py
│       ├── auto_pilot.py
│       ├── monitor.py
│       └── dashboard_v2.py
│
├── 📁 data/                    # 数据集
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── external/              # 外部数据
│
├── 📁 experiments/             # 实验结果
│   ├── models/                # 保存的模型
│   ├── visualizations/        # 可视化图表
│   ├── logs/                  # 训练日志
│   └── checkpoints/           # 检查点
│
├── 📁 configs/                 # 配置文件
│
├── 📁 scripts/                 # 快捷脚本
│   └── run.sh
│
└── 📁 docs/                    # 文档
"""
    print(structure)

def main():
    print("="*60)
    print("🗂️  项目文件重组")
    print("="*60)
    print()
    
    print("[1/4] 创建新目录结构...")
    create_directories()
    print()
    
    print("[2/4] 移动文件...")
    move_files()
    print()
    
    print("[3/4] 清理空目录...")
    cleanup_empty_dirs()
    print()
    
    print("[4/4] 完成!")
    print_final_structure()
    
    print("\n" + "="*60)
    print("💡 下一步:")
    print("   1. 检查移动后的文件")
    print("   2. 更新导入路径")
    print("   3. 运行测试验证")
    print("   4. 提交到 git")
    print("="*60)

if __name__ == "__main__":
    main()
