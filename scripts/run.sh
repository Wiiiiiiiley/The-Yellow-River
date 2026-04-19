#!/bin/bash
# 启动脚本 - 黄河防汛系统

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  🌊 智慧黄河防汛决策系统 v2.0 启动脚本"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在,请先运行: python -m venv .venv"
    exit 1
fi

# 激活虚拟环境
echo "✓ 激活虚拟环境..."
source .venv/bin/activate

# 确保依赖已安装
echo "✓ 检查依赖..."
pip install -q -r requirements.txt

# 创建必要的目录
mkdir -p data/archive
mkdir -p models/autopilot
mkdir -p results/predictions
mkdir -p results/reports
mkdir -p logs

echo ""
echo "选择启动模式:"
echo ""
echo "1. 完整系统（推理 + 仪表板）"
echo "2. 仪表板 v2（推荐）"
echo "3. CLI 监控器"
echo "4. AutoPilot 自主系统"
echo "5. 训练模式"
echo ""

read -p "请选择 (1-5) [默认 2]: " choice
choice=${choice:-2}

case $choice in
    1)
        echo ""
        echo "启动完整系统..."
        python smart_flood_system/main_system.py --mode full
        ;;
    2)
        echo ""
        echo "启动仪表板 v2..."
        echo "打开浏览器访问: http://localhost:8501"
        echo ""
        streamlit run smart_flood_system/app/dashboard_v2.py --server.port 8501
        ;;
    3)
        echo ""
        echo "启动 CLI 监控器..."
        python smart_flood_system/monitor.py
        ;;
    4)
        echo ""
        echo "启动 AutoPilot 自主系统..."
        python smart_flood_system/auto_pilot.py --mode full
        ;;
    5)
        echo ""
        echo "启动训练模式..."
        python smart_flood_system/main_system.py --mode train
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
