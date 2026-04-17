"""
AutoPilot Monitor - Simple Status Dashboard
===========================================

Run this to see current system status:
    python smart_flood_system/monitor.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

def show_status():
    """Display current AutoPilot status."""
    
    print("=" * 70)
    print("🤖  AutoPilot - 实时状态监控")
    print("=" * 70)
    
    # Check logs
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = sorted(log_dir.glob("auto_pilot_*.log"))
        if log_files:
            latest_log = log_files[-1]
            print(f"\n📄 最新日志: {latest_log.name}")
            
            # Show last 10 lines
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                print("\n📝 最近日志 (最后10行):")
                for line in lines[-10:]:
                    print(f"   {line.strip()[:100]}")
    
    # Check reports
    report_dir = Path("reports/evolution")
    if report_dir.exists():
        reports = list(report_dir.glob("*.md"))
        print(f"\n📊 报告文件: {len(reports)} 个")
        
        latest = report_dir / "evolution_report_latest.md"
        if latest.exists():
            print(f"   ✓ 最新报告已生成")
    
    # Check models
    model_dir = Path("models/autopilot")
    if model_dir.exists():
        models = list(model_dir.glob("*.pth"))
        print(f"\n🧠 模型文件: {len(models)} 个")
        
        best = model_dir / "best_model.pth"
        if best.exists():
            size_mb = best.stat().st_size / (1024 * 1024)
            print(f"   ✓ Best Model: {size_mb:.1f} MB")
    
    # System time
    print(f"\n⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Next scheduled events
    print("\n📅 即将执行的任务:")
    print("   • 下次数据下载: 明天 02:00")
    print("   • 下次训练检查: 6小时后")
    print("   • 下次健康检查: 5分钟后")
    print("   • 下次月度回溯: 5月1日 03:00")
    
    print("\n" + "=" * 70)
    print("💡 提示: 按 Ctrl+C 退出监控，系统将在后台继续运行")
    print("=" * 70)

if __name__ == "__main__":
    show_status()
