"""
CLI 监控器 - 终端状态查看器
用于无界面操作环境查看 AutoPilot 系统状态
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import logging

# 简单的颜色输出
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def clear_screen():
    """清空屏幕"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("  🌊 智慧黄河防汛决策系统 v2.0 - CLI 监控器")
    print("=" * 60)
    print(f"{Colors.END}\n")


def print_section(title: str):
    """打印章节标题"""
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 60}{Colors.END}")


def print_status_item(label: str, value: str, status: str = "ok"):
    """打印状态项"""
    if status == "error":
        color = Colors.RED
    elif status == "warning":
        color = Colors.YELLOW
    else:
        color = Colors.GREEN
    
    print(f"  {label:<30} {color}{value}{Colors.END}")


def get_system_status() -> dict:
    """获取系统状态"""
    try:
        from smart_flood_system.main_system import SmartFloodSystem
        system = SmartFloodSystem()
        status = system.get_system_status()
        return status
    except Exception as e:
        return {'error': str(e)}


def get_autopilot_status() -> dict:
    """获取 AutoPilot 状态"""
    try:
        # 检查日志文件
        log_path = Path("logs/autopilot.log")
        evolution_history = Path("results/evolution_history.json")
        
        status = {
            'log_exists': log_path.exists(),
            'evolution_history_exists': evolution_history.exists(),
            'last_log_time': None,
            'evolution_count': 0,
        }
        
        # 读取最后更新时间
        if log_path.exists():
            status['last_log_time'] = datetime.fromtimestamp(log_path.stat().st_mtime)
        
        # 读取进化历史
        if evolution_history.exists():
            with open(evolution_history, 'r') as f:
                history = json.load(f)
                status['evolution_count'] = len(history)
        
        return status
    except Exception as e:
        return {'error': str(e)}


def get_model_status() -> dict:
    """获取模型状态"""
    try:
        model_dir = Path("models/autopilot")
        status = {
            'available_models': [],
            'latest_model': None,
        }
        
        if model_dir.exists():
            for model_file in model_dir.glob("*.pt"):
                status['available_models'].append(model_file.name)
                # 获取最新模型
                if status['latest_model'] is None or model_file.stat().st_mtime > status['latest_model']['mtime']:
                    status['latest_model'] = {
                        'name': model_file.name,
                        'mtime': model_file.stat().st_mtime,
                        'size': model_file.stat().st_size / (1024 * 1024),  # MB
                    }
        
        return status
    except Exception as e:
        return {'error': str(e)}


def get_data_status() -> dict:
    """获取数据状态"""
    try:
        from smart_flood_system.data_adapter import get_data_adapter
        adapter = get_data_adapter()
        
        # 获取最新读数
        latest = adapter.get_latest_readings(hours=1)
        station_status = adapter.get_station_status()
        
        return {
            'latest_records': len(latest),
            'active_stations': len(station_status),
            'last_sync': datetime.now().isoformat(),
        }
    except Exception as e:
        return {'error': str(e)}


def display_main_dashboard():
    """显示主仪表板"""
    clear_screen()
    print_header()
    
    # 系统状态
    print_section("💻 系统信息")
    print_status_item("系统版本", "v2.0.0")
    print_status_item("运行状态", "运行中", "ok")
    print_status_item("启动时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # AutoPilot 状态
    print_section("🤖 AutoPilot 状态")
    autopilot_status = get_autopilot_status()
    if 'error' not in autopilot_status:
        print_status_item("日志文件", "✓ 存在" if autopilot_status['log_exists'] else "✗ 不存在")
        print_status_item("进化代数", str(autopilot_status['evolution_count']))
        if autopilot_status['last_log_time']:
            print_status_item("最后活动", autopilot_status['last_log_time'].strftime("%H:%M:%S"))
    else:
        print_status_item("状态", f"错误: {autopilot_status['error']}", "error")
    
    # 模型状态
    print_section("🧠 模型状态")
    model_status = get_model_status()
    if 'error' not in model_status:
        print_status_item("可用模型数", str(len(model_status['available_models'])))
        if model_status['latest_model']:
            latest = model_status['latest_model']
            print_status_item("最新模型", f"{latest['name']} ({latest['size']:.1f}MB)")
    else:
        print_status_item("状态", f"错误: {model_status['error']}", "error")
    
    # 数据状态
    print_section("📊 数据状态")
    data_status = get_data_status()
    if 'error' not in data_status:
        print_status_item("最新记录数", str(data_status['latest_records']))
        print_status_item("活跃站点", str(data_status['active_stations']) + "/5")
        print_status_item("同步时间", data_status['last_sync'].split('T')[1][:8])
    else:
        print_status_item("状态", f"错误: {data_status['error']}", "error")
    
    # 系统状态
    print_section("🌊 实时水文数据")
    try:
        system_status = get_system_status()
        if 'stations' in system_status:
            for sid, info in list(system_status['stations'].items())[:5]:
                water_level = info.get('water_level', 0)
                status_icon = info.get('water_status', '?').split()[0]  # 获取图标
                print_status_item(
                    info.get('station_name', sid),
                    f"水位 {water_level:.0f}m {status_icon}",
                    "ok" if "🟢" in info.get('water_status', '') else "warning"
                )
    except Exception as e:
        print_status_item("错误", str(e), "error")
    
    # 底部信息
    print(f"\n{Colors.GRAY}")
    print("=" * 60)
    print("💡 更多功能: streamlit run smart_flood_system/app/dashboard_v2.py")
    print("=" * 60)
    print(f"{Colors.END}\n")


def run_interactive_monitor():
    """运行交互式监控器"""
    
    while True:
        try:
            display_main_dashboard()
            
            print("\n选项:")
            print("  [1] 刷新")
            print("  [2] 显示日志")
            print("  [3] 显示最近报告")
            print("  [4] 触发进化")
            print("  [5] 退出")
            
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == '1':
                continue
            elif choice == '2':
                show_logs()
            elif choice == '3':
                show_recent_reports()
            elif choice == '4':
                trigger_evolution()
            elif choice == '5':
                print("退出监控器")
                break
            else:
                print("无效选择")
                time.sleep(2)
        
        except KeyboardInterrupt:
            print("\n\n退出监控器")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            time.sleep(2)


def show_logs():
    """显示日志"""
    clear_screen()
    print_header()
    print_section("📜 最近日志")
    
    log_path = Path("logs/autopilot.log")
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # 显示最后 20 行
            for line in lines[-20:]:
                print(line.rstrip())
    else:
        print("日志文件不存在")
    
    input("\n按 Enter 继续...")


def show_recent_reports():
    """显示最近报告"""
    clear_screen()
    print_header()
    print_section("📋 最近生成的报告")
    
    report_dir = Path("results/reports")
    if report_dir.exists():
        reports = sorted(report_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        for i, report in enumerate(reports[:5], 1):
            mtime = datetime.fromtimestamp(report.stat().st_mtime)
            print(f"  {i}. {report.name}")
            print(f"     修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    else:
        print("报告目录不存在")
    
    input("\n按 Enter 继续...")


def trigger_evolution():
    """触发进化"""
    clear_screen()
    print_header()
    print_section("🧬 触发进化")
    
    print("尝试手动触发进化循环...")
    
    try:
        from smart_flood_system.auto_pilot import AutoPilot
        import asyncio
        
        autopilot = AutoPilot()
        asyncio.run(autopilot.trigger_evolution())
        
        print(f"{Colors.GREEN}✓ 进化已触发{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ 触发失败: {e}{Colors.END}")
    
    time.sleep(2)


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLI 监控器')
    parser.add_argument('--once', action='store_true', help='显示一次后退出')
    parser.add_argument('--logs', action='store_true', help='显示日志')
    parser.add_argument('--reports', action='store_true', help='显示报告')
    
    args = parser.parse_args()
    
    if args.logs:
        show_logs()
    elif args.reports:
        show_recent_reports()
    elif args.once:
        display_main_dashboard()
    else:
        try:
            run_interactive_monitor()
        except KeyboardInterrupt:
            print("\n\n再见!")


if __name__ == '__main__':
    main()
