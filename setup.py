"""
设置脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart-flood-system",
    version="2.0.0",
    author="Yellow River Digital Twin Project",
    description="智慧黄河防汛决策系统 - 端到端智能防洪控制系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wiiiiiiiley/The-Yellow-River",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.20",
        "pandas>=1.2",
        "pyyaml>=5.4",
        "streamlit>=1.0",
        "plotly>=5.0",
        "apscheduler>=3.9",
        "optuna>=2.10",
        "aiohttp>=3.8",
        "scikit-learn>=0.24",
        "scipy>=1.6",
    ],
    entry_points={
        "console_scripts": [
            "syrfms=smart_flood_system.main_system:main",
            "autopilot=smart_flood_system.auto_pilot:main",
            "monitor=smart_flood_system.monitor:main",
        ],
    },
)
