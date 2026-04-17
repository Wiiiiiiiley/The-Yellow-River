"""Visualization functions for Yellow River data."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_time_series(df: pd.DataFrame, 
                     date_col: str = 'date',
                     value_cols: List[str] = ['water_level', 'sediment_content'],
                     title: str = "黄河水位与泥沙含量变化"):
    """Plot time series of water level and sediment content."""
    fig, axes = plt.subplots(len(value_cols), 1, 
                             figsize=(14, 4*len(value_cols)), 
                             sharex=True)
    
    if len(value_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, value_cols):
        ax.plot(pd.to_datetime(df[date_col]), df[col], label=col)
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_predictions(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     target_names: List[str] = ['Water Level', 'Sediment']):
    """Plot true vs predicted values."""
    fig, axes = plt.subplots(1, len(target_names), figsize=(14, 5))
    
    if len(target_names) == 1:
        axes = [axes]
    
    for ax, name, i in zip(axes, target_names, range(len(target_names))):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                [y_true[:, i].min(), y_true[:, i].max()], 
                'r--', lw=2)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}: True vs Predicted')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
