"""Feature engineering for Yellow River data."""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Create temporal features from date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Season (Chinese meteorological seasons)
    df['season'] = df['month'].map({
        3: 0, 4: 0, 5: 0,      # Spring
        6: 1, 7: 1, 8: 1,      # Summer
        9: 2, 10: 2, 11: 2,    # Autumn
        12: 3, 1: 3, 2: 3     # Winter
    })
    
    return df


def create_lag_features(df: pd.DataFrame, columns: List[str], 
                        lags: List[int]) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str],
                            windows: List[int]) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = df.copy()
    for col in columns:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
    return df
