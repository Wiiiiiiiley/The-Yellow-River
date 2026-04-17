"""Data loading utilities."""

import pandas as pd
from pathlib import Path
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """Load raw data from CSV files."""
    raw_dir = Path(data_dir) / "raw"
    
    # Combine all CSV files in raw directory
    dfs = []
    for csv_file in raw_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
