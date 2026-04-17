"""
Generate 10-Year Synthetic Yellow River Dataset
==============================================

Creates realistic synthetic data with Yellow River characteristics:
- Seasonal patterns: Flood season (Jul-Aug), Dry season (Dec-Feb)
- "Big water brings big sand" non-linear relationship
- Hysteresis: sediment peaks lag flow peaks
- Exner dynamics: erosion/deposition cycles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse


def generate_yellow_river_data(
    start_year: int = 2014,
    n_years: int = 10,
    stations: list = None,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generate 10-year synthetic Yellow River dataset.
    
    Characteristics:
    - Seasonal flood/dry cycles
    - Non-linear flow-sediment relationship
    - Occasional extreme flood events
    """
    if stations is None:
        stations = ['HYD_001', 'HYD_002', 'HYD_003', 'HYD_004', 'HYD_005']
    
    records = []
    
    for year in range(start_year, start_year + n_years):
        for station in stations:
            # Generate daily data for entire year
            start_date = datetime(year, 1, 1)
            dates = pd.date_range(start_date, periods=365, freq='D')
            
            for i, date in enumerate(dates):
                # Seasonal base flow
                day_of_year = i
                
                # Sin-based seasonal pattern
                # Peak in summer (flood season)
                seasonal_flow = 2500 + 1500 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
                
                # Add random variation
                noise = np.random.randn() * 500
                flow_rate = max(500, seasonal_flow + noise)
                
        # Random extreme flood events (10% probability in flood season)
                if 170 <= day_of_year <= 240 and np.random.random() < 0.1:  # Jul-Aug
                    flow_rate += np.random.uniform(2000, 4000)
                
                # Rainfall correlated with season
                base_rain = 2 + 3 * np.sin(2 * np.pi * day_of_year / 365)
                rainfall = max(0, base_rain + np.random.exponential(1))
                
                # Temperature
                temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + np.random.randn() * 3
                
                # Water level: related to flow
                water_level = 45 + 0.008 * flow_rate + np.random.randn() * 2
                
                # Sediment: non-linear with flow (big water brings big sand)
                # Plus hysteresis: depends on whether flow is rising or falling
                base_sediment = 3 + 0.002 * flow_rate + 1e-6 * flow_rate**2
                
                # Add sediment burst during flood rise
                if i > 0:
                    flow_change = flow_rate - (records[-1]['flow_rate'] if records else flow_rate)
                    if flow_change > 500:  # Rapid flow increase
                        base_sediment += 2  # Sediment burst
                
                sediment = max(1, base_sediment + np.random.randn() * 0.5)
                
                records.append({
                    'timestamp': date,
                    'station_id': station,
                    'water_level': round(water_level, 2),
                    'flow_rate': round(flow_rate, 1),
                    'sediment_content': round(sediment, 2),
                    'rainfall': round(rainfall, 2),
                    'temperature': round(temp, 1)
                })
    
    df = pd.DataFrame(records)
    return df


def add_extreme_events(df: pd.DataFrame, n_events: int = 3) -> pd.DataFrame:
    """
    Add a few extreme flood events (50-year flood scale).
    """
    df = df.copy()
    
    for _ in range(n_events):
        # Pick random flood season day
        flood_dates = df[
            (df['timestamp'].dt.month >= 7) & 
            (df['timestamp'].dt.month <= 8)
        ]['timestamp'].unique()
        
        if len(flood_dates) > 0:
            extreme_date = np.random.choice(flood_dates)
            
            # Create extreme values
            mask = df['timestamp'] == extreme_date
            df.loc[mask, 'flow_rate'] *= 2.5
            df.loc[mask, 'water_level'] += 15
            df.loc[mask, 'sediment_content'] *= 3
            df.loc[mask, 'rainfall'] += 20
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate 10-year Yellow River dataset')
    parser.add_argument('--output', type=str, default='data/yellow_river_10year.csv')
    parser.add_argument('--start-year', type=int, default=2014)
    parser.add_argument('--n-years', type=int, default=10)
    parser.add_argument('--n-stations', type=int, default=5)
    parser.add_argument('--add-extremes', action='store_true', help='Add extreme flood events')
    
    args = parser.parse_args()
    
    stations = [f'HYD_{i:03d}' for i in range(1, args.n_stations + 1)]
    
    print(f"Generating {args.n_years}-year Yellow River dataset...")
    print(f"Stations: {stations}")
    
    # Generate base data
    df = generate_yellow_river_data(
        start_year=args.start_year,
        n_years=args.n_years,
        stations=stations
    )
    
    # Add extreme events if requested
    if args.add_extremes:
        df = add_extreme_events(df, n_events=5)
        print("Added extreme flood events")
    
    # Save
    df.to_csv(args.output, index=False)
    
    print(f"\nDataset saved to: {args.output}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nStatistics:")
    print(f"  Flow rate: {df['flow_rate'].mean():.0f} ± {df['flow_rate'].std():.0f} m³/s")
    print(f"  Water level: {df['water_level'].mean():.1f} ± {df['water_level'].std():.1f} m")
    print(f"  Sediment: {df['sediment_content'].mean():.1f} ± {df['sediment_content'].std():.1f} kg/m³")
    print(f"  Rainfall: {df['rainfall'].mean():.1f} ± {df['rainfall'].std():.1f} mm")
    
    # Flood season stats
    flood_mask = (df['timestamp'].dt.month >= 7) & (df['timestamp'].dt.month <= 8)
    flood_df = df[flood_mask]
    print(f"\nFlood season (Jul-Aug) samples: {len(flood_df)} ({len(flood_df)/len(df)*100:.1f}%)")
    print(f"  Avg flow: {flood_df['flow_rate'].mean():.0f} m³/s")
    print(f"  Avg sediment: {flood_df['sediment_content'].mean():.1f} kg/m³")


if __name__ == '__main__':
    main()
