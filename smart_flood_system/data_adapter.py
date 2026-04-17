"""
Data Adapter - Real-time Sensor Data Interface
===============================================

Connects dashboard to real sensor data sources:
1. CSV files (historical/archived data)
2. SQLite database (autopilot storage)
3. API endpoints (real-time streaming)
4. Simulation mode (for testing)

Usage:
    from data_adapter import SensorDataAdapter
    
    adapter = SensorDataAdapter()
    df = adapter.get_latest_readings(station_id='HYD_001', hours=24)
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SensorDataAdapter:
    """
    Unified interface for sensor data access.
    
    Supports multiple data sources with fallback:
    1. Real SQLite database
    2. CSV archives
    3. Simulated data (fallback)
    """
    
    # Station configurations (Yellow River hydrological stations)
    STATIONS = {
        'HYD_001': {'name': '兰州', 'lat': 36.061, 'lon': 103.834, 'type': 'upstream'},
        'HYD_002': {'name': '河口镇', 'lat': 40.333, 'lon': 111.133, 'type': 'midstream'},
        'HYD_003': {'name': '龙门', 'lat': 35.612, 'lon': 110.572, 'type': 'midstream'},
        'HYD_004': {'name': '三门峡', 'lat': 34.800, 'lon': 111.200, 'type': 'downstream'},
        'HYD_005': {'name': '花园口', 'lat': 34.910, 'lon': 113.650, 'type': 'downstream'},
    }
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.archive_dir = self.data_dir / "archive"
        self.db_path = self.data_dir / "autopilot.db"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Data source priority
        self.use_simulation = not self._has_real_data()
        
        if self.use_simulation:
            logger.info("No real data found. Using simulation mode.")
            self._init_simulation()
        else:
            logger.info(f"Connected to real data sources: {self.db_path}")
    
    def _has_real_data(self) -> bool:
        """Check if real data sources exist."""
        return self.db_path.exists() or any(self.archive_dir.glob("**/*.csv"))
    
    def _init_simulation(self):
        """Initialize realistic simulation based on Yellow River patterns."""
        self.sim_seed = int(datetime.now().timestamp())
        np.random.seed(self.sim_seed)
        logger.info(f"Simulation initialized with seed {self.sim_seed}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_latest_readings(self, hours: int = 24) -> pd.DataFrame:
        """
        Get latest sensor readings for all stations.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            DataFrame with columns: [timestamp, station_id, water_level, 
                                     flow_rate, sediment, temperature, rainfall]
        """
        if self.use_simulation:
            return self._simulate_readings(hours)
        
        # Try database first
        if self.db_path.exists():
            return self._read_from_db(hours)
        
        # Fall back to CSV
        return self._read_from_csv(hours)
    
    def get_station_status(self) -> pd.DataFrame:
        """Get current status of all monitoring stations."""
        now = datetime.now()
        
        records = []
        for station_id, info in self.STATIONS.items():
            # Get latest reading
            latest = self.get_latest_readings(hours=1)
            if not latest.empty:
                station_data = latest[latest['station_id'] == station_id]
                if not station_data.empty:
                    latest_row = station_data.iloc[-1]
                    records.append({
                        'station_id': station_id,
                        'name': info['name'],
                        'type': info['type'],
                        'lat': info['lat'],
                        'lon': info['lon'],
                        'last_update': latest_row['timestamp'],
                        'water_level': latest_row['water_level'],
                        'flow_rate': latest_row['flow_rate'],
                        'sediment': latest_row['sediment'],
                        'status': self._calculate_status(latest_row['water_level']),
                        'data_age_minutes': (now - pd.to_datetime(latest_row['timestamp'])).total_seconds() / 60
                    })
        
        return pd.DataFrame(records)
    
    def get_prediction_inputs(self, station_id: str = None) -> Dict:
        """
        Get formatted inputs for model prediction.
        
        Returns features needed by PINN model:
        - flow_rate, rainfall, temperature
        - temporal encodings (sin/cos year, season, hour)
        """
        df = self.get_latest_readings(hours=1)
        
        if station_id:
            df = df[df['station_id'] == station_id]
        
        if df.empty:
            return self._get_default_inputs()
        
        latest = df.iloc[-1]
        now = datetime.now()
        
        # Temporal encodings
        day_of_year = now.timetuple().tm_yday
        hour = now.hour
        
        return {
            'flow_rate': latest['flow_rate'],
            'sediment': latest['sediment'],
            'temperature': latest['temperature'],
            'rainfall': latest['rainfall'],
            'water_level': latest['water_level'],
            'sin_year': np.sin(2 * np.pi * day_of_year / 365),
            'cos_year': np.cos(2 * np.pi * day_of_year / 365),
            'sin_season': np.sin(2 * np.pi * day_of_year / 365 * 4),  # 4 seasons
            'cos_season': np.cos(2 * np.pi * day_of_year / 365 * 4),
            'is_flood_season': 1 if day_of_year in range(181, 273) else 0,  # July-Sept
            'sin_hour': np.sin(2 * np.pi * hour / 24),
            'cos_hour': np.cos(2 * np.pi * hour / 24),
            'timestamp': now.isoformat()
        }
    
    def get_historical_extremes(self, station_id: str = None, years: int = 5) -> Dict:
        """
        Get historical extreme flood data for backtrack validation.
        
        Returns 2021 flood data or simulated extremes.
        """
        # Check for archived extreme data
        extreme_file = self.archive_dir / "hydrology" / "2021_flood.csv"
        
        if extreme_file.exists():
            df = pd.read_csv(extreme_file)
            if station_id:
                df = df[df['station_id'] == station_id]
            return {
                'has_real_data': True,
                'max_water_level': df['water_level'].max(),
                'max_flow_rate': df['flow_rate'].max(),
                'max_sediment': df['sediment'].max(),
                'data': df.to_dict('records')
            }
        
        # Return simulated 2021 extreme flood
        return self._simulate_2021_flood(station_id)
    
    def save_prediction_result(self, prediction: Dict):
        """Save model prediction to database for tracking."""
        if not self.db_path.exists():
            self._init_database()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, station_id, predicted_level, predicted_flow, 
             confidence, physics_residual, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction.get('station_id', 'unknown'),
            prediction.get('water_level', 0),
            prediction.get('flow_rate', 0),
            prediction.get('confidence', 0),
            prediction.get('physics_residual', 0),
            prediction.get('model_version', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved prediction for {prediction.get('station_id')}")
    
    # =========================================================================
    # PRIVATE METHODS - Data Sources
    # =========================================================================
    
    def _read_from_db(self, hours: int) -> pd.DataFrame:
        """Read data from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = f"""
            SELECT * FROM sensor_data 
            WHERE timestamp > '{since}'
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def _read_from_csv(self, hours: int) -> pd.DataFrame:
        """Read data from CSV archives."""
        all_data = []
        
        for csv_file in self.archive_dir.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {csv_file}: {e}")
        
        if not all_data:
            return self._simulate_readings(hours)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter by time
        since = datetime.now() - timedelta(hours=hours)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] > since]
        
        return df.sort_values('timestamp', ascending=False)
    
    # =========================================================================
    # PRIVATE METHODS - Simulation
    # =========================================================================
    
    def _simulate_readings(self, hours: int) -> pd.DataFrame:
        """Generate realistic sensor readings based on Yellow River patterns."""
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(hours, 0, -1)]
        
        records = []
        
        for station_id, info in self.STATIONS.items():
            # Base values vary by station position
            base_level = 50 + (info['lat'] - 34) * 0.5  # Higher upstream
            base_flow = 2000 + np.random.randn() * 500
            
            for t in timestamps:
                # Add diurnal cycle
                hour = t.hour
                diurnal = np.sin(2 * np.pi * hour / 24) * 0.5
                
                # Add seasonal trend (flood season July-Sept)
                day_of_year = t.timetuple().tm_yday
                seasonal = 0
                if 180 <= day_of_year <= 270:  # Flood season
                    seasonal = np.sin(2 * np.pi * (day_of_year - 180) / 90) * 5
                
                # Random variation
                noise = np.random.randn() * 0.3
                
                water_level = base_level + diurnal + seasonal + noise
                flow_rate = base_flow * (1 + (water_level - base_level) * 0.02)
                
                # Sediment follows "big water brings big sand" principle
                sediment = 15 + flow_rate * 0.005 + np.random.randn() * 2
                
                records.append({
                    'timestamp': t.isoformat(),
                    'station_id': station_id,
                    'station_name': info['name'],
                    'water_level': round(water_level, 2),
                    'flow_rate': round(flow_rate, 1),
                    'sediment': round(max(0, sediment), 2),
                    'temperature': round(20 + seasonal * 0.5 + np.random.randn() * 2, 1),
                    'rainfall': round(max(0, np.random.exponential(2)), 1),
                    'is_simulated': True
                })
        
        return pd.DataFrame(records)
    
    def _simulate_2021_flood(self, station_id: str = None) -> Dict:
        """Simulate 2021 extreme flood data for backtrack testing."""
        # 2021 July flood event data (simulated but realistic)
        base_records = [
            {'timestamp': '2021-07-20T00:00:00', 'water_level': 58.5, 'flow_rate': 4500, 'sediment': 45},
            {'timestamp': '2021-07-20T06:00:00', 'water_level': 62.1, 'flow_rate': 5200, 'sediment': 58},
            {'timestamp': '2021-07-20T12:00:00', 'water_level': 65.8, 'flow_rate': 6100, 'sediment': 72},
            {'timestamp': '2021-07-20T18:00:00', 'water_level': 67.2, 'flow_rate': 6800, 'sediment': 85},
            {'timestamp': '2021-07-21T00:00:00', 'water_level': 64.5, 'flow_rate': 5500, 'sediment': 65},
        ]
        
        df = pd.DataFrame(base_records)
        
        return {
            'has_real_data': False,
            'max_water_level': df['water_level'].max(),
            'max_flow_rate': df['flow_rate'].max(),
            'max_sediment': df['sediment'].max(),
            'data': df.to_dict('records')
        }
    
    def _calculate_status(self, water_level: float) -> str:
        """Calculate alert status based on water level."""
        if water_level >= 65:
            return 'CRITICAL'
        elif water_level >= 55:
            return 'WARNING'
        return 'NORMAL'
    
    def _get_default_inputs(self) -> Dict:
        """Return default inputs when no data available."""
        return {
            'flow_rate': 2000,
            'sediment': 25,
            'temperature': 22,
            'rainfall': 0,
            'water_level': 52,
            'sin_year': 0,
            'cos_year': 1,
            'sin_season': 0,
            'cos_season': 1,
            'is_flood_season': 0,
            'sin_hour': 0,
            'cos_hour': 1,
            'timestamp': datetime.now().isoformat()
        }
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                station_id TEXT,
                water_level REAL,
                flow_rate REAL,
                sediment REAL,
                temperature REAL,
                rainfall REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                station_id TEXT,
                predicted_level REAL,
                predicted_flow REAL,
                confidence REAL,
                physics_residual REAL,
                model_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized database at {self.db_path}")


# =========================================================================
# Convenience Functions
# =========================================================================

def get_data_adapter() -> SensorDataAdapter:
    """Factory function to get configured data adapter."""
    return SensorDataAdapter(data_dir="data/")


if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.INFO)
    
    adapter = get_data_adapter()
    
    print("\n=== Latest Readings ===")
    df = adapter.get_latest_readings(hours=6)
    print(df.head(10))
    
    print("\n=== Station Status ===")
    status = adapter.get_station_status()
    print(status)
    
    print("\n=== Prediction Inputs ===")
    inputs = adapter.get_prediction_inputs('HYD_001')
    print(json.dumps(inputs, indent=2, ensure_ascii=False))
