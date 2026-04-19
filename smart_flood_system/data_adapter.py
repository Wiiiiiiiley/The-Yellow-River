"""
统一数据适配器 - 多源数据集成层
支持: SQLite / CSV / API / 模拟数据
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """数据源基类"""
    
    @abstractmethod
    def get_readings(self, station_id: str, hours: int = 24) -> pd.DataFrame:
        """获取传感器读数"""
        pass
    
    @abstractmethod
    def get_latest_reading(self, station_id: str) -> Dict:
        """获取最新读数"""
        pass


class SQLiteSource(DataSource):
    """SQLite 数据源"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._ensure_db()
    
    def _ensure_db(self):
        """确保数据库和表存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                water_level REAL,
                discharge REAL,
                sediment_concentration REAL,
                temperature REAL,
                rainfall REAL,
                prediction_target REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(station_id, timestamp)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_station_time ON sensor_data(station_id, timestamp)')
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """获取数据库连接"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def get_readings(self, station_id: str, hours: int = 24) -> pd.DataFrame:
        """从 SQLite 获取读数"""
        conn = self.get_connection()
        
        since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        query = f'''
            SELECT station_id, timestamp, water_level, discharge, 
                   sediment_concentration, temperature, rainfall
            FROM sensor_data
            WHERE station_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''
        
        try:
            df = pd.read_sql_query(query, conn, params=(station_id, since_timestamp))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        except Exception as e:
            logger.error(f"SQLite 查询失败: {e}")
            return pd.DataFrame()
    
    def get_latest_reading(self, station_id: str) -> Dict:
        """获取最新读数"""
        conn = self.get_connection()
        
        query = '''
            SELECT * FROM sensor_data
            WHERE station_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        '''
        
        try:
            cursor = conn.execute(query, (station_id,))
            row = cursor.fetchone()
            if row:
                columns = ['id', 'station_id', 'timestamp', 'water_level', 'discharge',
                          'sediment_concentration', 'temperature', 'rainfall', 'prediction_target', 'created_at']
                return dict(zip(columns, row))
            return {}
        except Exception as e:
            logger.error(f"获取最新读数失败: {e}")
            return {}
    
    def insert_reading(self, station_id: str, timestamp: int, 
                      water_level: float, discharge: float, 
                      sediment_concentration: float, temperature: float = None,
                      rainfall: float = None):
        """插入新读数"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO sensor_data 
                (station_id, timestamp, water_level, discharge, 
                 sediment_concentration, temperature, rainfall)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (station_id, timestamp, water_level, discharge, 
                  sediment_concentration, temperature, rainfall))
            conn.commit()
        except Exception as e:
            logger.error(f"插入数据失败: {e}")


class CSVSource(DataSource):
    """CSV 文件数据源"""
    
    def __init__(self, csv_dir: str = "data/archive/"):
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
    
    def get_readings(self, station_id: str, hours: int = 24) -> pd.DataFrame:
        """从 CSV 获取读数"""
        csv_files = list(self.csv_dir.glob(f"*{station_id}*.csv"))
        
        if not csv_files:
            return pd.DataFrame()
        
        dfs = []
        since_time = datetime.now() - timedelta(hours=hours)
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['timestamp'] >= since_time]
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"读取 {csv_file} 失败: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
        return pd.DataFrame()
    
    def get_latest_reading(self, station_id: str) -> Dict:
        """获取最新读数"""
        df = self.get_readings(station_id, hours=720)  # 30 天
        if not df.empty:
            latest = df.iloc[-1].to_dict()
            return latest
        return {}


class SimulatedSource(DataSource):
    """模拟数据源"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def get_readings(self, station_id: str, hours: int = 24) -> pd.DataFrame:
        """生成模拟读数"""
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='h')
        
        # 模拟"大水大沙"黄河特征
        base_water_level = 1000 + 50 * np.sin(np.arange(hours) * 2 * np.pi / 168)  # 周期变化
        water_level = base_water_level + np.random.normal(0, 5, hours)
        
        # 泥沙浓度与流量正相关
        base_discharge = 1500 + 300 * np.sin(np.arange(hours) * 2 * np.pi / 168)
        discharge = base_discharge + np.random.normal(0, 50, hours)
        
        # 泥沙滞后于流量峰值
        sediment_lag = 6  # 6 小时滞后
        base_sediment = 50 + 30 * np.sin(np.arange(hours) * 2 * np.pi / 168)
        # 创建滞后效果：前 sediment_lag 小时为 0，后面为滞后的基础沙量
        sediment_concentration = np.concatenate([
            np.zeros(sediment_lag),
            base_sediment[:-sediment_lag]
        ]) + np.random.normal(0, 5, hours)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'station_id': station_id,
            'water_level': np.maximum(water_level, 800),
            'discharge': np.maximum(discharge, 500),
            'sediment_concentration': np.maximum(sediment_concentration, 0),
            'temperature': 15 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 168) + np.random.normal(0, 2, hours),
            'rainfall': np.maximum(np.random.exponential(2, hours), 0)
        })
        
        return df
    
    def get_latest_reading(self, station_id: str) -> Dict:
        """获取最新模拟读数"""
        df = self.get_readings(station_id, hours=1)
        if not df.empty:
            latest = df.iloc[-1].to_dict()
            return latest
        return {}


class DataAdapter:
    """统一数据适配器"""
    
    STATIONS = {
        'HYD_001': {'name': '兰州', 'lat': 36.0611, 'lon': 103.8343},
        'HYD_002': {'name': '河口镇', 'lat': 35.7897, 'lon': 110.2406},
        'HYD_003': {'name': '龙门', 'lat': 35.1064, 'lon': 110.5921},
        'HYD_004': {'name': '三门峡', 'lat': 34.7753, 'lon': 111.1934},
        'HYD_005': {'name': '花园口', 'lat': 34.9031, 'lon': 113.1889},
    }
    
    ALERT_THRESHOLDS = {
        'normal_max_water_level': 1000,
        'warning_water_level': 1100,
        'critical_water_level': 1200,
        'sediment_concentration_warning': 100,  # kg/m³
    }
    
    def __init__(self, db_path: str = "data/autopilot.db", 
                 csv_dir: str = "data/archive/",
                 use_simulated: bool = True):
        self.sources: List[DataSource] = []
        
        # 优先级顺序
        self.sources.append(SQLiteSource(db_path))
        self.sources.append(CSVSource(csv_dir))
        if use_simulated:
            self.sources.append(SimulatedSource())
    
    def get_latest_readings(self, hours: int = 24, 
                           station_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """获取所有站点的最新读数"""
        if station_ids is None:
            station_ids = list(self.STATIONS.keys())
        
        all_data = []
        
        for station_id in station_ids:
            for source in self.sources:
                try:
                    df = source.get_readings(station_id, hours=hours)
                    if not df.empty:
                        all_data.append(df)
                        break  # 获取到数据就停止
                except Exception as e:
                    logger.debug(f"从 {source.__class__.__name__} 获取数据失败: {e}")
                    continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True).sort_values('timestamp')
        
        # 回退到模拟数据
        logger.warning("无法获取真实数据，使用模拟数据")
        simulated = SimulatedSource()
        dfs = [simulated.get_readings(sid, hours=hours) for sid in station_ids]
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    
    def get_station_status(self) -> Dict[str, Dict]:
        """获取站点状态及警报"""
        status = {}
        
        for station_id in self.STATIONS.keys():
            for source in self.sources:
                try:
                    reading = source.get_latest_reading(station_id)
                    if reading:
                        status[station_id] = self._compute_status(reading)
                        break
                except:
                    continue
            
            if station_id not in status:
                # 使用模拟数据
                sim_source = SimulatedSource()
                reading = sim_source.get_latest_reading(station_id)
                status[station_id] = self._compute_status(reading)
        
        return status
    
    def _compute_status(self, reading: Dict) -> Dict:
        """计算站点状态"""
        water_level = reading.get('water_level', 0)
        sediment = reading.get('sediment_concentration', 0)
        timestamp = reading.get('timestamp', datetime.now())
        
        # 确定水位状态
        if water_level >= self.ALERT_THRESHOLDS['critical_water_level']:
            water_status = '🔴 临界'
        elif water_level >= self.ALERT_THRESHOLDS['warning_water_level']:
            water_status = '🟡 警告'
        else:
            water_status = '🟢 正常'
        
        # 确定泥沙状态
        if sediment >= self.ALERT_THRESHOLDS['sediment_concentration_warning']:
            sediment_status = '⚠️ 高浓度'
        else:
            sediment_status = '✓ 正常'
        
        return {
            'station_id': reading.get('station_id'),
            'station_name': self.STATIONS.get(reading.get('station_id'), {}).get('name'),
            'water_level': water_level,
            'water_status': water_status,
            'discharge': reading.get('discharge', 0),
            'sediment_concentration': sediment,
            'sediment_status': sediment_status,
            'temperature': reading.get('temperature', 0),
            'rainfall': reading.get('rainfall', 0),
            'timestamp': timestamp,
        }
    
    def get_prediction_inputs(self, station_id: str, 
                             sequence_length: int = 24) -> Tuple[np.ndarray, Dict]:
        """获取预测输入张量"""
        df = self.get_latest_readings(hours=sequence_length * 2, station_ids=[station_id])
        
        if df.empty:
            return np.zeros((sequence_length, 6)), {}
        
        # 特征标准化
        features = ['water_level', 'discharge', 'sediment_concentration', 
                   'temperature', 'rainfall']
        
        # 取最后 sequence_length 个数据点
        df = df.tail(sequence_length).reset_index(drop=True)
        
        # 填充缺失值
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
            df[feat].fillna(df[feat].mean(), inplace=True)
        
        # 标准化
        features_array = df[features].values.astype(np.float32)
        feature_stats = {
            'mean': features_array.mean(axis=0),
            'std': features_array.std(axis=0) + 1e-8,
        }
        features_array = (features_array - feature_stats['mean']) / feature_stats['std']
        
        # 添加时间特征
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour / 24.0
        
        # 合并成输入张量（sequence_length, 6）
        inputs = np.column_stack([
            features_array,
            df['hour'].values.reshape(-1, 1)
        ])
        
        return inputs.astype(np.float32), feature_stats
    
    def get_historical_extremes(self, station_id: str, 
                               year: int = 2021) -> pd.DataFrame:
        """获取历史极端事件数据"""
        csv_dir = Path("data/archive/")
        
        # 查找包含年份的历史数据
        extreme_files = list(csv_dir.glob(f"*{year}*{station_id}*.csv"))
        
        dfs = []
        for f in extreme_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except:
                pass
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        
        # 如果没有找到，生成模拟极端数据
        logger.warning(f"未找到 {year} 年 {station_id} 的历史数据，生成模拟极端洪水数据")
        df = self._generate_extreme_event(station_id, year)
        return df
    
    def _generate_extreme_event(self, station_id: str, year: int) -> pd.DataFrame:
        """生成模拟极端洪水数据"""
        # 模拟 2021 年极端洪水
        timestamps = pd.date_range(start=f'{year}-07-01', end=f'{year}-09-30', freq='h')
        n = len(timestamps)
        
        # 极端洪水特征：高水位 + 高泥沙
        water_level = 1300 + 200 * np.sin(np.arange(n) * 2 * np.pi / n) + np.random.normal(0, 20, n)
        discharge = 5000 + 1000 * np.sin(np.arange(n) * 2 * np.pi / n) + np.random.normal(0, 100, n)
        sediment = 300 + 100 * np.sin(np.arange(n) * 2 * np.pi / n) + np.random.normal(0, 30, n)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'station_id': station_id,
            'water_level': np.maximum(water_level, 1000),
            'discharge': np.maximum(discharge, 2000),
            'sediment_concentration': np.maximum(sediment, 0),
            'temperature': 25 + np.random.normal(0, 3, n),
            'rainfall': np.maximum(np.random.exponential(5, n), 0),
        })
        
        return df


# 全局单例
_adapter_instance = None


def get_data_adapter() -> DataAdapter:
    """获取全局数据适配器实例"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = DataAdapter()
    return _adapter_instance


if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    adapter = get_data_adapter()
    
    # 测试获取最新读数
    print("=== 最新读数 ===")
    latest = adapter.get_latest_readings(hours=24)
    print(latest.head())
    
    # 测试获取站点状态
    print("\n=== 站点状态 ===")
    status = adapter.get_station_status()
    for sid, info in status.items():
        print(f"{info['station_name']}: {info['water_status']} | {info['sediment_status']}")
    
    # 测试获取预测输入
    print("\n=== 预测输入 ===")
    inputs, stats = adapter.get_prediction_inputs('HYD_001')
    print(f"输入形状: {inputs.shape}")
    print(f"特征统计: {stats}")
