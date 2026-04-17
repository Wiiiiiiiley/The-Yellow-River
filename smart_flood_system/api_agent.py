"""
API Agent - Automated Data Downloader
====================================

Autonomous data ingestion from government/research APIs.

Features:
- Daily scheduled downloads (2 AM)
- Multi-API support (hydrology, meteorology)
- Automatic retry with exponential backoff
- Data validation and archival
- Health monitoring
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class APIDownloader:
    """
    Automated data downloader with fault tolerance.
    
    Handles:
    - Daily data synchronization
    - API authentication
    - Retry logic
    - Data validation
    - Archive management
    """
    
    def __init__(
        self,
        apis: List[str],
        archive_dir: str = "data/archive/",
        retry_attempts: int = 3,
        timeout: int = 60
    ):
        self.apis = apis
        self.archive_dir = Path(archive_dir)
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
        # Create archive directories
        for api in apis:
            (self.archive_dir / api).mkdir(parents=True, exist_ok=True)
        
        # API configurations
        self.api_configs = self._load_api_configs()
        
        # Statistics
        self.download_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0,
            'last_success': None
        }
        
        logger.info(f"APIDownloader initialized: {apis}")
    
    def _load_api_configs(self) -> Dict:
        """Load API configurations."""
        # Default configurations - would be loaded from secure config in production
        configs = {
            'hydrology': {
                'base_url': 'https://api.yellowriver.gov/hydrology/v1',
                'endpoints': {
                    'daily': '/stations/daily',
                    'realtime': '/stations/realtime'
                },
                'auth_type': 'api_key',  # or 'oauth', 'basic'
                'rate_limit': 100  # requests per minute
            },
            'meteorology': {
                'base_url': 'https://api.weather.gov.cn/yellow-river',
                'endpoints': {
                    'forecast': '/forecast/24h',
                    'historical': '/historical/daily'
                },
                'auth_type': 'api_key',
                'rate_limit': 60
            }
        }
        
        return configs
    
    def download_yesterday_data(self) -> Dict:
        """
        Download yesterday's data from all configured APIs.
        
        Returns:
            {'success': bool, 'records': int, 'error': str}
        """
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
        
        logger.info(f"Starting download for {date_str}")
        
        total_records = 0
        errors = []
        
        for api_name in self.apis:
            try:
                records = self._download_api_data(api_name, yesterday)
                total_records += records
                
                logger.info(f"✓ {api_name}: {records} records")
                
            except Exception as e:
                error_msg = f"{api_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"✗ {error_msg}")
        
        # Update statistics
        self.download_stats['total_attempts'] += 1
        
        success = len(errors) == 0 or total_records > 0
        
        if success:
            self.download_stats['successful'] += 1
            self.download_stats['last_success'] = datetime.now().isoformat()
        else:
            self.download_stats['failed'] += 1
        
        return {
            'success': success,
            'records': total_records,
            'date': date_str,
            'errors': errors,
            'apis_downloaded': len(self.apis) - len(errors)
        }
    
    def _download_api_data(self, api_name: str, target_date: datetime) -> int:
        """
        Download data from specific API with retry logic.
        
        Args:
            api_name: Name of API to query
            target_date: Date to download data for
        
        Returns:
            Number of records downloaded
        """
        config = self.api_configs.get(api_name)
        if not config:
            raise ValueError(f"Unknown API: {api_name}")
        
        # Build URL
        base_url = config['base_url']
        endpoint = config['endpoints'].get('daily', '/data')
        url = f"{base_url}{endpoint}"
        
        # Prepare parameters
        params = {
            'date': target_date.strftime('%Y-%m-%d'),
            'format': 'json'
        }
        
        # Add authentication
        headers = self._get_auth_headers(api_name)
        
        # Retry loop
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.retry_attempts}: {url}")
                
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate and process
                    records = self._process_and_save(api_name, target_date, data)
                    return records
                    
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {self.retry_attempts} attempts")
    
    def _get_auth_headers(self, api_name: str) -> Dict:
        """Get authentication headers for API."""
        config = self.api_configs[api_name]
        
        headers = {
            'User-Agent': 'YellowRiver-AutoPilot/1.0',
            'Accept': 'application/json'
        }
        
        if config['auth_type'] == 'api_key':
            # Load from environment or secure storage
            api_key = os.getenv(f'{api_name.upper()}_API_KEY', 'demo_key')
            headers['X-API-Key'] = api_key
            
        elif config['auth_type'] == 'oauth':
            # Would implement OAuth flow
            token = self._get_oauth_token(api_name)
            headers['Authorization'] = f'Bearer {token}'
        
        return headers
    
    def _get_oauth_token(self, api_name: str) -> str:
        """Get OAuth token (placeholder)."""
        # Would implement actual OAuth flow
        return 'demo_token'
    
    def _process_and_save(
        self,
        api_name: str,
        date: datetime,
        data: Dict
    ) -> int:
        """
        Process API response and save to archive.
        
        Args:
            api_name: Source API name
            date: Data date
            data: API response data
        
        Returns:
            Number of records saved
        """
        # Convert to DataFrame
        if 'records' in data:
            records = data['records']
        elif 'data' in data:
            records = data['data']
        else:
            records = [data]  # Single record
        
        if not records:
            logger.warning(f"No records found for {date}")
            return 0
        
        df = pd.DataFrame(records)
        
        # Add metadata
        df['download_timestamp'] = datetime.now().isoformat()
        df['api_source'] = api_name
        df['data_date'] = date.strftime('%Y-%m-%d')
        
        # Validate required columns
        required_cols = ['timestamp', 'station_id', 'water_level', 'flow_rate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Save to archive
        year_month = date.strftime('%Y/%m')
        archive_path = self.archive_dir / api_name / year_month
        archive_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{date.strftime('%Y%m%d')}_{api_name}.parquet"
        filepath = archive_path / filename
        
        # Save as parquet (efficient storage)
        df.to_parquet(filepath, compression='snappy')
        
        # Also save CSV for human readability
        csv_path = archive_path / f"{date.strftime('%Y%m%d')}_{api_name}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.debug(f"Saved {len(df)} records to {filepath}")
        
        return len(df)
    
    def check_health(self) -> bool:
        """Check API connectivity health."""
        healthy = True
        
        for api_name in self.apis:
            try:
                config = self.api_configs[api_name]
                url = f"{config['base_url']}/health"
                
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"{api_name} health check failed: {response.status_code}")
                    healthy = False
                    
            except Exception as e:
                logger.warning(f"{api_name} health check error: {e}")
                healthy = False
        
        return healthy
    
    def get_stats(self) -> Dict:
        """Get download statistics."""
        return {
            **self.download_stats,
            'apis_configured': len(self.apis),
            'archive_dir': str(self.archive_dir),
            'total_size_gb': self._get_archive_size()
        }
    
    def _get_archive_size(self) -> float:
        """Calculate total archive size in GB."""
        total_bytes = 0
        
        for path in self.archive_dir.rglob('*'):
            if path.is_file():
                total_bytes += path.stat().st_size
        
        return total_bytes / (1024 ** 3)


if __name__ == "__main__":
    """Test API downloader."""
    print("Testing APIDownloader...")
    
    # Initialize
    downloader = APIDownloader(
        apis=['hydrology'],
        archive_dir='/tmp/test_archive'
    )
    
    # Test download
    result = downloader.download_yesterday_data()
    
    print(f"\nDownload result:")
    print(f"Success: {result['success']}")
    print(f"Records: {result['records']}")
    
    # Stats
    stats = downloader.get_stats()
    print(f"\nStats: {stats}")
    
    print("\n✓ APIDownloader test complete!")
