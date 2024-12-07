"""
Seismic data collector that integrates with USGS Earthquake API
"""
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SeismicDataCollector:
    """Collects real-time seismic data from USGS Earthquake API"""
    
    USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self, update_interval: int = 300):
        """
        Initialize the seismic data collector
        
        Args:
            update_interval: Time between updates in seconds (default: 5 minutes)
        """
        self.update_interval = update_interval
        self.last_update = None
        self.is_running = False
        
    async def get_earthquake_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_magnitude: float = 2.5,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch earthquake data from USGS API
        
        Args:
            start_time: Start time for data query
            end_time: End time for data query
            min_magnitude: Minimum earthquake magnitude to include
            limit: Maximum number of records to return
            
        Returns:
            List of earthquake events
        """
        if not start_time:
            start_time = datetime.now(timezone.utc) - timedelta(days=1)
        if not end_time:
            end_time = datetime.now(timezone.utc)
            
        params = {
            'format': 'geojson',
            'starttime': start_time.isoformat(),
            'endtime': end_time.isoformat(),
            'minmagnitude': str(min_magnitude),
            'limit': str(limit),
            'orderby': 'time'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.USGS_API_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_earthquake_data(data)
                    else:
                        logger.error(f"Failed to fetch earthquake data: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching earthquake data: {str(e)}")
            return []
            
    def _process_earthquake_data(self, data: Dict) -> List[Dict]:
        """
        Process raw earthquake data into standardized format
        
        Args:
            data: Raw data from USGS API
            
        Returns:
            List of processed earthquake events
        """
        processed_data = []
        
        for feature in data.get('features', []):
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            processed_event = {
                'id': feature.get('id'),
                'time': datetime.fromtimestamp(properties.get('time', 0) / 1000, timezone.utc),
                'magnitude': properties.get('mag'),
                'depth': properties.get('depth'),
                'location': {
                    'latitude': geometry.get('coordinates', [0, 0, 0])[1],
                    'longitude': geometry.get('coordinates', [0, 0, 0])[0]
                },
                'place': properties.get('place'),
                'type': properties.get('type'),
                'alert': properties.get('alert'),
                'significance': properties.get('sig')
            }
            
            processed_data.append(processed_event)
            
        return processed_data
        
    async def start_collection(self):
        """Start continuous data collection"""
        self.is_running = True
        while self.is_running:
            try:
                data = await self.get_earthquake_data()
                self.last_update = datetime.now(timezone.utc)
                # TODO: Implement data storage
                logger.info(f"Collected {len(data)} earthquake events")
                
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
                
            await asyncio.sleep(self.update_interval)
            
    def stop_collection(self):
        """Stop continuous data collection"""
        self.is_running = False
