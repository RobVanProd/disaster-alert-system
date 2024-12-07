"""
Weather data collector that integrates with OpenWeatherMap API
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
import asyncio
import os

logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collects real-time weather data from OpenWeatherMap API"""
    
    OWM_API_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: Optional[str] = None, update_interval: int = 900):
        """
        Initialize the weather data collector
        
        Args:
            api_key: OpenWeatherMap API key (can also be set via OWM_API_KEY env var)
            update_interval: Time between updates in seconds (default: 15 minutes)
        """
        self.api_key = api_key or os.getenv('OWM_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required")
            
        self.update_interval = update_interval
        self.last_update = None
        self.is_running = False
        self.monitored_locations = []
        
    def add_location(self, latitude: float, longitude: float, name: str = None):
        """
        Add a location to monitor
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            name: Optional location name
        """
        self.monitored_locations.append({
            'latitude': latitude,
            'longitude': longitude,
            'name': name or f"Location-{len(self.monitored_locations) + 1}"
        })
        
    async def get_weather_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch weather data for a specific location
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            
        Returns:
            Weather data dictionary
        """
        params = {
            'lat': str(latitude),
            'lon': str(longitude),
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.OWM_API_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_weather_data(data)
                    elif response.status == 401:
                        error_data = await response.json()
                        logger.error(f"OpenWeatherMap API key error: {error_data.get('message', 'Unknown error')}. "
                                   "If this is a new API key, please wait up to 2 hours for it to be fully activated.")
                        return {}
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to fetch weather data (status {response.status}): {error_data.get('message', 'Unknown error')}")
                        return {}
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return {}
            
    def _process_weather_data(self, data: Dict) -> Dict:
        """
        Process raw weather data into standardized format
        
        Args:
            data: Raw data from OpenWeatherMap API
            
        Returns:
            Processed weather data
        """
        weather = data.get('weather', [{}])[0]
        main = data.get('main', {})
        wind = data.get('wind', {})
        
        return {
            'time': datetime.now(timezone.utc),
            'location': {
                'latitude': data.get('coord', {}).get('lat'),
                'longitude': data.get('coord', {}).get('lon'),
                'name': data.get('name')
            },
            'temperature': main.get('temp'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'weather_main': weather.get('main'),
            'weather_description': weather.get('description'),
            'clouds': data.get('clouds', {}).get('all')
        }
        
    async def start_collection(self):
        """Start continuous data collection for all monitored locations"""
        self.is_running = True
        while self.is_running:
            try:
                for location in self.monitored_locations:
                    data = await self.get_weather_data(
                        location['latitude'],
                        location['longitude']
                    )
                    if data:
                        # TODO: Implement data storage
                        logger.info(f"Collected weather data for {location['name']}")
                        
                self.last_update = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
                
            await asyncio.sleep(self.update_interval)
            
    def stop_collection(self):
        """Stop continuous data collection"""
        self.is_running = False
