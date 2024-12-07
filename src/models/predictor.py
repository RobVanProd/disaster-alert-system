from datetime import datetime
from typing import Dict, Any
from .ensemble_predictor import EnsemblePredictor
from ..utils.data_collector import DataCollector

class DisasterPredictor:
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.data_collector = DataCollector()
    
    def evaluate_risk(self, latitude: float, longitude: float, region: str) -> Dict[str, Any]:
        """
        Evaluate disaster risks for a given location
        """
        # Collect all necessary data
        input_data = self._collect_data(latitude, longitude)
        input_data['region_name'] = region
        
        # Get predictions from ensemble
        prediction = self.ensemble.predict(input_data)
        
        # Add timestamp to prediction
        prediction['timestamp'] = datetime.now().isoformat()
        
        return prediction
    
    def _collect_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Collect all necessary data for prediction
        """
        # Get environmental data
        env_data = self.data_collector.get_environmental_data(latitude, longitude)
        
        # Get seismic data
        seismic_data = self.data_collector.get_seismic_data(latitude, longitude)
        
        # Get weather data
        weather_data = self.data_collector.get_weather_data(latitude, longitude)
        
        # Combine all data
        return {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": datetime.now().isoformat(),
            "environmental_data": {
                "seismic": seismic_data,
                "weather": weather_data,
                **env_data
            },
            "historical_data": self._get_historical_data(latitude, longitude)
        }
    
    def _get_historical_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get historical disaster data for the location
        """
        # This would typically fetch data from a database or external API
        # For now, returning mock data
        return {
            "avg_magnitude": 3.5,
            "frequency": 12,  # Number of events per year
            "max_magnitude": 5.8
        }
