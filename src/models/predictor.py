"""
Disaster prediction module using ensemble methods
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from utils.data_collector import DataCollector
from models.ensemble_predictor import EnsemblePredictor

class DisasterPredictor:
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.data_collector = DataCollector()
        self.monitored_regions = []
    
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
    
    def get_predictions(self) -> List[Dict[str, Any]]:
        """
        Get predictions for all monitored regions
        """
        predictions = []
        current_time = datetime.now()
        
        for region in self.monitored_regions:
            # Get current risk levels
            risk = self.evaluate_risk(
                region['latitude'],
                region['longitude'],
                region['name']
            )
            
            # Generate time series predictions
            for hours in range(24):
                prediction_time = current_time + timedelta(hours=hours)
                predictions.append({
                    'time': prediction_time,
                    'latitude': region['latitude'],
                    'longitude': region['longitude'],
                    'region': region['name'],
                    'disaster_type': risk['primary_risk'],
                    'risk_level': risk['risk_level'] * (1 + np.random.normal(0, 0.1)),
                    'description': risk['description']
                })
        
        return predictions
    
    def get_risk_areas(self) -> List[Dict[str, Any]]:
        """
        Get current high-risk areas
        """
        risk_areas = []
        
        for region in self.monitored_regions:
            risk = self.evaluate_risk(
                region['latitude'],
                region['longitude'],
                region['name']
            )
            
            if risk['risk_level'] > 3:  # Only include areas with significant risk
                risk_areas.append({
                    'latitude': region['latitude'],
                    'longitude': region['longitude'],
                    'region': region['name'],
                    'risk_level': risk['risk_level'],
                    'disaster_type': risk['primary_risk'],
                    'description': risk['description']
                })
        
        return risk_areas
    
    def get_recommended_actions(self) -> List[Dict[str, Any]]:
        """
        Get recommended actions based on current predictions
        """
        actions = []
        high_risk_areas = [area for area in self.get_risk_areas() if area['risk_level'] > 7]
        medium_risk_areas = [area for area in self.get_risk_areas() if 4 <= area['risk_level'] <= 7]
        
        # High risk recommendations
        for area in high_risk_areas:
            actions.append({
                'description': f"Immediate evacuation recommended for {area['region']} due to {area['disaster_type']} risk",
                'priority': 'danger',
                'region': area['region']
            })
        
        # Medium risk recommendations
        for area in medium_risk_areas:
            actions.append({
                'description': f"Prepare emergency supplies and monitor updates for {area['region']}",
                'priority': 'warning',
                'region': area['region']
            })
        
        # General recommendations
        if not actions:
            actions.append({
                'description': "No immediate actions required. Continue monitoring.",
                'priority': 'info',
                'region': 'All regions'
            })
        
        return actions
    
    def add_monitored_region(self, latitude: float, longitude: float, name: str):
        """
        Add a region to monitor
        """
        self.monitored_regions.append({
            'latitude': latitude,
            'longitude': longitude,
            'name': name
        })
    
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
            "max_magnitude": 5.8,
            "historical_events": [
                {
                    "type": "earthquake",
                    "magnitude": 4.2,
                    "date": "2023-12-01",
                    "damage_level": "moderate"
                },
                {
                    "type": "flood",
                    "severity": "high",
                    "date": "2023-11-15",
                    "damage_level": "severe"
                }
            ]
        }
