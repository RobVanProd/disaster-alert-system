import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from utils.data_collector import DataCollector

class DisasterPredictor:
    def __init__(self):
        self.data_collector = DataCollector()
        self.model = RandomForestClassifier()
        self.disaster_types = [
            "earthquake",
            "flood",
            "hurricane",
            "wildfire",
            "tsunami"
        ]
        
    def collect_environmental_data(self, latitude: float, longitude: float) -> Dict:
        """
        Collect real-time environmental data for the specified location
        """
        return self.data_collector.get_environmental_data(latitude, longitude)
    
    def analyze_seismic_activity(self, latitude: float, longitude: float) -> Dict:
        """
        Analyze recent seismic activity in the region
        """
        return self.data_collector.get_seismic_data(latitude, longitude)
    
    def analyze_weather_patterns(self, latitude: float, longitude: float) -> Dict:
        """
        Analyze weather patterns and forecasts
        """
        return self.data_collector.get_weather_data(latitude, longitude)
    
    def evaluate_risk(self, latitude: float, longitude: float, region: str) -> Dict:
        """
        Evaluate disaster risks based on collected data
        """
        # Collect all relevant data
        env_data = self.collect_environmental_data(latitude, longitude)
        seismic_data = self.analyze_seismic_activity(latitude, longitude)
        weather_data = self.analyze_weather_patterns(latitude, longitude)
        
        # Combine data for prediction
        features = self._prepare_features(env_data, seismic_data, weather_data)
        
        # Make prediction
        risk_scores = self._predict_risks(features)
        
        # Prepare risk assessment
        assessment = {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "region_name": region
            },
            "risk_level": self._determine_risk_level(risk_scores),
            "potential_disasters": self._identify_potential_disasters(risk_scores),
            "confidence": float(np.mean([score[1] for score in risk_scores])),
            "timestamp": datetime.now()
        }
        
        return assessment
    
    def _prepare_features(self, env_data: Dict, seismic_data: Dict, weather_data: Dict) -> np.ndarray:
        """
        Prepare feature vector for model prediction
        """
        # Combine all data into a feature vector
        # This is a simplified version - in reality, we would need more sophisticated feature engineering
        features = []
        features.extend([
            env_data.get("temperature", 0),
            env_data.get("humidity", 0),
            env_data.get("pressure", 0),
            seismic_data.get("magnitude", 0),
            seismic_data.get("depth", 0),
            weather_data.get("wind_speed", 0),
            weather_data.get("precipitation", 0)
        ])
        return np.array(features).reshape(1, -1)
    
    def _predict_risks(self, features: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict risk scores for each disaster type
        """
        # This is a simplified version - in reality, we would use more sophisticated models
        risk_scores = []
        for disaster_type in self.disaster_types:
            # Simulate prediction scores - in reality, this would use trained models
            score = np.random.random()  # Replace with actual model prediction
            risk_scores.append((disaster_type, score))
        return risk_scores
    
    def _determine_risk_level(self, risk_scores: List[Tuple[str, float]]) -> str:
        """
        Determine overall risk level based on individual risk scores
        """
        max_score = max([score[1] for score in risk_scores])
        if max_score > 0.7:
            return "HIGH"
        elif max_score > 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _identify_potential_disasters(self, risk_scores: List[Tuple[str, float]]) -> List[str]:
        """
        Identify potential disasters based on risk scores
        """
        return [disaster for disaster, score in risk_scores if score > 0.4]
