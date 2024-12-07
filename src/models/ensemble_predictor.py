from typing import Dict, Any, List
import numpy as np
from .base_model import BaseDisasterModel
from .earthquake_model import EarthquakeModel

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple disaster prediction models
    """
    
    def __init__(self):
        self.models: Dict[str, BaseDisasterModel] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all disaster prediction models"""
        # Add earthquake model with appropriate input dimensions
        # Input features: [latitude, longitude, depth, magnitude, time_of_day, day_of_week]
        earthquake_input_dim = 6
        self.models['earthquake'] = EarthquakeModel(input_dim=earthquake_input_dim)
        
        # TODO: Add more models for different disaster types
        # self.models['hurricane'] = HurricaneModel()
        # self.models['flood'] = FloodModel()
        # self.models['wildfire'] = WildfireModel()
    
    def train_all_models(self, training_data: Dict[str, Dict[str, Any]]) -> None:
        """Train all models with their respective training data"""
        for disaster_type, model in self.models.items():
            if disaster_type in training_data:
                model.train(training_data[disaster_type])
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using all available models
        """
        predictions = {}
        overall_risk_level = "LOW"
        confidence_scores = []
        potential_disasters = []
        
        for disaster_type, model in self.models.items():
            try:
                prediction = model.predict(input_data)
                predictions[disaster_type] = prediction
                
                # Update overall risk assessment
                if prediction['risk_level'] == "HIGH":
                    overall_risk_level = "HIGH"
                    potential_disasters.append(disaster_type)
                elif prediction['risk_level'] == "MEDIUM" and overall_risk_level != "HIGH":
                    overall_risk_level = "MEDIUM"
                    potential_disasters.append(disaster_type)
                
                confidence_scores.append(prediction['confidence'])
            except Exception as e:
                print(f"Error predicting {disaster_type}: {str(e)}")
        
        return {
            "location": {
                "latitude": input_data["latitude"],
                "longitude": input_data["longitude"],
                "region_name": input_data.get("region_name", "Unknown")
            },
            "risk_level": overall_risk_level,
            "potential_disasters": potential_disasters,
            "confidence": float(np.mean(confidence_scores)),
            "detailed_predictions": predictions
        }
    
    def evaluate_all(self, test_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models
        """
        evaluation_results = {}
        for disaster_type, model in self.models.items():
            if disaster_type in test_data:
                evaluation_results[disaster_type] = model.evaluate(test_data[disaster_type])
        return evaluation_results
