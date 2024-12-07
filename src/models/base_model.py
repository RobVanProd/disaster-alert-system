from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseDisasterModel(ABC):
    """Base class for all disaster prediction models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def preprocess_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for model"""
        pass
    
    @abstractmethod
    def train(self, training_data: Dict[str, Any]) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = [
            "latitude",
            "longitude",
            "timestamp",
            "environmental_data",
            "historical_data"
        ]
        return all(field in input_data for field in required_fields)
    
    def format_prediction(self, raw_prediction: Any) -> Dict[str, Any]:
        """Format model predictions into standardized output"""
        return {
            "risk_level": self._calculate_risk_level(raw_prediction),
            "confidence": float(np.mean(raw_prediction)),
            "details": self._get_prediction_details(raw_prediction)
        }
    
    def _calculate_risk_level(self, prediction: Any) -> str:
        """Calculate risk level from raw prediction"""
        risk_score = float(np.mean(prediction))
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _get_prediction_details(self, prediction: Any) -> Dict[str, Any]:
        """Get detailed prediction information"""
        return {
            "raw_score": float(np.mean(prediction)),
            "model_name": self.model_name,
            "confidence_interval": self._calculate_confidence_interval(prediction)
        }
    
    def _calculate_confidence_interval(self, prediction: Any) -> List[float]:
        """Calculate confidence interval for prediction"""
        mean = float(np.mean(prediction))
        std = float(np.std(prediction))
        return [mean - 1.96 * std, mean + 1.96 * std]
