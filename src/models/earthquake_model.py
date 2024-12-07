import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from .base_model import BaseDisasterModel

class EarthquakeModel(BaseDisasterModel):
    def __init__(self):
        super().__init__("DeepEarthquake")
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the deep learning model architecture"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),  # Input features
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_data(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for earthquake prediction
        """
        features = []
        
        # Location features
        features.extend([
            data['latitude'] / 90.0,  # Normalize latitude
            data['longitude'] / 180.0,  # Normalize longitude
        ])
        
        # Seismic features
        seismic_data = data['environmental_data'].get('seismic', {})
        features.extend([
            seismic_data.get('recent_activity', 0) / 10.0,
            seismic_data.get('plate_stress', 0) / 100.0,
            seismic_data.get('depth', 0) / 700.0  # Normalize depth
        ])
        
        # Historical features
        historical = data['historical_data']
        features.extend([
            historical.get('avg_magnitude', 0) / 10.0,
            historical.get('frequency', 0) / 365.0,
            historical.get('max_magnitude', 0) / 10.0
        ])
        
        # Time-based features
        current_time = datetime.fromisoformat(data['timestamp'])
        features.extend([
            np.sin(2 * np.pi * current_time.hour / 24),
            np.cos(2 * np.pi * current_time.hour / 24)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the earthquake prediction model
        """
        X = np.array([self.preprocess_data(sample) for sample in training_data['samples']])
        y = np.array(training_data['labels'])
        
        self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        self.is_trained = True
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make earthquake predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data format")
        
        # Preprocess input
        X = self.preprocess_data(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)
        
        # Generate multiple predictions with dropout enabled
        predictions = []
        for _ in range(10):
            pred = self.model(X, training=True)
            predictions.append(float(pred[0][0]))
        
        return self.format_prediction(predictions)
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        X = np.array([self.preprocess_data(sample) for sample in test_data['samples']])
        y = np.array(test_data['labels'])
        
        results = self.model.evaluate(X, y)
        return {
            "loss": float(results[0]),
            "accuracy": float(results[1])
        }
    
    def _get_prediction_details(self, prediction: List[float]) -> Dict[str, Any]:
        """
        Get detailed earthquake prediction information
        """
        base_details = super()._get_prediction_details(prediction)
        
        # Add earthquake-specific details
        mean_prediction = float(np.mean(prediction))
        base_details.update({
            "estimated_magnitude": self._estimate_magnitude(mean_prediction),
            "probability_within_24h": mean_prediction,
            "uncertainty": float(np.std(prediction))
        })
        
        return base_details
    
    def _estimate_magnitude(self, probability: float) -> float:
        """
        Estimate potential earthquake magnitude based on prediction probability
        """
        # This is a simplified estimation - in reality, would need more sophisticated calculation
        return 4.0 + (probability * 4.0)  # Estimates between 4.0 and 8.0 magnitude
