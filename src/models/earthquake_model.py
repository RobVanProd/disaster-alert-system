"""
Earthquake prediction model
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)

class EarthquakeModel:
    """Neural network model for earthquake prediction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize the model"""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build the neural network model"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """Train the model"""
        return self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            **kwargs
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        loss, accuracy = self.model.evaluate(X, y)
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
