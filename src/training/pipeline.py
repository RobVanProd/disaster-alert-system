"""
Model Training Pipeline for Disaster Prediction Models
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from models.base_model import BaseDisasterModel
from models.earthquake_model import EarthquakeModel
from models.ensemble_predictor import EnsemblePredictor

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Pipeline for training and validating disaster prediction models"""
    
    def __init__(
        self,
        model_class: Type,
        data_path: Path,
        model_params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize the pipeline"""
        self.model_class = model_class
        self.data_path = data_path
        self.model_params = model_params
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess training data"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Load data based on file type
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
        logger.info(f"Loaded {len(df)} records")
        return df
        
    def preprocess_data(
        self,
        df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for training"""
        logger.info("Preprocessing data")
        
        # Split features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
        
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[tuple] = None,
        **kwargs
    ):
        """Train the model"""
        logger.info("Training model")
        
        # Initialize model
        self.model = self.model_class(**self.model_params)
        
        # Train model
        self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            **kwargs
        )
        
        logger.info("Model training completed")
        
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model performance"""
        logger.info("Evaluating model")
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return self.model.evaluate(X_test, y_test)
        
    def save_model(self, save_path: Path):
        """Save trained model and artifacts"""
        logger.info(f"Saving model to {save_path}")
        
        if self.model is None:
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.model.save(save_path)  # For Keras models
        
        # Save scaler
        scaler_path = save_path.parent / "scaler.pkl"
        pd.to_pickle(self.scaler, scaler_path)
        
        logger.info("Model and artifacts saved successfully")
        
    def load_model(self, load_path: Path):
        """Load a trained model"""
        logger.info(f"Loading model from {load_path}")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")
            
        # Initialize model
        self.model = self.model_class(**self.model_params)
        
        # Load weights
        self.model.model = tf.keras.models.load_model(load_path)
        
    @classmethod
    def load_trained_model(
        cls,
        model_path: Path,
        model_class: Type[BaseDisasterModel]
    ) -> tuple[BaseDisasterModel, StandardScaler]:
        """Load a trained model and its scaler"""
        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model
        model = model_class.load(model_path)
        
        # Load scaler
        scaler_path = model_path.parent / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
        scaler = pd.read_pickle(scaler_path)
        
        logger.info("Model and scaler loaded successfully")
        return model, scaler
        
class EnsembleTrainingPipeline:
    """Pipeline for training ensemble models"""
    
    def __init__(
        self,
        model_configs: List[Dict],
        ensemble_params: Optional[Dict] = None
    ):
        self.model_configs = model_configs
        self.ensemble_params = ensemble_params or {}
        self.model_pipelines = []
        self.ensemble = None
        
    def train_base_models(self, data_path: Path, **kwargs):
        """Train all base models"""
        logger.info("Training base models")
        
        for config in self.model_configs:
            pipeline = ModelTrainingPipeline(
                model_class=config['model_class'],
                data_path=data_path,
                model_params=config.get('model_params')
            )
            
            # Load and preprocess data
            df = pipeline.load_data()
            X_train, X_test, y_train, y_test = pipeline.preprocess_data(df)
            
            # Train model
            pipeline.train_model(X_train, y_train, (X_test, y_test), **kwargs)
            
            # Evaluate model
            metrics = pipeline.evaluate_model(X_test, y_test)
            logger.info(f"Model {config['model_class'].__name__} metrics: {metrics}")
            
            self.model_pipelines.append(pipeline)
            
    def create_ensemble(self):
        """Create ensemble from trained models"""
        logger.info("Creating ensemble model")
        
        if not self.model_pipelines:
            raise ValueError("No base models have been trained")
            
        models = [p.model for p in self.model_pipelines]
        self.ensemble = EnsemblePredictor(models, **self.ensemble_params)
        
    def save_ensemble(self, save_dir: Path):
        """Save ensemble model and all base models"""
        logger.info(f"Saving ensemble to {save_dir}")
        
        if self.ensemble is None:
            raise ValueError("No ensemble model to save")
            
        # Create directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for i, pipeline in enumerate(self.model_pipelines):
            model_path = save_dir / f"base_model_{i}"
            pipeline.save_model(model_path)
            
        # Save ensemble
        ensemble_path = save_dir / "ensemble_model"
        self.ensemble.save(ensemble_path)
        
        logger.info("Ensemble and base models saved successfully")
