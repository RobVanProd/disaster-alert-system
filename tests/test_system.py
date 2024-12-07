"""
Integration tests for the Disaster Alert System
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collectors import SeismicDataCollector, WeatherDataCollector
from alerts.alert_manager import (
    AlertManager, AlertType, AlertSeverity,
    EmailAlertChannel, SMSAlertChannel, WebhookAlertChannel,
    SMTPConfig
)
from models.earthquake_model import EarthquakeModel
from training.pipeline import ModelTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@pytest.mark.asyncio
async def test_seismic_data_collection():
    """Test seismic data collection"""
    collector = SeismicDataCollector(update_interval=10)
    
    # Test single data fetch
    data = await collector.get_earthquake_data(
        start_time=datetime.now(UTC) - timedelta(days=1),
        min_magnitude=2.5
    )
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verify data structure
    event = data[0]
    assert 'id' in event
    assert 'magnitude' in event
    assert 'location' in event
    assert 'time' in event
    
    logger.info(f"Successfully fetched {len(data)} seismic events")
    return data

@pytest.mark.asyncio
async def test_weather_data_collection():
    """Test weather data collection"""
    api_key = os.getenv('OWM_API_KEY')
    logger.info(f"Testing weather data collection with API key: {api_key}")
    if not api_key:
        pytest.skip("OpenWeatherMap API key not found")

    collector = WeatherDataCollector(api_key=api_key)

    # Add test locations
    collector.add_location(
        latitude=34.0522,
        longitude=-118.2437,
        name="Los Angeles"
    )

    # Test data fetch
    data = await collector.get_weather_data(34.0522, -118.2437)
    logger.info(f"Received weather data: {data}")

    # If we get a 401 error, the API key might not be activated yet
    if not data:
        pytest.skip("Weather API returned no data. If using a new API key, it may take a few hours to activate.")

    assert isinstance(data, dict), "Weather data should be a dictionary"
    assert 'temperature' in data, "Weather data should contain temperature"
    assert 'humidity' in data, "Weather data should contain humidity"
    assert 'wind_speed' in data
    
    logger.info(f"Successfully fetched weather data for Los Angeles")
    return data

@pytest.mark.asyncio
async def test_alert_system():
    """Test alert system"""
    alert_manager = AlertManager()
    
    # Add test alert channels
    email_channel = EmailAlertChannel(
        config=SMTPConfig(
            smtp_server='test.smtp.com',
            smtp_port=587,
            smtp_username='test',
            smtp_password='test'
        )
    )
    sms_channel = SMSAlertChannel(api_key='test_key')
    webhook_channel = WebhookAlertChannel(webhook_url='http://test.com/webhook')
    
    alert_manager.add_channel(email_channel)
    alert_manager.add_channel(sms_channel)
    alert_manager.add_channel(webhook_channel)
    
    # Create test alert
    alert = alert_manager.create_alert(
        alert_type=AlertType.EARTHQUAKE,
        severity=AlertSeverity.HIGH,
        title="Test Earthquake Alert",
        description="Strong earthquake detected in test location",
        location={'latitude': 34.0522, 'longitude': -118.2437}
    )
    
    assert alert.type == AlertType.EARTHQUAKE
    assert alert.severity == AlertSeverity.HIGH
    
    # Process alert
    await alert_manager.process_alerts()
    
    # Check alert history
    history = alert_manager.get_alert_history(
        alert_type=AlertType.EARTHQUAKE,
        severity=AlertSeverity.HIGH
    )
    
    assert len(history) > 0
    assert history[0].id == alert.id
    
    logger.info("Successfully tested alert system")
    return alert

@pytest.mark.asyncio
async def test_model_training():
    """Test model training pipeline"""
    # Create test data directory
    data_dir = Path("tests/test_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy training data
    import numpy as np
    import pandas as pd
    
    # Generate synthetic earthquake data
    n_samples = 1000
    np.random.seed(42)
    
    data = {
        'magnitude': np.random.uniform(2.0, 8.0, n_samples),
        'depth': np.random.uniform(0, 100, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'target': np.random.randint(0, 2, n_samples)  # Binary classification
    }
    
    df = pd.DataFrame(data)
    data_path = data_dir / "test_earthquake_data.csv"
    df.to_csv(data_path, index=False)
    
    # Initialize training pipeline
    pipeline = ModelTrainingPipeline(
        model_class=EarthquakeModel,
        data_path=data_path,
        model_params={'input_dim': 4}
    )
    
    # Train model
    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(df)
    
    pipeline.train_model(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=2  # Small number for testing
    )
    
    # Evaluate model
    metrics = pipeline.evaluate_model(X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'loss' in metrics
    
    logger.info(f"Model training completed with metrics: {metrics}")
    return metrics

async def main():
    """Run all tests"""
    logger.info("Starting system tests...")
    
    try:
        # Test data collection
        seismic_data = await test_seismic_data_collection()
        weather_data = await test_weather_data_collection()
        
        # Test alert system
        alert = await test_alert_system()
        
        # Test model training
        metrics = await test_model_training()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
