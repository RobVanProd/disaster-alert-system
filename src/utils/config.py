from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # API Keys (these would be loaded from environment variables in production)
    WEATHER_API_KEY: str = ""
    SEISMIC_API_KEY: str = ""
    
    # Alert settings
    ALERT_THRESHOLD: float = 0.7
    WARNING_THRESHOLD: float = 0.4
    
    class Config:
        env_file = ".env"

settings = Settings()
