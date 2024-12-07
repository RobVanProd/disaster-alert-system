# AI-Powered Disaster Early Warning System

This project aims to save lives by providing early warnings for natural disasters using artificial intelligence and multiple data sources.

## Features

- Real-time monitoring of seismic activity, weather patterns, and environmental sensors
- Machine learning models for disaster prediction
- Automated alert system for communities at risk
- Interactive dashboard for disaster monitoring
- API endpoints for integration with other warning systems

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/main.py
```

## Project Structure

- `src/`: Source code directory
  - `main.py`: Application entry point
  - `models/`: ML models for disaster prediction
  - `data/`: Data processing and management
  - `api/`: FastAPI routes and endpoints
  - `utils/`: Utility functions
- `tests/`: Unit and integration tests
- `data/`: Sample and historical disaster data

## Contributing

This is an open-source project aimed at helping humanity. Contributions are welcome!
