from setuptools import setup, find_packages

setup(
    name="disaster_alert_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "tensorflow>=2.16.1",
        "torch>=2.2.1",
        "transformers>=4.35.2",
        "aiohttp>=3.9.1",
        "dash>=2.14.1",
        "dash-bootstrap-components>=1.5.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
)
