import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Disaster Early Warning System",
    description="AI-powered system for predicting and alerting natural disasters",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
