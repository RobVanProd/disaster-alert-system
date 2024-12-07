from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from models.predictor import DisasterPredictor
from utils.alerts import AlertSystem

router = APIRouter()
predictor = DisasterPredictor()
alert_system = AlertSystem()

class Location(BaseModel):
    latitude: float
    longitude: float
    region_name: str

class RiskAssessment(BaseModel):
    location: Location
    risk_level: str
    potential_disasters: List[str]
    confidence: float
    timestamp: datetime

@router.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {"status": "online", "message": "Disaster Early Warning System API is running"}

@router.post("/assess-risk", response_model=RiskAssessment)
async def assess_risk(location: Location):
    """
    Assess disaster risks for a given location using AI models
    """
    try:
        assessment = predictor.evaluate_risk(
            latitude=location.latitude,
            longitude=location.longitude,
            region=location.region_name
        )
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alert")
async def send_alert(location: Location, risk_assessment: RiskAssessment):
    """
    Send alerts to relevant authorities and communities
    """
    try:
        alert_system.dispatch_alert(location.dict(), risk_assessment.dict())
        return {"status": "success", "message": "Alert dispatched successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active-warnings")
async def get_active_warnings(region: Optional[str] = None):
    """
    Get all active warnings for a region or globally
    """
    try:
        warnings = alert_system.get_active_warnings(region)
        return {"status": "success", "warnings": warnings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
