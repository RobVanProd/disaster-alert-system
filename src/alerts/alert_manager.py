"""
Alert Manager for handling disaster alerts and notifications
"""
import asyncio
from datetime import datetime
import logging
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    EARTHQUAKE = "earthquake"
    WEATHER = "weather"
    FLOOD = "flood"
    TSUNAMI = "tsunami"

class Alert(BaseModel):
    """Alert data model"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    location: Dict[str, float]
    timestamp: datetime
    metadata: Optional[Dict] = None
    
class AlertChannel(BaseModel):
    """Base class for alert channels"""
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through the channel"""
        raise NotImplementedError

class EmailAlertChannel(AlertChannel):
    """Email-based alert channel"""
    def __init__(self, smtp_config: Dict):
        self.smtp_config = smtp_config
        
    async def send_alert(self, alert: Alert) -> bool:
        # TODO: Implement email sending
        logger.info(f"Sending email alert: {alert.title}")
        return True

class SMSAlertChannel(AlertChannel):
    """SMS-based alert channel"""
    def __init__(self, sms_config: Dict):
        self.sms_config = sms_config
        
    async def send_alert(self, alert: Alert) -> bool:
        # TODO: Implement SMS sending
        logger.info(f"Sending SMS alert: {alert.title}")
        return True

class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel"""
    def __init__(self, webhook_urls: List[str]):
        self.webhook_urls = webhook_urls
        
    async def send_alert(self, alert: Alert) -> bool:
        # TODO: Implement webhook calls
        logger.info(f"Sending webhook alert: {alert.title}")
        return True

class AlertManager:
    """Manages alert generation and distribution"""
    
    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self._alert_queue = asyncio.Queue()
        
    def add_channel(self, channel: AlertChannel):
        """Add a new alert channel"""
        self.channels.append(channel)
        
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        location: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> Alert:
        """Create a new alert"""
        alert = Alert(
            id=f"{alert_type.value}-{datetime.utcnow().timestamp()}",
            type=alert_type,
            severity=severity,
            title=title,
            description=description,
            location=location,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        asyncio.create_task(self._alert_queue.put(alert))
        return alert
        
    async def process_alerts(self):
        """Process alerts from the queue"""
        while True:
            alert = await self._alert_queue.get()
            self.alert_history.append(alert)
            
            # Send alert through all channels
            tasks = [
                channel.send_alert(alert)
                for channel in self.channels
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for channel, result in zip(self.channels, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send alert through {channel.__class__.__name__}: {str(result)}")
                elif not result:
                    logger.warning(f"Alert not sent through {channel.__class__.__name__}")
                    
            self._alert_queue.task_done()
            
    def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get filtered alert history"""
        filtered_alerts = self.alert_history
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.type == alert_type]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
            
        return filtered_alerts
