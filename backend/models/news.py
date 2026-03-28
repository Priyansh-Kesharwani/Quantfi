from typing import List, Dict, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


class NewsEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    event_type: str
    affected_assets: List[str]
    impact_scores: Dict[str, float]
    summary: Optional[str] = None
