from typing import Dict, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


def _cfg():
    from backend.core.config import get_backend_config
    return get_backend_config()


class UserSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    default_dca_cadence: str = 'monthly'
    default_dca_amount: Optional[float] = None
    dip_alert_threshold: Optional[float] = None
    score_weights: Optional[Dict[str, float]] = None

    def model_post_init(self, __context) -> None:
        cfg = _cfg()
        if self.default_dca_amount is None:
            self.default_dca_amount = cfg.user_default_dca_amount
        if self.dip_alert_threshold is None:
            self.dip_alert_threshold = cfg.user_default_dip_alert_threshold
        if self.score_weights is None:
            self.score_weights = dict(cfg.default_score_weights)
