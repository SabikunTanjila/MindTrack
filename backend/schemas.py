"""Strict assessment validation with the dataset field names."""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.config import AGE_MAX, AGE_MIN


class Assessment(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    Age: int = Field(ge=AGE_MIN, le=AGE_MAX, strict=True)
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Daily_Unlocks: int = Field(ge=0, le=2000, strict=True)
    Study_Hours: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(ge=0, le=24)
    Academic_Level: str = Field(min_length=1, max_length=80, pattern=r'.*\S.*')
    Most_Used_Platform: str = Field(min_length=1, max_length=80, pattern=r'.*\S.*')
    Purpose_Of_Use: str = Field(min_length=1, max_length=80, pattern=r'.*\S.*')


class ActionItem(BaseModel):
    """A single prioritised, actionable reflection prompt."""
    category: str
    priority: Literal['High', 'Medium', 'Low']
    title: str
    reason: str
    description: str
    action: str
    source_feature: str | None
    observed_value: float | None
    threshold_value: float | None


class RecommendationResponse(BaseModel):
    """Personalised recommendations returned alongside the prediction."""
    predicted_risk: str
    threshold_source: str
    summary: str
    items: List[ActionItem]
    disclaimer: str
