"""Strict assessment validation with the original dataset field names."""
from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field


class Assessment(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    Age: int = Field(ge=18, le=24, strict=True)
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Daily_Unlocks: int = Field(ge=0, le=2000, strict=True)
    Study_Hours: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(ge=0, le=24)
    Academic_Level: Literal['High School', 'Undergraduate', 'Graduate']
    Most_Used_Platform: str = Field(min_length=1, max_length=80, pattern=r'.*\S.*')
    Purpose_Of_Use: Literal['Education', 'Entertainment', 'Networking', 'News']


class ActionItem(BaseModel):
    """A single prioritised, actionable reflection prompt."""
    category: str
    priority: Literal['High', 'Medium', 'Low']
    title: str
    description: str
    action: str


class RecommendationResponse(BaseModel):
    """Personalised recommendations returned alongside the prediction."""
    risk_level: str
    cluster_id: int
    cluster_insight: str
    overall_summary: str
    actions: List[ActionItem]
