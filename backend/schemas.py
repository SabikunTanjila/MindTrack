"""Strict assessment validation with the dataset field names."""
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
