from enum import Enum
from typing import Optional

from pydantic import BaseModel
from schemas.calibration_schema import CalibrationResult


class SessionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"
    FAILED = "FAILED"


class SessionStartRequest(BaseModel):
    sessionId: str
    userId: int
    deviceId: str
    profileId: str
    calibration: CalibrationResult


class SessionStartData(BaseModel):
    sessionId: str
    userId: int
    deviceId: str
    status: SessionStatus
    startedAt: int


class SessionStartResponse(BaseModel):
    success: bool
    message: str
    data: SessionStartData


class SessionStatusData(BaseModel):
    sessionId: str
    status: SessionStatus
    bufferedWindowCount: int
    lastIntent: Optional[str] = None


class SessionStatusResponse(BaseModel):
    success: bool
    message: str
    data: SessionStatusData


class SessionEndRequest(BaseModel):
    sessionId: str


class SessionEndData(BaseModel):
    sessionId: str
    status: SessionStatus
    endedAt: int


class SessionEndResponse(BaseModel):
    success: bool
    message: str
    data: SessionEndData