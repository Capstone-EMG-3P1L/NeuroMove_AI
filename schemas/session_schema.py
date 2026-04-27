from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from schemas.calibration_schema import CalibrationResult, ChannelWindow


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


# WebSocket 으로 들어온 EmgWindowMessage 를 StreamService 에서 이 형태로 변환해서
# SessionService.append_window 로 넘겨준다.
# (CalibrationDataRequest 의 session 도메인 카운터파트)
class SessionWindow(BaseModel):
    sequenceNumber: int
    timestamp: int
    samplingRate: int
    windowSize: int
    channels: List[ChannelWindow]
