from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"
    EXPIRED = "EXPIRED"


class SessionStartRequest(BaseModel):
    sessionId: str
    userId: int
    deviceId: str


class SessionStartData(BaseModel):
    sessionId: str
    userId: int
    deviceId: str
    status: SessionStatus
    createdAt: int
    lastActivityAt: int


class SessionStartResponse(BaseModel):
    success: bool
    message: str
    data: SessionStartData


class SessionStatusData(BaseModel):
    sessionId: str
    userId: int
    deviceId: str
    status: SessionStatus
    createdAt: int
    lastActivityAt: int
    endedAt: Optional[int] = None


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