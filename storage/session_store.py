from typing import Dict, Optional
from pydantic import BaseModel

from schemas.session_schema import SessionStatus
from schemas.calibration_schema import CalibrationResult


class Session(BaseModel):
    sessionId: str
    userId: int
    deviceId: str
    profileId: str
    calibration: CalibrationResult
    status: SessionStatus = SessionStatus.ACTIVE
    startedAt: int
    endedAt: Optional[int] = None
    bufferedWindowCount: int = 0
    lastIntent: Optional[str] = None


class SessionStore:
    # 서버 실행되면 sessions 딕셔너리 구조 생성
    # {
    #   "DRV-2001": Session(...),
    #   "DRV-2002": Session(...),
    # }
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def create_session(self, session: Session) -> Session:
        self._sessions[session.sessionId] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
    ) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None

        session.status = status
        return session

    def end_session(
        self,
        session_id: str,
        ended_at: int,
    ) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None

        session.status = SessionStatus.ENDED
        session.endedAt = ended_at
        return session