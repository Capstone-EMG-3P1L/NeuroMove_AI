import time

from schemas.session_schema import *;
from storage.session_store import *;


class SessionService:
    def __init__(self, session_store: SessionStore):
        self.session_store = session_store

    def start_session(self, request: SessionStartRequest,) -> SessionStartResponse:
        if self.session_store.exists(request.sessionId):
            raise ValueError("session already exists")

        started_at = int(time.time() * 1000)

        session = Session(
            sessionId=request.sessionId,
            userId=request.userId,
            deviceId=request.deviceId,
            profileId=request.profileId,
            calibration=request.calibration,
            status=SessionStatus.ACTIVE,
            startedAt=started_at,
        )

        self.session_store.create_session(session)

        return SessionStartResponse(
            success=True,
            message="session created",
            data=SessionStartData(
                sessionId=session.sessionId,
                userId=session.userId,
                deviceId=session.deviceId,
                status=session.status,
                startedAt=session.startedAt,
            ),
        )

    def get_session_status(self, session_id: str,) -> SessionStatusResponse:
        session = self.session_store.get_session(session_id)
        if session is None:
            raise ValueError("session not found")

        return SessionStatusResponse(
            success=True,
            message="session status fetched",
            data=SessionStatusData(
                sessionId=session.sessionId,
                status=session.status,
                bufferedWindowCount=session.bufferedWindowCount,
                lastIntent=session.lastIntent,
            ),
        )

    def end_session(self, request: SessionEndRequest,) -> SessionEndResponse:
        session = self.session_store.get_session(request.sessionId)
        if session is None:
            raise ValueError("session not found")

        if session.status == SessionStatus.ENDED:
            raise ValueError("session already ended")

        ended_at = int(time.time() * 1000)

        ended_session = self.session_store.end_session(
            request.sessionId,
            ended_at,
        )
        if ended_session is None:
            raise ValueError("failed to end session")

        return SessionEndResponse(
            success=True,
            message="session ended",
            data=SessionEndData(
                sessionId=ended_session.sessionId,
                status=ended_session.status,
                endedAt=ended_session.endedAt,
            ),
        )