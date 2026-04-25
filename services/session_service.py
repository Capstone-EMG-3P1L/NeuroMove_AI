import time
from typing import Optional

from schemas.session_schema import *;
from storage.session_store import *;
from storage.device_mode_registry import DeviceModeRegistry


class SessionService:
    def __init__(
        self,
        session_store: SessionStore,
        device_mode_registry: DeviceModeRegistry,
    ):
        self.session_store = session_store
        self.device_mode_registry = device_mode_registry

    def start_session(self, request: SessionStartRequest,) -> SessionStartResponse:
        if self.session_store.exists(request.sessionId):
            raise ValueError("session already exists")

        # 같은 deviceId 가 이미 calibration / 다른 session 중이면 거절
        self.device_mode_registry.set_session(
            request.deviceId,
            request.sessionId,
        )

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

        # session 끝났으니 device 를 IDLE 로 풀어준다.
        self.device_mode_registry.clear(ended_session.deviceId)

        return SessionEndResponse(
            success=True,
            message="session ended",
            data=SessionEndData(
                sessionId=ended_session.sessionId,
                status=ended_session.status,
                endedAt=ended_session.endedAt,
            ),
        )

    def append_window(
        self,
        session_id: str,
        device_id: str,
        sequence_number: int,
    ) -> Optional[Session]:
        """
        WebSocket 으로 들어온 EMG window 한 개를 active session 에 누적.

        - 추론(inference) 은 아직 미구현. 현재는 bufferedWindowCount 만 증가시킨다.
        - 추후 inference_service 가 붙으면 여기에서 호출하게 된다. (TODO)
        """
        session = self.session_store.get_session(session_id)
        if session is None:
            return None
        if session.status != SessionStatus.ACTIVE:
            return None
        if session.deviceId != device_id:
            return None

        # TODO: sequenceNumber 연속성/gap 감지
        return self.session_store.increment_window_count(
            session_id,
            sequence_number,
        )