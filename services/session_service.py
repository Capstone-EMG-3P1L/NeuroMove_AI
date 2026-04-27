import time

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
        window: SessionWindow,
    ) -> Session:
        """
        WebSocket 으로 들어온 EMG window 한 개를 active session 에 누적.

        StreamService 가 deviceMode == SESSION 으로 라우팅한 후 호출하는,
        세션 도메인의 단일 ingest 엔트리포인트.

        - raw EMG 는 session_store._buffers (ring buffer) 에 저장되고,
          추후 추론 파이프라인이 get_recent_windows() 로 꺼내서 사용한다.
        - 검증 실패 시 ValueError (StreamService 가 catch 해 ack 로 변환).

        TODO: 일정 buffer 누적 시점에서 추론 파이프라인을 트리거한다.
              signal_processing -> feature -> inference -> backend POST /api/ai/intent
        """
        session = self.session_store.get_session(session_id)
        if session is None:
            raise ValueError("session not found")
        if session.status != SessionStatus.ACTIVE:
            raise ValueError("session is not active")
        if session.deviceId != device_id:
            raise ValueError("deviceId mismatch with active session")
        if (
            session.lastSequenceNumber is not None
            and window.sequenceNumber <= session.lastSequenceNumber
        ):
            raise ValueError("invalid sequence number")

        updated = self.session_store.append_window_data(session_id, window)
        if updated is None:
            raise ValueError("failed to append session window")

        # TODO: 추론(inference) 트리거 위치
        #   if self._should_trigger_inference(updated):
        #       recent = self.session_store.get_recent_windows(session_id, N)
        #       intent = self.inference_pipeline.run(updated, recent)
        #       self.session_store.update_last_intent(session_id, intent.intent)
        #       self.backend_client.send_intent(...)

        return updated
