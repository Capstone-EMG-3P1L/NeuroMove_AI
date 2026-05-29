from collections import deque
from typing import Deque, Dict, List, Optional

from pydantic import BaseModel

from schemas.session_schema import SessionStatus, SessionWindow
from schemas.calibration_schema import CalibrationResult


# 한 세션이 메모리에 들고 있을 EMG window 의 최대 개수.
# 추론은 보통 최근 N(=수~수십) window 만 보면 되므로 ring buffer 로 잘라둔다.
DEFAULT_SESSION_BUFFER_MAXLEN = 200


class Session(BaseModel):
    sessionId: str
    userId: str
    deviceId: str
    profileId: str
    calibration: CalibrationResult
    status: SessionStatus = SessionStatus.ACTIVE
    startedAt: int
    endedAt: Optional[int] = None
    # 누적 카운터(monotonic). 실제 raw window 데이터는 SessionStore._buffers 에 분리 저장.
    bufferedWindowCount: int = 0
    lastSequenceNumber: Optional[int] = None
    lastIntent: Optional[str] = None


class SessionStore:
    """
    세션 메타데이터(_sessions) 와 raw EMG window ring buffer(_buffers) 를 함께 관리.

    Session pydantic 모델은 API 응답 직렬화에도 쓰이므로 raw EMG 가 들어가지 않도록
    분리해 두었다. 추론 파이프라인은 get_recent_windows() 로 buffer 를 조회한다.
    """

    # 서버 실행되면 sessions 딕셔너리 구조 생성
    # {
    #   "DRV-2001": Session(...),
    #   "DRV-2002": Session(...),
    # }
    def __init__(self, buffer_maxlen: int = DEFAULT_SESSION_BUFFER_MAXLEN):
        self._sessions: Dict[str, Session] = {}
        self._buffers: Dict[str, Deque[SessionWindow]] = {}
        self._buffer_maxlen = buffer_maxlen

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def create_session(self, session: Session) -> Session:
        self._sessions[session.sessionId] = session
        self._buffers[session.sessionId] = deque(maxlen=self._buffer_maxlen)
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
        # buffer 는 종료 시점에 정리. (메타데이터는 /status 로 조회 가능하도록 남겨둠)
        self._buffers.pop(session_id, None)
        return session

    def append_window_data(
        self,
        session_id: str,
        window: SessionWindow,
    ) -> Optional[Session]:
        """
        raw EMG window 한 개를 ring buffer 에 적재 + 메타데이터 갱신.
        SessionService.append_window 의 단일 진입점에서만 호출된다.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        buf = self._buffers.get(session_id)
        if buf is None:
            return None

        buf.append(window)
        session.bufferedWindowCount += 1
        session.lastSequenceNumber = window.sequenceNumber
        return session

    def get_recent_windows(
        self,
        session_id: str,
        count: int,
    ) -> List[SessionWindow]:
        """
        추론 파이프라인이 최근 N 개 window 를 가져갈 때 호출.
        buffer 가 없거나 비어있으면 빈 리스트.
        """
        buf = self._buffers.get(session_id)
        if not buf:
            return []
        if count <= 0:
            return []
        if count >= len(buf):
            return list(buf)
        return list(buf)[-count:]

    def update_last_intent(
        self,
        session_id: str,
        intent: str,
    ) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None

        session.lastIntent = intent
        return session

    # ----- 하위 호환 -----
    # 아직 데이터 없이 카운트만 바꾸고 싶은 경로용. 신규 코드는 append_window_data 사용 권장.
    def increment_window_count(
        self,
        session_id: str,
        sequence_number: int,
    ) -> Optional[Session]:
        session = self.get_session(session_id)
        if session is None:
            return None

        session.bufferedWindowCount += 1
        session.lastSequenceNumber = sequence_number
        return session
