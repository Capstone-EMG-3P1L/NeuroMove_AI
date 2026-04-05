from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.calibration_schema import (
    CalibrationWindowRequest,
    CalibrationResult,
    CalibrationStatus,
    CalibrationStep,
)


class CalibrationSession(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceId: str
    status: CalibrationStatus = CalibrationStatus.READY
    currentStep: CalibrationStep = CalibrationStep.REST
    stepBuffers: Dict[CalibrationStep, List[CalibrationWindowRequest]] = Field(
        default_factory=lambda: {
            CalibrationStep.REST: [],
            CalibrationStep.LEFT: [],
            CalibrationStep.RIGHT: [],
            CalibrationStep.STOP: [],
        }
    )
    lastSequenceNumber: Optional[int] = None
    startedAt: int
    completedAt: Optional[int] = None
    result: Optional[CalibrationResult] = None

    
class CalibrationSessionStore:
    #서버 실행되면 sessions 딕셔너리 구조 생성
    # {
    # "CAL-1001": CalibrationSession(...),
    # "CAL-1002": CalibrationSession(...),
    # }   
    def __init__(self):
        self._sessions: Dict[str, CalibrationSession] = {}

    def exists(self, calibration_session_id: str) -> bool:
        return calibration_session_id in self._sessions

    # 새 세션을 저장소에 등록하는 함수
    def create_session(self, session: CalibrationSession) -> CalibrationSession:
        self._sessions[session.calibrationSessionId] = session
        return session

    # 저장된 세션을 꺼내오는 함수
    def get_session(self, calibration_session_id: str) -> Optional[CalibrationSession]:
        return self._sessions.get(calibration_session_id)

    # 새로운 step 으로 바꾸는 함수
    def update_step(
        self,
        calibration_session_id: str,
        step: CalibrationStep,
    ) -> Optional[CalibrationSession]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        session.currentStep = step
        return session

    # EMG 센서 데이터 버퍼에 저장
    def append_raw_data(
        self,
        calibration_session_id: str,
        sample: CalibrationWindowRequest,
    ) -> Optional[CalibrationSession]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None
        
        if session.deviceId != sample.deviceId:
            return None

        session.stepBuffers[session.currentStep].append(sample)
        session.lastSequenceNumber = sample.sequenceNumber

        if session.status == CalibrationStatus.READY:
            session.status = CalibrationStatus.RUNNING

        return session

    def get_step_window_counts(self, calibration_session_id: str) -> Optional[Dict[CalibrationStep, int]]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        return {
            step: len(session.stepBuffers[step])
            for step in CalibrationStep
        }

    # 최종 분석 결과를 세션에 저장
    def save_result(
        self,
        calibration_session_id: str,
        result: CalibrationResult,
        completed_at: int,
    ) -> Optional[CalibrationSession]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        session.result = result
        session.status = CalibrationStatus.COMPLETED
        session.completedAt = completed_at
        return session