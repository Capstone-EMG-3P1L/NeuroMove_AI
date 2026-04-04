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
    #서버 실행되면 sessions 생성 -> 요청 들어오면 세션 저장됨
    def __init__(self):
        self._sessions: Dict[str, CalibrationSession] = {}

    def exists(self, calibration_session_id: str) -> bool:
        return calibration_session_id in self._sessions

    def create_session(self, session: CalibrationSession) -> CalibrationSession:
        self._sessions[session.calibrationSessionId] = session
        return session

    def get_session(self, calibration_session_id: str) -> Optional[CalibrationSession]:
        return self._sessions.get(calibration_session_id)

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

    def append_raw_data(
        self,
        calibration_session_id: str,
        sample: CalibrationWindowRequest,
    ) -> Optional[CalibrationSession]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        session.stepBuffers[session.currentStep].append(sample)

        if session.status == CalibrationStatus.READY:
            session.status = CalibrationStatus.RUNNING

        return session

    def get_step_window_counts(self, calibration_session_id: str) -> Optional[dict]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        return {
            step: len(session.stepBuffers[step])
            for step in CalibrationStep
        }

    def save_result(
        self,
        calibration_session_id: str,
        result: CalibrationResult,
    ) -> Optional[CalibrationSession]:
        session = self.get_session(calibration_session_id)
        if session is None:
            return None

        session.result = result
        session.status = CalibrationStatus.COMPLETED
        return session