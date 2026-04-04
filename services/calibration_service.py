from schemas.calibration_schema import *;
from storage.calibration_store import *;
import time
from typing import Dict


class CalibrationService:
    def __init__(self, calibration_store: CalibrationSessionStore):
        self.calibration_store = calibration_store

    def start_calibration(self, request: CalibrationStartRequest,) -> CalibrationStartResponse:
        if self.calibration_store.exists(request.calibrationSessionId):
            raise ValueError("calibration session already exists")

        started_at = int(time.time()*1000)
        
        session = CalibrationSession(
            calibrationSessionId=request.calibrationSessionId,
            userId=request.userId,
            deviceId=request.deviceId,
            currentStep=request.initialStep,
            startedAt=started_at,
        )

        self.calibration_store.create_session(session)

        return CalibrationStartResponse(
            success=True,
            message="calibration session created",
            data=CalibrationStartData(
                calibrationSessionId=session.calibrationSessionId,
                userId=session.userId,
                deviceId=session.deviceId,
                status=session.status,
                currentStep=session.currentStep,
                status=session.status,
            ),
        )

    def get_calibration_status(self,calibration_session_id: str,) -> CalibrationStatusResponse:
        session = self.calibration_store.get_session(calibration_session_id)
        if session is None:
            raise ValueError("calibration session not found")

        step_window_counts = self.calibration_store.get_step_window_counts(
            calibration_session_id
        )

        return CalibrationStatusResponse(
            success=True,
            message="calibration status fetched",
            data=CalibrationStatusData(
                calibrationSessionId=session.calibrationSessionId,
                status=session.status,
                currentStep=session.currentStep,
                stepWindowCounts=step_window_counts,
                can_finish = all(count > 0 for count in step_window_counts.values()) #기준 일단 임시로 정의
            ),
        )