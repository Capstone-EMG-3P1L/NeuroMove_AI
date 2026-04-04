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
                startedAt=session.startedAt,
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
                canFinish = all(count > 0 for count in step_window_counts.values()) #기준 일단 임시로 정의
            ),
        )
    
    def update_calibration_step(self,request: CalibrationStepUpdateRequest,) -> CalibrationStepUpdateResponse:
        session = self.calibration_store.get_session(request.calibrationSessionId)
        if session is None:
            raise ValueError("calibration session not found")

        if session.status == CalibrationStatus.COMPLETED:
            raise ValueError("calibration session already completed")

        updated_session = self.calibration_store.update_step(
            request.calibrationSessionId,
            request.step,
        )
        if updated_session is None:
            raise ValueError("failed to update calibration step")

        return CalibrationStepUpdateResponse(
            success=True,
            message="calibration step updated",
            data=CalibrationStepUpdateData(
                calibrationSessionId=updated_session.calibrationSessionId,
                currentStep=updated_session.currentStep,
            ),
        )
    
    def append_calibration_data(self,request: CalibrationDataRequest,) -> CalibrationDataResponse:
        session = self.calibration_store.get_session(request.calibrationSessionId)
        if session is None:
            raise ValueError("calibration session not found")

        if session.status == CalibrationStatus.COMPLETED:
            raise ValueError("calibration session already completed")
        
        # 시퀀스 넘버가 이전보다 작거나 같으면 안됨 -> 증가만 하면 OK 
        # TODO:
        #실제 시스템은 “연속성 체크 + gap 감지” 로 변경하도록
        if (
            session.lastSequenceNumber is not None
            and request.sequenceNumber <= session.lastSequenceNumber
        ):
            raise ValueError("invalid sequence number")

        updated_session = self.calibration_store.append_raw_data(
            request.calibrationSessionId,
            request,
        )
        if updated_session is None:
            raise ValueError("failed to append calibration data")

        step_window_counts = self.calibration_store.get_step_window_counts(
            request.calibrationSessionId
        )
        if step_window_counts is None:
            raise ValueError("failed to load step window counts")

        return CalibrationDataResponse(
            success=True,
            message="calibration data appended",
            data=CalibrationDataResponseData(
                calibrationSessionId=updated_session.calibrationSessionId,
                currentStep=updated_session.currentStep,
                stepWindowCounts=step_window_counts,
            ),
        )
    
    def finish_calibration(self,request: CalibrationFinishRequest,) -> CalibrationFinishResponse:
        session = self.calibration_store.get_session(request.calibrationSessionId)
        if session is None:
            raise ValueError("calibration session not found")

        if session.status == CalibrationStatus.COMPLETED:
            raise ValueError("calibration session already completed")

        step_window_counts = self.calibration_store.get_step_window_counts(
            request.calibrationSessionId
        )
        if step_window_counts is None:
            raise ValueError("failed to load step window counts")

        can_finish = self._check_can_finish(step_window_counts)
        if not can_finish:
            raise ValueError("not enough calibration data to finish")

        completed_at = int(time.time() * 1000)

        # TODO:
        # 나중에 실제 signal processing / feature extraction / threshold 계산 로직으로 교체
        result = CalibrationResult(
            baseline=BaselineResult(
                ch1Mean=0.0,
                ch1Std=0.0,
                ch2Mean=0.0,
                ch2Std=0.0,
                ch3Mean=0.0,
                ch3Std=0.0,
            ),
            activationThreshold=0.0,
            intentThresholds={
                CalibrationStep.LEFT: 0.0,
                CalibrationStep.RIGHT: 0.0,
                CalibrationStep.STOP: 0.0,
            },
            fatigueBaseline=0.0,
            signalQuality=1.0,
        )

        completed_session = self.calibration_store.save_result(
            request.calibrationSessionId,
            result,
            completed_at,
        )
        if completed_session is None:
            raise ValueError("failed to save calibration result")

        return CalibrationFinishResponse(
            success=True,
            message="calibration finished",
            data=CalibrationFinishData(
                calibrationSessionId=completed_session.calibrationSessionId,
                userId=completed_session.userId,
                deviceId=completed_session.deviceId,
                result=completed_session.result,
                completedAt=completed_session.completedAt,
            ),
        )
        
    def _check_can_finish(self, step_counts: Dict[CalibrationStep, int]) -> bool:
        required_count = 5
        return all(count >= required_count for count in step_counts.values())