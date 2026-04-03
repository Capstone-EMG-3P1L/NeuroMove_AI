from schemas.calibration_schema import *;
from storage.calibration_store import *;

class CalibrationService:
    def __init__(self, calibration_store: CalibrationSessionStore):
        self.calibration_store = calibration_store

    def start_calibration(self, request: CalibrationStartRequest,) -> CalibrationStartResponse:
        if self.calibration_store.exists(request.calibrationSessionId):
            raise ValueError("calibration session already exists")

        session = CalibrationSession(
            calibrationSessionId=request.calibrationSessionId,
            userId=request.userId,
            deviceUuid=request.deviceUuid,
            currentStep=request.initialStep,
        )

        self.calibration_store.create_session(session)

        return CalibrationStartResponse(
            accepted=True,
            message="calibration session created",
            data=CalibrationStartData(
                calibrationSessionId=session.calibrationSessionId,
                currentStep=session.currentStep,
                status=session.status,
            ),
        )

    def get_calibration_status(self,calibration_session_id: str,) -> CalibrationStatusResponse:
        session = self.calibration_store.get_session(calibration_session_id)
        if session is None:
            raise ValueError("calibration session not found")

        step_sample_counts = self.calibration_store.get_step_sample_counts(
            calibration_session_id
        )

        return CalibrationStatusResponse(
            accepted=True,
            message="calibration status fetched",
            data=CalibrationStatusData(
                calibrationSessionId=session.calibrationSessionId,
                status=session.status,
                currentStep=session.currentStep,
                stepSampleCounts=step_sample_counts,
            ),
        )