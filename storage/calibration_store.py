from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.calibration_schema import (
    CalibrationRawSample,
    CalibrationResult,
    CalibrationStatus,
    CalibrationStep,
)


class CalibrationSession(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceUuid: str
    status: CalibrationStatus = CalibrationStatus.READY
    currentStep: CalibrationStep = CalibrationStep.REST
    stepBuffers: Dict[CalibrationStep, List[CalibrationRawSample]] = Field(
        default_factory=lambda: {
            CalibrationStep.REST: [],
            CalibrationStep.LEFT: [],
            CalibrationStep.RIGHT: [],
            CalibrationStep.STOP: [],
        }
    )
    result: Optional[CalibrationResult] = None