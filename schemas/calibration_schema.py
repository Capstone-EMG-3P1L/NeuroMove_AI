from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CalibrationStep(str, Enum):
    REST = "REST"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STOP = "STOP"


class CalibrationStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChannelWindow(BaseModel):
    channelIndex: int = Field(..., description="Channel index (0,1,2)")
    samples: List[int] = Field(..., description="Raw EMG samples for the channel")


class CalibrationWindowRequest(BaseModel):
    calibrationSessionId: str
    deviceId: str
    sequenceNumber: int
    timestamp: int
    samplingRate: int
    windowSize: int
    channels: List[ChannelWindow]

#Calibration 측정 시작
class CalibrationStartRequest(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceId: str
    initialStep: CalibrationStep = CalibrationStep.REST


class CalibrationStartData(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceId: str
    status: CalibrationStatus
    currentStep: CalibrationStep
    startedAt: int


class CalibrationStartResponse(BaseModel):
    success: bool
    message: str
    data: CalibrationStartData

# Calibration 단계 변경
class CalibrationStepUpdateRequest(BaseModel):
    calibrationSessionId: str
    step: CalibrationStep


class CalibrationStepUpdateData(BaseModel):
    calibrationSessionId: str
    currentStep: CalibrationStep


class CalibrationStepUpdateResponse(BaseModel):
    success: bool
    message: str
    data: CalibrationStepUpdateData

# EMG 센서에 측정되는 값 받기(esp32 보드)
class CalibrationDataRequest(BaseModel):
    calibrationSessionId: str
    deviceId: str
    sequenceNumber: int
    timestamp: int
    samplingRate: int
    windowSize: int
    channels: List[ChannelWindow]

    
class CalibrationDataResponseData(BaseModel):
    calibrationSessionId: str
    currentStep: CalibrationStep
    stepWindowCounts: Dict[CalibrationStep, int]

class CalibrationDataResponse(BaseModel):
    success: bool
    message: str
    data: CalibrationDataResponseData

# 백엔드한테 현재 진행 어느정도 됐는지 알리는 형식
class CalibrationStatusData(BaseModel):
    calibrationSessionId: str
    status: CalibrationStatus
    currentStep: CalibrationStep
    stepWindowCounts: Dict[CalibrationStep, int]
    canFinish: bool


class CalibrationStatusResponse(BaseModel):
    success: bool
    message: str
    data: CalibrationStatusData

# Calibration 종료 후 분석 결과
class BaselineResult(BaseModel):
    ch1Mean: float
    ch1Std: float
    ch2Mean: float
    ch2Std: float
    ch3Mean: float
    ch3Std: float


class CalibrationResult(BaseModel):
    baseline: BaselineResult
    activationThreshold: float
    intentThresholds: Dict[CalibrationStep, float]
    fatigueBaseline: float
    signalQuality: float


class CalibrationFinishRequest(BaseModel):
    calibrationSessionId: str


class CalibrationFinishData(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceId: str
    result: CalibrationResult
    completedAt: int


class CalibrationFinishResponse(BaseModel):
    success: bool
    message: str
    data: CalibrationFinishData