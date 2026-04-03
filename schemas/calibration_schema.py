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


class ChannelValues(BaseModel):
    ch1: int = Field(..., description="Raw EMG value for channel 1")
    ch2: int = Field(..., description="Raw EMG value for channel 2")
    ch3: int = Field(..., description="Raw EMG value for channel 3")


class CalibrationRawSample(BaseModel):
    timestamp: int = Field(..., description="Measurement timestamp")
    sequence: int = Field(..., description="Sequence number for packet ordering")
    channels: ChannelValues

#Calibration 측정 시작
class CalibrationStartRequest(BaseModel):
    calibrationSessionId: str
    userId: int
    deviceUuid: str
    initialStep: CalibrationStep = CalibrationStep.REST


class CalibrationStartData(BaseModel):
    calibrationSessionId: str
    currentStep: CalibrationStep
    status: CalibrationStatus


class CalibrationStartResponse(BaseModel):
    accepted: bool
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
    accepted: bool
    message: str
    data: CalibrationStepUpdateData

# EMG 센서에 측정되는 값 받기(esp32 보드)
class CalibrationDataRequest(BaseModel):
    calibrationSessionId: str
    timestamp: int
    sequence: int
    channels: ChannelValues


class CalibrationDataResponseData(BaseModel):
    calibrationSessionId: str
    currentStep: CalibrationStep
    bufferedCount: int


class CalibrationDataResponse(BaseModel):
    accepted: bool
    message: str
    data: CalibrationDataResponseData

# 백엔드한테 현재 진행 어느정도 됐는지 알리는 형식
class CalibrationStatusData(BaseModel):
    calibrationSessionId: str
    status: CalibrationStatus
    currentStep: CalibrationStep
    stepSampleCounts: Dict[CalibrationStep, int]


class CalibrationStatusResponse(BaseModel):
    accepted: bool
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
    status: CalibrationStatus
    result: CalibrationResult


class CalibrationFinishResponse(BaseModel):
    accepted: bool
    message: str
    data: CalibrationFinishData