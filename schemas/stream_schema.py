from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from storage.device_mode_registry import DeviceMode


# 공통 채널
class EMGChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channelIndex: int = Field(..., ge=0)
    samples: List[int] = Field(..., min_length=1)

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, samples: List[int]) -> List[int]:
        for sample in samples:
            if sample is None:
                raise ValueError("samples must not contain null values")
        return samples


# =========================
# 통합 EMG WebSocket 메시지
# =========================
# ESP32 한 보드에서 실시간으로 들어오는 EMG window 패킷.
# - calibration / session 모드 구분은 ESP32 가 하지 않는다.
# - 서버의 DeviceModeRegistry 가 deviceId 기준으로 현재 활성 모드를 판단해
#   calibration_store 또는 session_store 로 라우팅하고, 둘 다 아니면 무시한다.
class EmgWindowMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deviceId: str = Field(..., min_length=1, max_length=100)
    sequenceNumber: int = Field(..., ge=0)
    timestamp: int = Field(..., ge=0)
    samplingRate: int = Field(..., gt=0, le=5000)
    windowSize: int = Field(..., gt=0, le=5000)
    channels: List[EMGChannel] = Field(..., min_length=1, max_length=16)

    @field_validator("deviceId")
    @classmethod
    def validate_device_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("deviceId must not be blank")
        return value

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, channels: List[EMGChannel]) -> List[EMGChannel]:
        channel_indexes = [channel.channelIndex for channel in channels]

        if len(channel_indexes) != len(set(channel_indexes)):
            raise ValueError("channelIndex must be unique")

        sample_lengths = [len(channel.samples) for channel in channels]
        if len(set(sample_lengths)) != 1:
            raise ValueError("all channels must have the same number of samples")

        return channels

    @model_validator(mode="after")
    def validate_window_matches_samples(self):
        actual_size = len(self.channels[0].samples)

        if self.windowSize != actual_size:
            raise ValueError(
                f"windowSize({self.windowSize}) does not match actual sample size({actual_size})"
            )

        return self


class EmgWindowAckData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deviceId: str
    mode: DeviceMode
    activeId: Optional[str] = None
    acceptedSequenceNumber: int
    bufferedWindowCount: int


class EmgWindowAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["emg_window_ack"] = "emg_window_ack"
    success: bool
    message: str
    data: Optional[EmgWindowAckData] = None


# =========================
# 내부 추론 결과 Schema
# =========================
class InferenceResultSchema(BaseModel):
    """AI 모델 추론 결과를 내부 서비스 간 전달할 때 쓰는 schema."""

    model_config = ConfigDict(extra="forbid")

    predicted_intent: Literal[
        "LEFT",
        "RIGHT",
        "REST",
        "STOP",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    feature_vector: List[float]
    model_version: str


# =========================
# AI Server → Backend POST /api/ai/intent
# =========================
class BackendInferenceSchema(BaseModel):
    """추론 결과를 백엔드로 보낼 때 쓰는 request body."""

    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(..., min_length=1, max_length=100)
    sequenceNumber: int
    emgDeviceId: str
    timestamp: int = Field(..., ge=0)

    intent: Literal[
        "LEFT",
        "RIGHT",
        "REST",
        "STOP",
    ]

    confidence: float = Field(..., ge=0.0, le=1.0)
    fatigueScore: float = Field(..., ge=0.0, le=1.0)
    signalQuality: float = Field(..., ge=0.0, le=1.0)

    @field_validator("sessionId")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sessionId must not be blank")
        return v


class BackendCommandSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commandId: str = Field(..., min_length=1, max_length=100)
    command: Literal[
        "LEFT",
        "RIGHT",
        "REST",
        "STOP",
    ]
    speedLevel: int = Field(..., ge=0, le=10)
    issuedAt: str = Field(..., min_length=1)


class BackendInferenceResponseDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intentId: str = Field(..., min_length=1, max_length=100)
    sessionId: str = Field(..., min_length=1, max_length=100)
    accepted: bool
    riskScore: float = Field(..., ge=0.0, le=1.0)
    command: Optional[BackendCommandSchema] = None

