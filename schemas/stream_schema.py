from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class EMGChannelData(BaseModel):
    
    # 채널 한 개의 EMG 샘플 데이터
    # 예: channel_index=0, samples=[123, 118, 130, ...]
    
    channel_index: int = Field(..., ge=0, description="EMG 채널 인덱스",alias="channelIndex")
    samples: List[int] = Field(
        ...,
        min_length=1,
        description="해당 채널에서 수집된 raw EMG 샘플 배열"
    )

    # samples 값이 정상인지 추가 검사
    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v: List[float]) -> List[float]:

        for sample in v:
            if sample is None:
                raise ValueError("samples must not contain null values")
        return v


class StreamRequestSchema(BaseModel):
    
    # ESP32 -> AI Server 로 들어오는 실시간 스트림 입력 schema
    
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="세션 식별자",
        alias="sessionId"
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="스트림 패킷 순서 번호",
        alias="sequenceNumber"
    )
    timestamp: int = Field(
        ...,
        ge=0,
        description="클라이언트 기준 전송 시각 (epoch ms)",
        alias="timestamp"
    )
    sampling_rate: int = Field(
        ...,
        gt=0,
        le=5000,
        description="EMG 샘플링 레이트(Hz)",
        alias="samplingRate"
    )
    window_size: int = Field(
        ...,
        gt=0,
        le=5000,
        description="현재 요청에 포함된 샘플 수",
        alias="windowSize"
    )
    channels: List[EMGChannelData] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="채널별 EMG 데이터 목록",
        alias="channels"
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id must not be blank")
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: List[EMGChannelData]) -> List[EMGChannelData]:
        if len(v) == 0:
            raise ValueError("channels must not be empty")

        # channels 안에는 각 채널이 한 번씩만 나와야 한다.
        channel_indexes = [channel.channel_index for channel in v]
        if len(channel_indexes) != len(set(channel_indexes)):
            raise ValueError("channel_index must be unique")

        sample_lengths = [len(channel.samples) for channel in v]
        if len(set(sample_lengths)) != 1:
            raise ValueError("all channels must have the same number of samples")

        return v

    @model_validator(mode="after")
    def validate_window_matches_samples(self):
        if not self.channels:
            raise ValueError("channels must not be empty")

        actual_size = len(self.channels[0].samples)
        if self.window_size != actual_size:
            raise ValueError(
                f"window_size({self.window_size}) does not match "
                f"actual sample size({actual_size})"
            )
        return self


class InferenceResultSchema(BaseModel):
    # AI 추론 결과 서버 내부 사용 용도
    
    predicted_intent: Literal[
        "LEFT",
        "RIGHT",
        "FORWARD",
        "BACKWARD",
        "STOP",
        "UNKNOWN"
    ] = Field(..., description="예측된 사용자 의도")

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="예측 confidence score"
    )

    feature_vector: Optional[List[float]] = Field(
        default=None,
        description="추론에 사용된 feature vector (디버깅/개발용)"
    )

    model_version: Optional[str] = Field(
        default=None,
        description="사용된 모델 버전"
    )


class StreamAckDataSchema(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    device_id: str = Field(..., alias="deviceId")
    accepted_sequence_number: int = Field(..., alias="acceptedSequenceNumber")
    buffered_window_count: int = Field(..., alias="bufferedWindowCount")
    inference_triggered: bool = Field(..., alias="inferenceTriggered")

class StreamAckResponseSchema(BaseModel):
    success: bool
    message: str
    data: StreamAckDataSchema | None = None
    
    
class BackendInferenceSchema(BaseModel):
    
    # AI Server → Backend 전달용 추론 결과 schema
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(..., min_length=1, max_length=100)
    sequenceNumber : int
    emgDeviceId : str
    timestamp: int = Field(..., ge=0)

    intent: Literal[
        "LEFT",
        "RIGHT",
        "FORWARD",
        "BACKWARD",
        "STOP",
        "UNKNOWN"
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
        "FORWARD",
        "BACKWARD",
        "STOP",
        "UNKNOWN"
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


class BackendInferenceResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    code: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=500)
    data: Optional[BackendInferenceResponseDataSchema] = None