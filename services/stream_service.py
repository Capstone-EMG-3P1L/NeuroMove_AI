"""
StreamService

ESP32 보드 한 개에서 WebSocket 으로 실시간 들어오는 EMG window 패킷을
DeviceModeRegistry 의 mode 에 따라 calibration / session 으로 라우팅한다.

흐름:
    ESP32 ──▶ /ai/stream/ws ──▶ StreamService.handle_emg_window
                                      │
                                      ├─ IDLE         → drop (요건 4)
                                      ├─ CALIBRATION  → calibration_store.append_raw_data
                                      └─ SESSION      → session_service.append_window

이 서비스는 feature / inference / signal_processing 서비스를 호출하지 않는다.
(추론 로직은 별도 모듈에서 추후 연결 예정 — 본 파일에서는 import 하지 않음.)
"""

from typing import Optional

from schemas.calibration_schema import (
    CalibrationDataRequest,
    ChannelWindow,
)
from schemas.stream_schema import (
    EmgWindowMessage,
    EmgWindowAck,
    EmgWindowAckData,
)
from storage.calibration_store import CalibrationSessionStore
from storage.session_store import SessionStore
from storage.device_mode_registry import (
    DeviceModeRegistry,
    DeviceMode,
)


class StreamService:
    def __init__(
        self,
        device_mode_registry: DeviceModeRegistry,
        calibration_store: CalibrationSessionStore,
        session_store: SessionStore,
    ):
        self.device_mode_registry = device_mode_registry
        self.calibration_store = calibration_store
        self.session_store = session_store

    def handle_emg_window(self, msg: EmgWindowMessage) -> EmgWindowAck:
        state = self.device_mode_registry.get(msg.deviceId)

        if state.mode == DeviceMode.IDLE:
            # calibration 도, session 도 진행 중이 아님 → 무시 (요건 4)
            return EmgWindowAck(
                success=True,
                message="device idle, dropped",
                data=EmgWindowAckData(
                    deviceId=msg.deviceId,
                    mode="IDLE",
                    activeId=None,
                    acceptedSequenceNumber=msg.sequenceNumber,
                    bufferedWindowCount=0,
                ),
            )

        if state.mode == DeviceMode.CALIBRATION:
            return self._route_to_calibration(msg, state.activeId)

        if state.mode == DeviceMode.SESSION:
            return self._route_to_session(msg, state.activeId)

        return EmgWindowAck(
            success=False,
            message=f"unknown device mode: {state.mode}",
            data=None,
        )

    # -----------------------
    # Calibration 라우팅
    # -----------------------
    def _route_to_calibration(
        self,
        msg: EmgWindowMessage,
        calibration_session_id: Optional[str],
    ) -> EmgWindowAck:
        if calibration_session_id is None:
            return EmgWindowAck(
                success=False,
                message="calibration mode but no active calibrationSessionId",
                data=None,
            )

        session = self.calibration_store.get_session(calibration_session_id)
        if session is None:
            return EmgWindowAck(
                success=False,
                message="calibration session not found",
                data=None,
            )

        if session.deviceId != msg.deviceId:
            return EmgWindowAck(
                success=False,
                message="deviceId mismatch with active calibration session",
                data=None,
            )

        # sequenceNumber 단조 증가 검증
        # TODO: 추후 “연속성 + gap 감지” 로 업그레이드
        if (
            session.lastSequenceNumber is not None
            and msg.sequenceNumber <= session.lastSequenceNumber
        ):
            return EmgWindowAck(
                success=False,
                message="invalid sequence number",
                data=None,
            )

        cal_request = CalibrationDataRequest(
            calibrationSessionId=calibration_session_id,
            deviceId=msg.deviceId,
            sequenceNumber=msg.sequenceNumber,
            timestamp=msg.timestamp,
            samplingRate=msg.samplingRate,
            windowSize=msg.windowSize,
            channels=[
                ChannelWindow(
                    channelIndex=ch.channelIndex,
                    samples=ch.samples,
                )
                for ch in msg.channels
            ],
        )

        updated = self.calibration_store.append_raw_data(
            calibration_session_id,
            cal_request,
        )
        if updated is None:
            return EmgWindowAck(
                success=False,
                message="failed to append calibration data",
                data=None,
            )

        buffered = sum(
            len(buf) for buf in updated.stepBuffers.values()
        )

        return EmgWindowAck(
            success=True,
            message="calibration window appended",
            data=EmgWindowAckData(
                deviceId=msg.deviceId,
                mode="CALIBRATION",
                activeId=calibration_session_id,
                acceptedSequenceNumber=msg.sequenceNumber,
                bufferedWindowCount=buffered,
            ),
        )

    # -----------------------
    # Session(driving) 라우팅
    # -----------------------
    def _route_to_session(
        self,
        msg: EmgWindowMessage,
        session_id: Optional[str],
    ) -> EmgWindowAck:
        if session_id is None:
            return EmgWindowAck(
                success=False,
                message="session mode but no active sessionId",
                data=None,
            )

        session = self.session_store.get_session(session_id)
        if session is None:
            return EmgWindowAck(
                success=False,
                message="session not found",
                data=None,
            )

        if session.deviceId != msg.deviceId:
            return EmgWindowAck(
                success=False,
                message="deviceId mismatch with active session",
                data=None,
            )

        # sequenceNumber 단조 증가 검증
        if (
            session.lastSequenceNumber is not None
            and msg.sequenceNumber <= session.lastSequenceNumber
        ):
            return EmgWindowAck(
                success=False,
                message="invalid sequence number",
                data=None,
            )

        updated = self.session_store.increment_window_count(
            session_id,
            msg.sequenceNumber,
        )
        if updated is None:
            return EmgWindowAck(
                success=False,
                message="failed to append session window",
                data=None,
            )

        # TODO: 추론(inference) 트리거 위치.
        #   - 추후 별도 inference 파이프라인이 붙으면 여기서 호출.
        #   - 현재 단계에서는 buffer 누적만 수행.

        return EmgWindowAck(
            success=True,
            message="session window appended",
            data=EmgWindowAckData(
                deviceId=msg.deviceId,
                mode="SESSION",
                activeId=session_id,
                acceptedSequenceNumber=msg.sequenceNumber,
                bufferedWindowCount=updated.bufferedWindowCount,
            ),
        )
