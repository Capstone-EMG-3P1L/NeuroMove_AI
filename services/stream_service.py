"""
StreamService

ESP32 보드 한 개에서 WebSocket 으로 실시간 들어오는 EMG window 패킷을
DeviceModeRegistry 의 mode 에 따라 calibration / session 으로 라우팅한다.

흐름:
    ESP32 ──▶ /ai/stream/ws ──▶ StreamService.handle_emg_window
                                      │
                                      ├─ IDLE         → drop (요건 4)
                                      ├─ CALIBRATION  → CalibrationService.append_calibration_data
                                      └─ SESSION      → SessionService.append_window

이 서비스는 store 를 직접 호출하지 않는다. 도메인 검증 로직은 모두
SessionService / CalibrationService 안에서 일어나고, StreamService 는
- DeviceMode 라우팅
- 패킷 → 도메인 request 변환
- 활성 device 의 lastActiveAt 갱신 (lock 누수 방어)
- 도메인 ValueError → ack 변환
네 가지만 담당한다.
"""

from typing import Optional

from schemas.calibration_schema import (
    CalibrationDataRequest,
    ChannelWindow,
)
from schemas.session_schema import SessionWindow
from schemas.stream_schema import (
    EmgWindowMessage,
    EmgWindowAck,
    EmgWindowAckData,
)
from services.calibration_service import CalibrationService
from services.session_service import SessionService
from storage.device_mode_registry import (
    DeviceModeRegistry,
    DeviceMode,
)


class StreamService:
    def __init__(
        self,
        device_mode_registry: DeviceModeRegistry,
        calibration_service: CalibrationService,
        session_service: SessionService,
    ):
        self.device_mode_registry = device_mode_registry
        self.calibration_service = calibration_service
        self.session_service = session_service

    def handle_emg_window(self, msg: EmgWindowMessage) -> EmgWindowAck:
        state = self.device_mode_registry.get(msg.deviceId)

        if state.mode == DeviceMode.IDLE:
            # calibration 도, session 도 진행 중이 아님 → 무시 (요건 4)
            # IDLE 인 device 는 lock 자체가 없으므로 touch 하지 않는다.
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

        # 활성 device 가 살아있다는 신호 → lock 누수 방어 sweeper 가 안 풀도록 갱신.
        # (검증 실패해도 일단 device 자체는 살아있으므로 touch 한다.)
        self.device_mode_registry.touch(msg.deviceId)

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

        try:
            updated = self.calibration_service.append_calibration_data(cal_request)
        except ValueError as e:
            return EmgWindowAck(
                success=False,
                message=str(e),
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

        session_window = SessionWindow(
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

        try:
            updated = self.session_service.append_window(
                session_id=session_id,
                device_id=msg.deviceId,
                window=session_window,
            )
        except ValueError as e:
            return EmgWindowAck(
                success=False,
                message=str(e),
                data=None,
            )

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
