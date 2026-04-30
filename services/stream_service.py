"""
StreamService

ESP32 보드 한 개에서 WebSocket 으로 실시간 들어오는 EMG window 패킷을
DeviceModeRegistry 의 mode 에 따라 calibration / session 으로 라우팅한다.

흐름:
    ESP32 ──▶ /ai/stream/ws ──▶ StreamService.handle_emg_window
                                      │
                                      ├─ IDLE         → drop
                                      ├─ CALIBRATION  → CalibrationService.append_calibration_data
                                      └─ SESSION      → SessionService.append_window

이 서비스는 store 를 직접 호출하지 않는다. 도메인 검증 로직은 모두
SessionService / CalibrationService 안에서 일어나고, StreamService 는
- DeviceMode 라우팅
- 패킷 → 도메인 request 변환
- 활성 device 의 lastActiveAt 갱신
- 도메인 ValueError → ack 변환
- inference payload 생성
- 백엔드 전송 queue 적재
를 담당한다.
"""

import time
from queue import Empty, Full, Queue
from threading import Thread
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
    BackendInferenceSchema,
)
from services.backend_service import BackendService
from services.calibration_service import CalibrationService
from services.session_service import SessionService
from services.signal_processing_service import preprocess_channels
from services.feature_service import extract_feature_vector
from services.inference_service import predict_intent
from services.signal_metric_service import calculate_signal_metrics
from storage.device_mode_registry import (
    DeviceModeRegistry,
    DeviceMode,
)


INFERENCE_MIN_WINDOW_COUNT = 5
INFERENCE_WINDOW_COUNT = 5
INFERENCE_INTERVAL = 5

BACKEND_QUEUE_MAX_SIZE = 100
BACKEND_SEND_RETRY_COUNT = 3
BACKEND_SEND_RETRY_DELAY_SECONDS = 1


class StreamService:
    def __init__(
        self,
        device_mode_registry: DeviceModeRegistry,
        calibration_service: CalibrationService,
        session_service: SessionService,
        backend_service: BackendService,
    ):
        self.device_mode_registry = device_mode_registry
        self.calibration_service = calibration_service
        self.session_service = session_service
        self.backend_service = backend_service

        # 백엔드 전송은 stream ingest 경로를 막지 않도록 background queue에서 처리
        self.backend_intent_queue: Queue[BackendInferenceSchema] = Queue(
            maxsize=BACKEND_QUEUE_MAX_SIZE
        )

        self.backend_worker = Thread(
            target=self._backend_intent_worker,
            daemon=True,
        )
        self.backend_worker.start()

    def handle_emg_window(self, msg: EmgWindowMessage) -> EmgWindowAck:
        state = self.device_mode_registry.get(msg.deviceId)

        if state.mode == DeviceMode.IDLE:
            # calibration도, session도 진행 중이 아님 → 무시
            # IDLE인 device는 lock 자체가 없으므로 touch하지 않는다.
            return EmgWindowAck(
                success=True,
                message="device idle, dropped",
                data=EmgWindowAckData(
                    deviceId=msg.deviceId,
                    mode=DeviceMode.IDLE,
                    activeId=None,
                    acceptedSequenceNumber=msg.sequenceNumber,
                    bufferedWindowCount=0,
                ),
            )

        # 활성 device가 살아있다는 신호 → lock 누수 방어 sweeper가 안 풀도록 갱신
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

        buffered = sum(len(buf) for buf in updated.stepBuffers.values())

        return EmgWindowAck(
            success=True,
            message="calibration window appended",
            data=EmgWindowAckData(
                deviceId=msg.deviceId,
                mode=DeviceMode.CALIBRATION,
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

        try:
            inference_payload = self._run_inference_if_ready(
                msg=msg,
                session_id=session_id,
                buffered_window_count=updated.bufferedWindowCount,
            )

            if inference_payload is not None:
                self.session_service.update_last_intent(
                    session_id=session_id,
                    intent=inference_payload.intent,
                )

                self._enqueue_backend_intent(inference_payload)

        except ValueError as e:
            # 추론 / metric / queue 적재 실패해도 WebSocket 수신 자체는 계속 유지
            print(
                f"Failed to process inference result "
                f"(sessionId={session_id}, sequenceNumber={msg.sequenceNumber}): {e}"
            )

        except Exception as e:
            # 예상하지 못한 에러도 ack 흐름을 끊지 않음
            print(
                f"Unexpected inference pipeline error "
                f"(sessionId={session_id}, sequenceNumber={msg.sequenceNumber}): {e}"
            )

        return EmgWindowAck(
            success=True,
            message="session window appended",
            data=EmgWindowAckData(
                deviceId=msg.deviceId,
                mode=DeviceMode.SESSION,
                activeId=session_id,
                acceptedSequenceNumber=msg.sequenceNumber,
                bufferedWindowCount=updated.bufferedWindowCount,
            ),
        )

    def _run_inference_if_ready(
        self,
        msg: EmgWindowMessage,
        session_id: str,
        buffered_window_count: int,
    ) -> Optional[BackendInferenceSchema]:
        # 최소 window 개수보다 적으면 아직 추론하지 않음
        if buffered_window_count < INFERENCE_MIN_WINDOW_COUNT:
            return None

        # 매 window마다 추론하지 않고, 일정 간격마다 추론
        if buffered_window_count % INFERENCE_INTERVAL != 0:
            return None

        session = self.session_service.get_session(session_id)

        recent_windows = self.session_service.get_recent_windows(
            session_id=session_id,
            count=INFERENCE_WINDOW_COUNT,
        )

        if not recent_windows:
            return None

        merged_channels = self._merge_recent_windows_channels(recent_windows)

        processed_channels = preprocess_channels(
            channels=merged_channels,
            calibration=session.calibration,
        )

        feature_vector = extract_feature_vector(processed_channels)

        inference_result = predict_intent(
            feature_vector=feature_vector,
            calibration=session.calibration,
        )

        metrics = calculate_signal_metrics(
            processed_channels=processed_channels,
            calibration=session.calibration,
        )

        return BackendInferenceSchema(
            sessionId=session.sessionId,
            sequenceNumber=msg.sequenceNumber,
            emgDeviceId=msg.deviceId,
            timestamp=msg.timestamp,
            intent=inference_result.predicted_intent,
            confidence=inference_result.confidence,
            fatigueScore=metrics["fatigueScore"],
            signalQuality=metrics["signalQuality"],
        )

    def _enqueue_backend_intent(
        self,
        payload: BackendInferenceSchema,
    ) -> None:
        """
        백엔드 전송 job을 queue에 적재한다.
        stream/window ingest 경로가 막히지 않도록 put_nowait을 사용한다.
        """
        try:
            self.backend_intent_queue.put_nowait(payload)

        except Full:
            # queue가 가득 찬 경우 stream ack를 막지 않고 해당 payload만 drop
            print(
                f"Backend intent queue is full. Dropped payload "
                f"(sessionId={payload.sessionId}, sequenceNumber={payload.sequenceNumber})"
            )

    def _backend_intent_worker(self) -> None:
        """
        백그라운드에서 queue에 쌓인 inference payload를 백엔드로 전송한다.
        전송 실패 시 retry하고, 최종 실패해도 stream ingest 경로에는 영향 주지 않는다.
        """
        while True:
            try:
                payload = self.backend_intent_queue.get(timeout=1.0)

            except Empty:
                continue

            try:
                self._send_backend_intent_with_retry(payload)

            finally:
                self.backend_intent_queue.task_done()

    def _send_backend_intent_with_retry(
        self,
        payload: BackendInferenceSchema,
    ) -> None:
        for attempt in range(1, BACKEND_SEND_RETRY_COUNT + 1):
            try:
                sent = self.backend_service.send_intent(payload)
                print(
                    f"Backend intent sent: {sent} "
                    f"(sessionId={payload.sessionId}, sequenceNumber={payload.sequenceNumber})"
                )
                return

            except ValueError as e:
                print(
                    f"Failed to send intent to backend "
                    f"(attempt={attempt}, sessionId={payload.sessionId}, "
                    f"sequenceNumber={payload.sequenceNumber}): {e}"
                )

                if attempt < BACKEND_SEND_RETRY_COUNT:
                    time.sleep(BACKEND_SEND_RETRY_DELAY_SECONDS)

            except Exception as e:
                print(
                    f"Unexpected backend send error "
                    f"(attempt={attempt}, sessionId={payload.sessionId}, "
                    f"sequenceNumber={payload.sequenceNumber}): {e}"
                )

                if attempt < BACKEND_SEND_RETRY_COUNT:
                    time.sleep(BACKEND_SEND_RETRY_DELAY_SECONDS)

    def _merge_recent_windows_channels(
        self,
        windows: list[SessionWindow],
    ) -> list[ChannelWindow]:
        channel_samples: dict[int, list[int]] = {}

        for window in windows:
            for channel in window.channels:
                if channel.channelIndex not in channel_samples:
                    channel_samples[channel.channelIndex] = []

                channel_samples[channel.channelIndex].extend(channel.samples)

        return [
            ChannelWindow(
                channelIndex=channel_index,
                samples=samples,
            )
            for channel_index, samples in sorted(channel_samples.items())
        ]