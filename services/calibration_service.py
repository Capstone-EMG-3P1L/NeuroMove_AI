from schemas.calibration_schema import *;
from storage.calibration_store import *;
from storage.device_mode_registry import DeviceModeRegistry
import time
from typing import Dict, List

import numpy as np

# [수정] calibration finish 시 신호처리를 적용하기 위해 import
from services.signal_processing_service import remove_dc_offset, rectify_signal


class CalibrationService:
    def __init__(
        self,
        calibration_store: CalibrationSessionStore,
        device_mode_registry: DeviceModeRegistry,
    ):
        self.calibration_store = calibration_store
        self.device_mode_registry = device_mode_registry

    def start_calibration(self, request: CalibrationStartRequest,) -> CalibrationStartResponse:
        if self.calibration_store.exists(request.calibrationSessionId):
            raise ValueError("calibration session already exists")

        # 같은 deviceId 가 이미 calibration / session 중이면 거절(요건 2: 동시 진행 금지)
        self.device_mode_registry.set_calibration(
            request.deviceId,
            request.calibrationSessionId,
        )

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

    def append_calibration_data(
        self,
        request: CalibrationDataRequest,
    ) -> CalibrationSession:
        """
        WebSocket 으로 들어온 EMG window 한 개를 calibration step buffer 에 누적.

        StreamService 가 deviceMode == CALIBRATION 으로 라우팅한 후 호출하는,
        calibration 도메인의 단일 ingest 엔트리포인트.

        검증 실패 시 ValueError 를 던진다 (StreamService 가 catch 해 ack 로 변환).
        """
        session = self.calibration_store.get_session(request.calibrationSessionId)
        if session is None:
            raise ValueError("calibration session not found")

        if session.status == CalibrationStatus.COMPLETED:
            raise ValueError("calibration session already completed")

        if session.deviceId != request.deviceId:
            raise ValueError("deviceId mismatch with active calibration session")

        # 시퀀스 넘버가 이전보다 작거나 같으면 안됨 -> 증가만 하면 OK
        # TODO: 실제 시스템은 "연속성 체크 + gap 감지" 로 변경
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

        return updated_session

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

        # [수정] 버퍼에 쌓인 raw 데이터를 신호처리(DC 제거 + 정류)한 뒤 실제 calibration 결과 계산
        result = self._compute_calibration_result(session)

        completed_session = self.calibration_store.save_result(
            request.calibrationSessionId,
            result,
            completed_at,
        )
        if completed_session is None:
            raise ValueError("failed to save calibration result")

        # calibration 끝났으니 device 를 IDLE 로 풀어준다.
        # → 같은 deviceId 로 session 시작 가능해짐
        self.device_mode_registry.clear(completed_session.deviceId)

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

    # ──────────────────────────────────────────────
    # [수정] 신호처리 기반 calibration 결과 계산
    # ──────────────────────────────────────────────

    def _collect_channel_samples(
        self,
        buffers: List[CalibrationDataRequest],
    ) -> Dict[int, List[int]]:
        """step buffer에서 채널별 raw sample을 하나로 모은다."""
        channel_samples: Dict[int, List[int]] = {}
        for window in buffers:
            for ch in window.channels:
                if ch.channelIndex not in channel_samples:
                    channel_samples[ch.channelIndex] = []
                channel_samples[ch.channelIndex].extend(ch.samples)
        return channel_samples

    def _apply_zscore_rectify(
        self,
        raw_samples: np.ndarray,
        baseline_mean: float,
        baseline_std: float,
    ) -> np.ndarray:
        """
        [수정] 세션 중 preprocess_channels가 하는 것과 동일한 처리
        z-score 정규화(baseline 기준) → 정류 — 세션과 동일 스케일 보장
        """
        signal = raw_samples.astype(float)
        if baseline_std > 0:
            signal = (signal - baseline_mean) / baseline_std
        else:
            signal = signal - baseline_mean
        signal = np.abs(signal)  # rectify
        return signal

    def _compute_step_activation_zscore(
        self,
        buffers: List[CalibrationDataRequest],
        baseline_stats: Dict[int, Dict[str, float]],
    ) -> float:
        """
        [수정] step buffer 데이터를 z-score + 정류 처리한 뒤 평균 활성도 계산
        세션 중 추론 파이프라인과 동일한 스케일
        """
        if not buffers:
            return 0.0

        channel_samples = self._collect_channel_samples(buffers)
        activations = []
        for ch_idx in sorted(channel_samples.keys()):
            if ch_idx not in baseline_stats:
                continue
            raw = np.array(channel_samples[ch_idx], dtype=float)
            processed = self._apply_zscore_rectify(
                raw,
                baseline_stats[ch_idx]["mean"],
                baseline_stats[ch_idx]["std"],
            )
            activations.append(float(np.mean(processed)))

        return float(np.mean(activations)) if activations else 0.0

    def _compute_calibration_result(
        self,
        session: CalibrationSession,
    ) -> CalibrationResult:
        """
        [수정] calibration 종료 시 신호처리된 데이터로 baseline / threshold 계산

        처리 순서:
        1) REST raw 데이터에서 채널별 mean/std 추출 → baseline (세션 중 z-score 정규화 기준)
        2) 이 baseline으로 모든 step 데이터를 z-score + 정류 → 세션과 동일 스케일
        3) z-score 스케일 기준으로 threshold / fatigue / signalQuality 계산
        """
        # ── 1. REST 단계에서 채널별 baseline (raw mean/std) 계산 ──
        rest_buffers = session.stepBuffers[CalibrationStep.REST]
        rest_channel_samples = self._collect_channel_samples(rest_buffers)

        baseline_stats: Dict[int, Dict[str, float]] = {}
        for ch_idx in range(3):
            raw = np.array(rest_channel_samples.get(ch_idx, [0]), dtype=float)
            baseline_stats[ch_idx] = {
                "mean": float(np.mean(raw)),
                "std": float(np.std(raw)),
            }

        # ── 2. 각 step별 평균 활성도 (z-score + 정류 — 세션과 동일 스케일) ──
        rest_activation = self._compute_step_activation_zscore(
            rest_buffers, baseline_stats
        )
        left_activation = self._compute_step_activation_zscore(
            session.stepBuffers[CalibrationStep.LEFT], baseline_stats
        )
        right_activation = self._compute_step_activation_zscore(
            session.stepBuffers[CalibrationStep.RIGHT], baseline_stats
        )
        stop_activation = self._compute_step_activation_zscore(
            session.stepBuffers[CalibrationStep.STOP], baseline_stats
        )

        # ── 3. activationThreshold ──
        # REST와 가장 약한 능동 동작의 중간값 (z-score 스케일)
        min_active = min(left_activation, right_activation)
        activation_threshold = (rest_activation + min_active) / 2

        # ── 4. intentThresholds (z-score 스케일) ──
        intent_thresholds = {
            CalibrationStep.LEFT: round(left_activation * 0.5, 4),
            CalibrationStep.RIGHT: round(right_activation * 0.5, 4),
            CalibrationStep.STOP: round(stop_activation * 0.5, 4),
        }

        # ── 5. fatigueBaseline (z-score 스케일) ──
        # 세션 중 signal_metric_service가 현재 활성도 / fatigueBaseline 로 피로도 계산
        fatigue_baseline = (left_activation + right_activation) / 2

        # ── 6. signalQuality (z-score 스케일) ──
        # REST 데이터를 z-score + 정류한 뒤 mean + std 를 기준 품질값으로 저장
        rest_zscored_all = []
        for ch_idx in sorted(rest_channel_samples.keys()):
            if ch_idx not in baseline_stats:
                continue
            raw = np.array(rest_channel_samples[ch_idx], dtype=float)
            processed = self._apply_zscore_rectify(
                raw,
                baseline_stats[ch_idx]["mean"],
                baseline_stats[ch_idx]["std"],
            )
            rest_zscored_all.extend(processed.tolist())

        rest_signal = np.array(rest_zscored_all, dtype=float)
        signal_quality = float(np.mean(rest_signal) + np.std(rest_signal))

        return CalibrationResult(
            baseline=BaselineResult(
                ch1Mean=round(baseline_stats[0]["mean"], 4),
                ch1Std=round(baseline_stats[0]["std"], 4),
                ch2Mean=round(baseline_stats[1]["mean"], 4),
                ch2Std=round(baseline_stats[1]["std"], 4),
                ch3Mean=round(baseline_stats[2]["mean"], 4),
                ch3Std=round(baseline_stats[2]["std"], 4),
            ),
            activationThreshold=round(activation_threshold, 4),
            intentThresholds=intent_thresholds,
            fatigueBaseline=round(fatigue_baseline, 4),
            signalQuality=round(signal_quality, 4),
        )
