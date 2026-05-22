import numpy as np


def _flatten_signals(processed_channels: dict[int, list[float]]) -> np.ndarray:
    # 채널 데이터가 없으면 계산 불가
    if not processed_channels:
        raise ValueError("processed channels must not be empty")

    all_samples = []

    # 채널 번호 순서대로 전체 신호를 하나로 합침
    for channel_index in sorted(processed_channels.keys()):
        signal = processed_channels[channel_index]

        if not signal:
            raise ValueError(f"signal for channel {channel_index} must not be empty")

        all_samples.extend(signal)

    return np.array(all_samples, dtype=float)


def _clamp_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _get_fatigue_baseline(calibration) -> float | None:
    if calibration is None:
        return None

    value = getattr(calibration, "fatigueBaseline", None)
    if value is None:
        return None

    return float(value)


def _get_calibration_signal_quality(calibration) -> float | None:
    if calibration is None:
        return None

    value = getattr(calibration, "signalQuality", None)
    if value is None:
        return None

    return float(value)


def calculate_signal_quality(
    processed_channels: dict[int, list[float]],
    calibration=None,
) -> float:
    """
    signalQuality 계산

    calibration 유무에 따라 분기 처리
    - calibration 있음: 신호처리된 데이터의 mean+std 를 calibration 당시 기준값과 비교
      → 비율이 1에 가까울수록 calibration 때와 동일한 품질
    - calibration 없음: 채널별 변동계수(CV)로 신호 안정성 추정
      → CV가 낮을수록(=신호가 안정적일수록) 높은 품질
    """
    signal = _flatten_signals(processed_channels)
    mean_value = float(np.mean(signal))
    std_value = float(np.std(signal))

    calibration_quality = _get_calibration_signal_quality(calibration)

    if calibration_quality is not None and calibration_quality > 0:
        # calibration 기준값 대비 현재 신호 품질 비율
        raw_current = mean_value + std_value
        quality = raw_current / calibration_quality
        quality = _clamp_score(quality)
        return round(quality, 4)

    # calibration 없을 때: 변동계수(CV) 기반 품질 추정
    # mean이 0이면 신호 자체가 없으므로 품질 0
    if mean_value <= 0:
        return 0.0

    cv = std_value / mean_value  # coefficient of variation
    # CV가 0이면 완벽한 신호 → 1.0, CV가 클수록 품질 낮음
    # CV=1 이상이면 품질 ≈ 0
    quality = _clamp_score(1.0 - cv)
    return round(quality, 4)


def calculate_fatigue_score(
    processed_channels: dict[int, list[float]],
    calibration=None,
) -> float:
    """
    fatigueScore 계산

    calibration 유무에 따라 분기 처리
    - calibration 있음: 현재 평균 활성도를 calibration 기준값과 비교
      → 1에 가까우면 정상, 낮아지면 피로 누적
    - calibration 없음: 원시 활성도를 그대로 0~1로 clamp
      (calibration 없이는 절대 기준이 없으므로 참고용)
    """
    signal = _flatten_signals(processed_channels)
    current_fatigue = float(np.mean(signal))

    fatigue_baseline = _get_fatigue_baseline(calibration)

    if fatigue_baseline is not None and fatigue_baseline > 0:
        # calibration 기준값 대비 비율
        fatigue_score = current_fatigue / fatigue_baseline
        fatigue_score = _clamp_score(fatigue_score)
        return round(fatigue_score, 4)

    # calibration 없을 때: raw 값 자체가 클 수 있으므로
    # 의미 있는 범위로 매핑 (신호처리 후 평균 활성도 기준)
    fatigue_score = _clamp_score(current_fatigue)
    return round(fatigue_score, 4)


def calculate_signal_metrics(
    processed_channels: dict[int, list[float]],
    calibration=None,
) -> dict[str, float]:
    """
    백엔드 응답에 사용할 신호 분석 지표를 한 번에 계산
    """
    return {
        "fatigueScore": calculate_fatigue_score(processed_channels, calibration),
        "signalQuality": calculate_signal_quality(processed_channels, calibration),
    }
