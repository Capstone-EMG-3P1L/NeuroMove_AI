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
    현재 기준:
    - mean + std로 현재 신호 품질 계산
    - calibration.signalQuality가 있으면 기준 품질과 현재 품질을 함께 반영
    - calibration ratio 계산 후 최종 결과만 0.0 ~ 1.0 범위로 제한
    """

    signal = _flatten_signals(processed_channels)
    mean_value = float(np.mean(signal))
    std_value = float(np.std(signal))

    # calibration ratio 계산 전에 clamp하지 않고 raw 값을 유지
    raw_current = mean_value + std_value
    calibration_quality = _get_calibration_signal_quality(calibration)

    if calibration_quality is None or calibration_quality <= 0:
        current_quality = _clamp_score(raw_current)
        return round(current_quality, 4)

    # calibration 당시 품질과 현재 품질을 비교해서 보정
    quality = raw_current / calibration_quality
    # 최종 결과만 0~1 범위로 제한
    quality = _clamp_score(quality)

    return round(quality, 4)


def calculate_fatigue_score(
    processed_channels: dict[int, list[float]],
    calibration=None,
) -> float:
    """
    fatigueScore 계산

    현재 기준:
    - 전처리된 신호의 평균 활성도를 현재 fatigue 지표로 사용
    - calibration.fatigueBaseline이 있으면 기준값 대비 비율로 계산
    - 값은 0.0 ~ 1.0 범위로 반환

    나중에 실제 EMG 데이터가 들어오면
    MNF / MDF 같은 주파수 기반 피로도 지표로 교체 가능
    """
    signal = _flatten_signals(processed_channels)

    current_fatigue = float(np.mean(signal))
    fatigue_baseline = _get_fatigue_baseline(calibration)

    if fatigue_baseline is None or fatigue_baseline <= 0:
        fatigue_score = current_fatigue
    else:
        fatigue_score = current_fatigue / fatigue_baseline

    fatigue_score = _clamp_score(fatigue_score)

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