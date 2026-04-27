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


def calculate_signal_quality(processed_channels: dict[int, list[float]]) -> float:
    """
    signalQuality 계산

    현재는 간단한 기준으로 계산:
    - 신호가 너무 약하면 품질 낮음
    - 신호 변화량이 적당히 있으면 품질 높음
    - 값은 0.0 ~ 1.0 범위로 반환
    """
    signal = _flatten_signals(processed_channels)

    mean_value = float(np.mean(signal))
    std_value = float(np.std(signal))

    # 평균 활성도와 표준편차를 이용한 간단한 품질 점수
    quality = mean_value + std_value

    # 0~1 범위로 제한
    quality = max(0.0, min(quality, 1.0))

    return round(quality, 4)


def calculate_fatigue_score(processed_channels: dict[int, list[float]]) -> float:
    """
    fatigueScore 계산

    현재는 실제 주파수 분석 기반 피로도 계산 전 임시 지표:
    - 전처리된 신호의 평균 활성도가 높을수록 피로도가 높다고 가정
    - 값은 0.0 ~ 1.0 범위로 반환

    나중에 실제 EMG 데이터가 들어오면
    MNF / MDF 같은 주파수 기반 피로도 지표로 교체 가능
    """
    signal = _flatten_signals(processed_channels)

    fatigue_score = float(np.mean(signal))

    # 0~1 범위로 제한
    fatigue_score = max(0.0, min(fatigue_score, 1.0))

    return round(fatigue_score, 4)


def calculate_signal_metrics(processed_channels: dict[int, list[float]]) -> dict[str, float]:
    """
    백엔드 응답에 사용할 신호 분석 지표를 한 번에 계산
    """
    return {
        "fatigueScore": calculate_fatigue_score(processed_channels),
        "signalQuality": calculate_signal_quality(processed_channels),
    }