import numpy as np


def _to_array(signal: list[float]) -> np.ndarray:
    # signal이 비어 있으면 feature 계산 불가
    if not signal:
        raise ValueError("signal must not be empty")

    return np.array(signal, dtype=float)


def calculate_mav(signal: list[float]) -> float:
    # 평균 절댓값(MAV) 계산
    # 신호의 평균 활성 크기
    signal_array = _to_array(signal)
    return float(np.mean(np.abs(signal_array)))


def calculate_rms(signal: list[float]) -> float:
    # RMS 계산
    # 큰 신호값에 더 민감한 에너지 지표
    signal_array = _to_array(signal)
    return float(np.sqrt(np.mean(np.square(signal_array))))


def extract_feature_vector(processed_channels: dict[int, list[float]]) -> list[float]:
    # 채널 데이터가 없으면 feature 생성 불가
    if not processed_channels:
        raise ValueError("processed channels must not be empty")

    # 최종 feature 저장 리스트
    feature_vector = []

    # 채널 번호 순서대로 처리
    for channel_index in sorted(processed_channels.keys()):
        signal = processed_channels[channel_index]

        # MAV 추가
        feature_vector.append(calculate_mav(signal))

        # RMS 추가
        feature_vector.append(calculate_rms(signal))

    # 예:
    # [ch0_mav, ch0_rms, ch1_mav, ch1_rms, ...]

    return feature_vector