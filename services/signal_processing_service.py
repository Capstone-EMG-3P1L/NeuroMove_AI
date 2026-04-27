import numpy as np


def _get_channel_index(channel) -> int:
    # 채널 객체에서 channel index 추출
    channel_index = getattr(channel, "channel_index", None)

    # channel_index가 없으면 channelIndex 사용
    if channel_index is None:
        channel_index = getattr(channel, "channelIndex", None)

    # 둘 다 없으면 예외 발생
    if channel_index is None:
        raise ValueError("channel index is missing")

    return channel_index


def _to_float_array(samples: list[int]) -> np.ndarray:
    # samples가 비어있으면 예외 발생
    if not samples:
        raise ValueError("samples must not be empty")

    # numpy float 배열로 변환
    return np.array(samples, dtype=float)


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    # 평균값 제거
    return signal - np.mean(signal)


def rectify_signal(signal: np.ndarray) -> np.ndarray:
    # 절댓값 처리
    return np.abs(signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    # 최대값 추출
    max_value = np.max(np.abs(signal))

    # 최대값이 0이면 그대로 반환
    if max_value == 0:
        return signal

    # 0~1 범위로 정규화
    return signal / max_value


def preprocess_signal(samples: list[int], normalize: bool = True) -> list[float]:
    # raw samples를 numpy 배열로 변환
    signal = _to_float_array(samples)

    # 평균 제거
    signal = remove_dc_offset(signal)

    # 절댓값 처리
    signal = rectify_signal(signal)

    # 정규화 옵션이 켜져 있으면 수행
    if normalize:
        signal = normalize_signal(signal)

    # list 형태로 반환
    return signal.tolist()


def preprocess_channels(channels, normalize: bool = True) -> dict[int, list[float]]:
    # 채널별 전처리 결과 저장
    processed_channels = {}

    # 모든 채널 반복 처리
    for channel in channels:
        # 채널 번호 추출
        channel_index = _get_channel_index(channel)

        # 채널 samples 전처리 후 저장
        processed_channels[channel_index] = preprocess_signal(
            samples=channel.samples,
            normalize=normalize,
        )

    return processed_channels