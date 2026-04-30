import numpy as np

MAX_SUPPORTED_CHANNELS = 3


def _get_channel_index(channel) -> int:
    # 채널 객체에서 channel index 추출
    channel_index = getattr(channel, "channel_index", None)

    # channel_index가 없으면 channelIndex 사용
    if channel_index is None:
        channel_index = getattr(channel, "channelIndex", None)

    # 둘 다 없으면 예외 발생
    if channel_index is None:
        raise ValueError("채널 인덱스가 없습니다")

    return channel_index


def _to_float_array(samples: list[int]) -> np.ndarray:
    # samples가 비어있으면 예외 발생
    if not samples:
        raise ValueError("samples가 비어 있습니다")

    # numpy float 배열로 변환
    return np.array(samples, dtype=float)


def _get_baseline_stats(calibration, channel_index: int) -> tuple[float, float] | None:
    """
    channelIndex 기준으로 calibration baseline mean/std를 가져온다.

    현재 센서 구성은 3채널 기준:
    channelIndex 0 -> ch1Mean / ch1Std
    channelIndex 1 -> ch2Mean / ch2Std
    channelIndex 2 -> ch3Mean / ch3Std

    calibration이 없으면 None을 반환해서 기존 remove_dc_offset 방식으로 fallback
    calibration이 있는데 지원하지 않는 채널이거나 baseline 필드가 없으면 예외를 발생시킴
    """
    if calibration is None:
        return None

    baseline = getattr(calibration, "baseline", None)
    if baseline is None:
        raise ValueError("calibration baseline 정보가 없습니다")

    if channel_index < 0 or channel_index >= MAX_SUPPORTED_CHANNELS:
        raise ValueError(
            f"지원하지 않는 channelIndex입니다: {channel_index}. "
            f"허용 범위는 0부터 {MAX_SUPPORTED_CHANNELS - 1}까지입니다"
        )

    channel_number = channel_index + 1
    mean_field = f"ch{channel_number}Mean"
    std_field = f"ch{channel_number}Std"

    if not hasattr(baseline, mean_field) or not hasattr(baseline, std_field):
        raise ValueError(
            f"channelIndex {channel_index}에 해당하는 calibration baseline 필드가 없습니다: "
            f"{mean_field}, {std_field}"
        )

    mean_value = getattr(baseline, mean_field)
    std_value = getattr(baseline, std_field)

    if mean_value is None or std_value is None:
        raise ValueError(
            f"channelIndex {channel_index}의 calibration baseline 값이 비어 있습니다: "
            f"{mean_field}, {std_field}"
        )

    return float(mean_value), float(std_value)


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    # 평균값 제거
    return signal - np.mean(signal)


def apply_calibration_baseline(
    signal: np.ndarray,
    calibration,
    channel_index: int,
) -> np.ndarray:
    # calibration baseline이 있으면 채널별 baseline 기준으로 보정
    stats = _get_baseline_stats(calibration, channel_index)

    if stats is None:
        return remove_dc_offset(signal)

    baseline_mean, baseline_std = stats

    # baseline std가 0이면 나눌 수 없으므로 mean만 제거
    if baseline_std == 0:
        return signal - baseline_mean

    return (signal - baseline_mean) / baseline_std


def rectify_signal(signal: np.ndarray) -> np.ndarray:
    # 절댓값 처리
    return np.abs(signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    # 최대 절댓값 추출
    max_value = np.max(np.abs(signal))

    # 최대값이 0이면 그대로 반환
    if max_value == 0:
        return signal

    # 0~1 범위로 정규화
    return signal / max_value


def preprocess_signal(
    samples: list[int],
    normalize: bool = True,
    calibration=None,
    channel_index: int | None = None,
) -> list[float]:
    # raw samples를 numpy 배열로 변환
    signal = _to_float_array(samples)

    # calibration과 channel_index가 있으면 baseline 기준으로 보정
    # 없으면 기존 방식대로 window 평균 제거
    if calibration is not None and channel_index is not None:
        signal = apply_calibration_baseline(signal, calibration, channel_index)
    else:
        signal = remove_dc_offset(signal)

    # 절댓값 처리
    signal = rectify_signal(signal)

    # 정규화 옵션이 켜져 있으면 수행
    if normalize:
        signal = normalize_signal(signal)

    # list 형태로 반환
    return signal.tolist()


def preprocess_channels(
    channels,
    normalize: bool = True,
    calibration=None,
) -> dict[int, list[float]]:
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
            calibration=calibration,
            channel_index=channel_index,
        )

    return processed_channels