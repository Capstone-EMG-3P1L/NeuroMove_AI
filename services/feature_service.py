import numpy as np

def calculate_mav(signal: list[float]) -> float:
    # 평균 절댓값(MAV) 계산
    # 신호의 평균 활성 크기
    return float(np.mean(np.abs(signal)))

def calculate_rms(signal: list[float]) -> float:
    # RMS 계산
    # 큰 신호값에 더 민감한 에너지 지표
    return float(np.sqrt(np.mean(np.square(signal))))

def extract_feature_vector(processed_channels: dict[int, list[float]]) -> list[float]:
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