from schemas.stream_schema import InferenceResultSchema

# 간단한 mock 모델로 feature vector 기반 intent 예측
MODEL_VERSION = "mock-v1"


def predict_intent(feature_vector: list[float]) -> InferenceResultSchema:
    # feature가 없으면 예측 불가
    if not feature_vector:
        return InferenceResultSchema(
            predicted_intent="UNKNOWN",
            confidence=0.0,
            feature_vector=[],
            model_version=MODEL_VERSION,
        )

    # feature 순서: [ch0_mav, ch0_rms, ch1_mav, ch1_rms, ...]
    channel_scores = []

    # 채널별 MAV/RMS 평균 점수 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    # 가장 강한 채널 점수 확인
    max_score = max(channel_scores)

    # 신호가 너무 약하면 STOP 처리
    if max_score < 0.15:
        intent = "STOP"
    else:
        # 가장 강한 채널 기준으로 임시 intent 결정
        max_channel = channel_scores.index(max_score)

        if max_channel == 0:
            intent = "LEFT"
        elif max_channel == 1:
            intent = "RIGHT"
        elif max_channel == 2:
            intent = "FORWARD"
        else:
            intent = "UNKNOWN"

    # mock confidence는 가장 강한 채널 점수 사용
    confidence = min(max_score, 1.0)

    # stream_schema.py의 InferenceResultSchema 형식으로 반환
    return InferenceResultSchema(
        predicted_intent=intent,
        confidence=round(confidence, 4),
        feature_vector=feature_vector,
        model_version=MODEL_VERSION,
    )