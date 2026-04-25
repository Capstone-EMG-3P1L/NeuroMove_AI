import os
import pickle

from schemas.stream_schema import InferenceResultSchema


# 모델 파일 경로
MODEL_PATH = "models/intent_model.pkl"

# 모델 버전
MODEL_VERSION = "mock-v2"


def load_model():
    # pkl 파일이 존재하면 모델 로드
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        return model

    # 없으면 None 반환
    return None


def _mock_predict(feature_vector: list[float]) -> tuple[str, float]:
    # feature가 없으면 UNKNOWN
    if not feature_vector:
        return "UNKNOWN", 0.0

    channel_scores = []

    # 채널별 MAV/RMS 평균 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    max_score = max(channel_scores)

    # 약한 신호는 STOP
    if max_score < 0.15:
        return "STOP", round(max_score, 4)

    max_channel = channel_scores.index(max_score)

    if max_channel == 0:
        intent = "LEFT"
    elif max_channel == 1:
        intent = "RIGHT"
    elif max_channel == 2:
        intent = "FORWARD"
    else:
        intent = "UNKNOWN"

    return intent, round(min(max_score, 1.0), 4)


def predict_intent(feature_vector: list[float]) -> InferenceResultSchema:
    # 모델 로드 시도
    model = load_model()

    # 실제 모델이 있으면 predict 수행
    if model is not None:
        prediction = model.predict([feature_vector])[0]

        # predict_proba 지원 모델이면 confidence 계산
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([feature_vector])[0]
            confidence = round(float(max(probabilities)), 4)
        else:
            confidence = 0.8

        return InferenceResultSchema(
            predicted_intent=prediction,
            confidence=confidence,
            feature_vector=feature_vector,
            model_version="trained-model",
        )

    # 모델 없으면 mock 예측 수행
    intent, confidence = _mock_predict(feature_vector)

    return InferenceResultSchema(
        predicted_intent=intent,
        confidence=confidence,
        feature_vector=feature_vector,
        model_version=MODEL_VERSION,
    )