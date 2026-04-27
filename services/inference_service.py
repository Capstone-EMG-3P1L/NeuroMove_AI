import os
import pickle
from functools import lru_cache

from schemas.stream_schema import InferenceResultSchema


MODEL_PATH = "models/intent_model.pkl"
MODEL_VERSION = "mock-v4"

# 현재 프로젝트에서 사용하는 intent 라벨
# LEFT  : 왼쪽으로 고개 돌림
# RIGHT : 오른쪽으로 고개 돌림
# REST  : 힘 빼고 가만히 있음 / 알 수 없는 신호
# STOP  : 힘줘서 멈춤
VALID_INTENTS = {"LEFT", "RIGHT", "REST", "STOP"}


@lru_cache(maxsize=1)
def load_model():
    # 모델 파일이 있으면 한 번만 로드해서 캐싱
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    return None


def _mock_predict(feature_vector: list[float]) -> tuple[str, float]:
    # feature가 없으면 REST
    if not feature_vector:
        return "REST", 0.0

    channel_scores = []

    # 채널별 MAV/RMS 평균 계산
    for i in range(0, len(feature_vector), 2):
        mav = feature_vector[i]
        rms = feature_vector[i + 1] if i + 1 < len(feature_vector) else 0.0
        channel_scores.append((mav + rms) / 2)

    max_score = max(channel_scores)

    # 전체 신호가 약하면 REST
    # REST = 힘 빼고 가만히 있는 상태 / 알 수 없는 신호
    if max_score < 0.15:
        return "REST", round(max_score, 4)

    # 전체적으로 신호가 강하면 STOP으로 처리
    # STOP = 특정 방향이 아니라 힘줘서 멈추는 상태
    avg_score = sum(channel_scores) / len(channel_scores)

    if avg_score >= 0.5:
        return "STOP", round(min(avg_score, 1.0), 4)

    max_channel = channel_scores.index(max_score)

    # 현재는 좌/우 고개 방향만 사용
    # ch0이 가장 강하면 LEFT
    # ch1이 가장 강하면 RIGHT
    intent_map = {
        0: "LEFT",
        1: "RIGHT",
    }

    # 매핑되지 않는 채널이면 REST 처리
    intent = intent_map.get(max_channel, "REST")

    return intent, round(min(max_score, 1.0), 4)


def predict_intent(feature_vector: list[float]) -> InferenceResultSchema:
    model = load_model()

    if model is not None:
        prediction = str(model.predict([feature_vector])[0])

        # 모델이 허용되지 않은 라벨을 예측하면 REST 처리
        if prediction not in VALID_INTENTS:
            prediction = "REST"

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

    intent, confidence = _mock_predict(feature_vector)

    return InferenceResultSchema(
        predicted_intent=intent,
        confidence=confidence,
        feature_vector=feature_vector,
        model_version=MODEL_VERSION,
    )